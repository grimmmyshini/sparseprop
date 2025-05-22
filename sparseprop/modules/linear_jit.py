from concurrent.futures import ThreadPoolExecutor, wait
import ctypes
from llvmlite import ir, binding

from sparseprop.modules.jit_utils import JITOptions

class LinearJIT:
    def __init__(self, jit_options=JITOptions(), name="jit_sparse_linear_forward"):
        self.options = jit_options
        self.unroll_times = jit_options.batch_size // 8

        self.fn_fwd = None
        self.fn_bwd = None

        # Initialize LLVM
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        self.module = ir.Module(name)
        self.opt_level = 3

        target = binding.Target.from_default_triple()
        self.target_machine = target.create_target_machine(
            cpu=binding.get_host_cpu_name(),
            features=binding.get_host_cpu_features().flatten(),
            opt=self.opt_level,
        )
        backing_mod = binding.parse_assembly("")
        self.engine = binding.create_mcjit_compiler(backing_mod, self.target_machine)

        # Common util functions
        self.fma_intr = ir.Function(
            self.module,
            ir.FunctionType(
                ir.VectorType(ir.FloatType(), 8), 3 * [ir.VectorType(ir.FloatType(), 8)]
            ),
            name="llvm.fma.v8f32",
        )

        ############# Printf
        # printf_ty = ir.FunctionType(
        #     ir.IntType(32), [ir.PointerType(ir.IntType(8))], var_arg=True
        # )
        # printf = ir.Function(self.module, printf_ty, name="printf")

        # fmt_str = "%d\n\0"
        # fmt_bytes = bytearray(fmt_str.encode("utf8"))
        # fmt_type = ir.ArrayType(ir.IntType(8), len(fmt_bytes))

        # global_fmt = ir.GlobalVariable(self.module, fmt_type, name="fstr")
        # global_fmt.global_constant = True
        # global_fmt.initializer = ir.Constant(fmt_type, fmt_bytes)
        ############# Printf

    def add_unrolling(self, B):
        # This is a bit tricky because ideally we want to re-jit as less as possible
        # However, if unrolling is added and the batch size changes, then we HAVE to re-jit.
        assert self.options.batch_size == B or not (self.fn_fwd and self.fn_bwd), (
            "Changing batchsize after a function has been jit-ted is not yet supported!"
        )

        self.options.batch_size = B
        self.options.unroll = True
        self.unroll_times = B // 8

    @staticmethod
    def vec_load_arr(builder, arr, idx, name=None):
        vec_float_ty = ir.VectorType(ir.FloatType(), 8)
        if name is not None:
            vec = builder.alloca(vec_float_ty, name=name)
        else:
            vec = builder.alloca(vec_float_ty)
        builder.store(
            builder.load(
                builder.bitcast(builder.gep(arr, [idx]), ir.PointerType(vec_float_ty))
            ),
            vec,
        )
        return vec

    @staticmethod
    def optimize(module, target_machine):
        pb = binding.PassBuilder(
            target_machine, binding.PipelineTuningOptions(speed_level=3, size_level=0)
        )
        pm = pb.getModulePassManager()
        pm.run(module, pb)

    def fwd_fma(self, builder, X, X_idx, O, O_idx, v):
        # __m256 x = _mm256_loadu_ps(X + (idx * B + j));
        x = LinearJIT.vec_load_arr(builder, X, X_idx)
        # __m256 o = _mm256_loadu_ps(O + (i * B + j));
        o = LinearJIT.vec_load_arr(builder, O, O_idx)
        # __m256 r = _mm256_fmadd_ps(x,v,o);
        r = builder.alloca(ir.VectorType(ir.FloatType(), 8))
        fma = builder.call(
            self.fma_intr,
            [builder.load(x), builder.load(v), builder.load(o)],
        )
        builder.store(fma, r)
        # _mm256_storeu_ps(O + (i * B + j), r);
        builder.store(
            builder.load(r),
            builder.bitcast(
                builder.gep(O, [O_idx]),
                ir.PointerType(ir.VectorType(ir.FloatType(), 8)),
            ),
        )

    def fwd_block_loop(self, builder, B, idx, i_val, X, O, v):
        if self.options.unroll and self.options.batch_size:
            for i in range(self.unroll_times):
                j = ir.Constant(ir.IntType(32), i * 8)
                X_idx = builder.add(builder.mul(builder.load(idx), B), j)
                O_idx = builder.add(builder.mul(i_val, B), j)
                self.fwd_fma(builder, X, X_idx, O, O_idx, v)
        else:
            # int j = 0;
            j = builder.alloca(ir.IntType(32))
            builder.store(ir.Constant(ir.IntType(32), 0), j)

            # for(; j < B-7; j+=8){
            block_loop = builder.append_basic_block("block_loop")
            builder.branch(block_loop)
            builder.position_at_start(block_loop)
            j_val = builder.load(j)
            cond3 = builder.icmp_signed(
                "<", j_val, builder.sub(B, ir.Constant(ir.IntType(32), 7))
            )
            with builder.if_then(cond3):
                # __m256 x = _mm256_loadu_ps(X + (idx * B + j));
                # __m256 o = _mm256_loadu_ps(O + (i * B + j));
                # __m256 r = _mm256_fmadd_ps(x,v,o);
                # _mm256_storeu_ps(O + (i * B + j), r);
                X_idx = builder.add(builder.mul(builder.load(idx), B), j_val)
                O_idx = builder.add(builder.mul(i_val, B), j_val)
                self.fwd_fma(builder, X, X_idx, O, O_idx, v)

                builder.store(builder.add(j_val, ir.Constant(ir.IntType(32), 8)), j)
                builder.branch(block_loop)

    def _jit_forward(self, fn_name="sparse_fwd"):
        # void fn(int B, int M, int N_start, int N, int W_nnz, float* X, int* W_idx_N, int* W_idx_M, float* W_val, float* O)
        float_ptr_ty = ir.PointerType(ir.FloatType())
        int_ptr_ty = ir.PointerType(ir.IntType(32))
        int_ty = ir.IntType(32)
        vec_float_ty = ir.VectorType(ir.FloatType(), 8)
        func_ty = ir.FunctionType(
            ir.VoidType(),
            [
                int_ty,
                int_ty,
                int_ty,
                int_ty,
                int_ty,
                float_ptr_ty,
                int_ptr_ty,
                int_ptr_ty,
                float_ptr_ty,
                float_ptr_ty,
            ],
        )

        func = ir.Function(self.module, func_ty, name=fn_name)
        B, M, N_start, N, W_nnz, X, W_idx_N, W_idx_M, W_val, O = func.args

        block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        # for(int i = 0; i < N; i++){
        i = builder.alloca(int_ty, name="i")
        builder.store(N_start, i)

        outer_loop = builder.append_basic_block("outer_loop")
        builder.branch(outer_loop)
        builder.position_at_start(outer_loop)

        i_val = builder.load(i)

        cond = builder.icmp_signed("<", i_val, N)
        with builder.if_then(cond):
            # int k = W_idx_N[i];
            k = builder.alloca(int_ty, name="k")
            builder.store(builder.load(builder.gep(W_idx_N, [i_val])), k)

            # for(; k < W_idx_N[i+1]; k++){
            inner_loop = builder.append_basic_block("inner_loop")
            builder.branch(inner_loop)
            builder.position_at_start(inner_loop)

            k_val = builder.load(k)
            W_idx_N_i_1 = builder.load(
                builder.gep(W_idx_N, [builder.add(i_val, ir.Constant(int_ty, 1))])
            )

            cond2 = builder.icmp_signed("<", k_val, W_idx_N_i_1)
            with builder.if_then(cond2):
                # int idx = W_idx_M[k];
                idx = builder.alloca(int_ty, name="idx")
                builder.store(builder.load(builder.gep(W_idx_M, [k_val])), idx)

                # __m256 v = _mm256_set1_ps(W_val[k]);
                v = builder.alloca(vec_float_ty, name="v")
                v_replaced = builder.insert_element(
                    builder.load(v),
                    builder.load(builder.gep(W_val, [k_val])),
                    ir.Constant(int_ty, 0),
                )
                v_set = builder.shuffle_vector(
                    v_replaced,
                    v_replaced,
                    ir.Constant(ir.VectorType(int_ty, 8), 0),
                )
                builder.store(v_set, v)

                self.fwd_block_loop(builder, B, idx, i_val, X, O, v)

                builder.store(builder.add(k_val, ir.Constant(int_ty, 1)), k)
                builder.branch(inner_loop)

            builder.store(builder.add(i_val, ir.Constant(int_ty, 1)), i)
            builder.branch(outer_loop)
        builder.ret_void()

        llvm_ir = str(self.module)
        mod = binding.parse_assembly(llvm_ir)

        LinearJIT.optimize(mod, self.target_machine)

        self.engine.add_module(mod)
        self.engine.finalize_object()

        fn_ptr = self.engine.get_function_address(fn_name)

        self.fn_fwd = ctypes.CFUNCTYPE(
            None,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        )(fn_ptr)

        ctypes.CDLL(None)

    def bck_fma(self, builder, dLdX, x_idx, X, dLdO, O_idx, v, acc):
        # __m256 dx0 = _mm256_loadu_ps(dLdX + (r * B + k));
        dx0 = LinearJIT.vec_load_arr(builder, dLdX, x_idx)

        # __m256 x0 = _mm256_loadu_ps(X + (r * B + k));
        x0 = LinearJIT.vec_load_arr(builder, X, x_idx)

        # __m256 do0 = _mm256_loadu_ps(dLdO + (i * B + k));
        do0 = LinearJIT.vec_load_arr(builder, dLdO, O_idx)

        # __m256 s0 = _mm256_fmadd_ps(v, do0, dx0);
        s0 = builder.alloca(ir.VectorType(ir.FloatType(), 8))
        fma = builder.call(
            self.fma_intr,
            [builder.load(v), builder.load(do0), builder.load(dx0)],
        )
        builder.store(fma, s0)

        # acc = _mm256_fmadd_ps(do0,x0,acc);
        fma2 = builder.call(
            self.fma_intr,
            [builder.load(do0), builder.load(x0), builder.load(acc)],
        )
        builder.store(fma2, acc)

        # _mm256_storeu_ps(dLdX + (r * B + k), s0);
        builder.store(
            builder.load(s0),
            builder.bitcast(
                builder.gep(dLdX, [x_idx]),
                ir.PointerType(ir.VectorType(ir.FloatType(), 8)),
            ),
        )

    def bck_block_loop(self, builder, B, r, i_val, dLdX, X, dLdO, v, acc):
        if self.options.unroll and self.options.batch_size:
            for i in range(self.unroll_times):
                k = ir.Constant(ir.IntType(32), i * 8)
                x_idx = builder.add(builder.mul(builder.load(r), B), k)
                O_idx = builder.add(builder.mul(i_val, B), k)
                self.bck_fma(builder, dLdX, x_idx, X, dLdO, O_idx, v, acc)
        else:
            # int k = 0;
            k = builder.alloca(ir.IntType(32))
            builder.store(ir.Constant(ir.IntType(32), 0), k)

            # for(; k < B-7; k+=8){
            block_loop = builder.append_basic_block("block_loop")
            builder.branch(block_loop)
            builder.position_at_start(block_loop)

            k_val = builder.load(k)
            cond3 = builder.icmp_signed(
                "<", k_val, builder.sub(B, ir.Constant(ir.IntType(32), 7))
            )

            with builder.if_then(cond3):
                # __m256 dx0 = _mm256_loadu_ps(dLdX + (r * B + k));
                # __m256 x0 = _mm256_loadu_ps(X + (r * B + k));
                # __m256 do0 = _mm256_loadu_ps(dLdO + (i * B + k));
                x_idx = builder.add(builder.mul(builder.load(r), B), k_val)
                O_idx = builder.add(builder.mul(i_val, B), k_val)
                self.bck_fma(builder, dLdX, x_idx, X, dLdO, O_idx, v, acc)

                builder.store(builder.add(k_val, ir.Constant(ir.IntType(32), 8)), k)
                builder.branch(block_loop)

    def _jit_backward(self, fn_name="sparse_bwd"):
        # void fn(int B, int M, int N, int W_nnz, float* X, int* W_idx_N, int* W_idx_M,float* W_val, float* dLdO, float* dLdX, float* dLdW_val)
        float_ptr_ty = ir.PointerType(ir.FloatType())
        int_ptr_ty = ir.PointerType(ir.IntType(32))
        int_ty = ir.IntType(32)
        vec_float_ty = ir.VectorType(ir.FloatType(), 8)
        vec_float_half_ty = ir.VectorType(ir.FloatType(), 4)
        func_ty = ir.FunctionType(
            ir.VoidType(),
            [
                int_ty,
                int_ty,
                int_ty,
                int_ty,
                int_ty,
                float_ptr_ty,
                int_ptr_ty,
                int_ptr_ty,
                float_ptr_ty,
                float_ptr_ty,
                float_ptr_ty,
                float_ptr_ty,
            ],
        )

        func = ir.Function(self.module, func_ty, name=fn_name)
        B, M, N_start, N, W_nnz, X, W_idx_N, W_idx_M, W_val, dLdO, dLdX, dLdW_val = (
            func.args
        )

        block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        # for(int i = 0; i < N; i++){
        i = builder.alloca(int_ty, name="i")
        builder.store(N_start, i)

        outer_loop = builder.append_basic_block("outer_loop")
        builder.branch(outer_loop)
        builder.position_at_start(outer_loop)

        i_val = builder.load(i)

        cond = builder.icmp_signed("<", i_val, N)
        with builder.if_then(cond):
            # for(int j = W_idx_N[i]; j < W_idx_N[i+1]; j++){
            j = builder.alloca(int_ty, name="j")
            builder.store(builder.load(builder.gep(W_idx_N, [i_val])), j)

            inner_loop = builder.append_basic_block("inner_loop")
            builder.branch(inner_loop)
            builder.position_at_start(inner_loop)

            j_val = builder.load(j)
            W_idx_N_i_1 = builder.load(
                builder.gep(W_idx_N, [builder.add(i_val, ir.Constant(int_ty, 1))])
            )

            cond2 = builder.icmp_signed("<", j_val, W_idx_N_i_1)
            with builder.if_then(cond2):
                # int r = W_idx_M[j];
                r = builder.alloca(int_ty, name="r")
                builder.store(builder.load(builder.gep(W_idx_M, [j_val])), r)

                # float sv = W_val[j];
                sv = builder.alloca(ir.FloatType(), name="sv")
                builder.store(builder.load(builder.gep(W_val, [j_val])), sv)

                # __m256 v = _mm256_set1_ps(sv);
                v = builder.alloca(vec_float_ty, name="v")
                v_replaced = builder.insert_element(
                    builder.load(v), builder.load(sv), ir.Constant(int_ty, 0)
                )
                v_set = builder.shuffle_vector(
                    v_replaced, v_replaced, ir.Constant(ir.VectorType(int_ty, 8), 0)
                )
                builder.store(v_set, v)

                # float sacc = 0;
                sacc = builder.alloca(ir.FloatType(), name="sacc")
                builder.store(ir.Constant(ir.FloatType(), 0), sacc)

                # __m256 acc = _mm256_setzero_ps();
                acc = builder.alloca(vec_float_ty, name="acc")
                builder.store(ir.Constant(vec_float_ty, 0), acc)

                self.bck_block_loop(builder, B, r, i_val, dLdX, X, dLdO, v, acc)

                # const __m128 hiQuad0 = _mm256_extractf128_ps(acc, 1);
                # shufflevector <8 x float> %89, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
                acc_val = builder.load(acc)
                hiQuad0 = builder.alloca(vec_float_half_ty, name="hiQuad0")
                builder.store(
                    builder.shuffle_vector(
                        acc_val,
                        acc_val,
                        ir.Constant(ir.VectorType(int_ty, 4), [4, 5, 6, 7]),
                    ),
                    hiQuad0,
                )

                # const __m128 loQuad0 = _mm256_castps256_ps128(acc);
                # shufflevector <8 x float> %89, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
                loQuad0 = builder.alloca(vec_float_half_ty, name="loQuad0")
                builder.store(
                    builder.shuffle_vector(
                        acc_val,
                        acc_val,
                        ir.Constant(ir.VectorType(int_ty, 4), [0, 1, 2, 3]),
                    ),
                    loQuad0,
                )

                # const __m128 sumQuad0 = _mm_add_ps(loQuad0, hiQuad0);
                sumQuad0 = builder.alloca(vec_float_half_ty, name="sumQuad0")
                builder.store(
                    builder.fadd(builder.load(loQuad0), builder.load(hiQuad0)), sumQuad0
                )

                # const __m128 hiDual0 = _mm_movehl_ps(sumQuad0, sumQuad0);
                # shufflevector <4 x float> %50, <4 x i32> <i32 2, i32 3, i32 poison, i32 poison>
                hiDual0 = builder.alloca(vec_float_half_ty, name="hiDual0")
                sumQuad0_val = builder.load(sumQuad0)
                builder.store(
                    builder.shuffle_vector(
                        sumQuad0_val,
                        sumQuad0_val,
                        ir.Constant(ir.VectorType(int_ty, 4), [6, 7, 2, 3]),
                    ),
                    hiDual0,
                )

                # const __m128 sumDual0 = _mm_add_ps(sumQuad0, hiDual0);
                sumDual0 = builder.alloca(vec_float_half_ty, name="sumDual0")
                builder.store(
                    builder.fadd(builder.load(sumQuad0), builder.load(hiDual0)),
                    sumDual0,
                )

                # const __m128 hi0 = _mm_shuffle_ps(sumDual0, sumDual0, 0x1);
                # shufflevector <4 x float> %52, <4 x float> poison, <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>
                hi0 = builder.alloca(vec_float_half_ty, name="hi0")
                sumDual0_val = builder.load(sumDual0)
                builder.store(
                    builder.shuffle_vector(
                        sumDual0_val,
                        sumDual0_val,
                        ir.Constant(ir.VectorType(int_ty, 4), [1, 0, 0, 0]),
                    ),
                    hi0,
                )

                # const __m128 sum0 = _mm_add_ss(sumDual0, hi0);
                sum0 = builder.alloca(vec_float_half_ty, name="sum0")
                builder.store(
                    builder.fadd(builder.load(sumDual0), builder.load(hi0)), sum0
                )

                # dLdW_val[j] = sacc + _mm_cvtss_f32(sum0);
                # extractelement <4 x float> %129, i64 0
                cvt = builder.extract_element(
                    builder.load(sum0), ir.Constant(int_ty, 0)
                )
                builder.store(
                    builder.fadd(builder.load(sacc), cvt),
                    builder.gep(dLdW_val, [j_val]),
                )

                builder.store(builder.add(j_val, ir.Constant(int_ty, 1)), j)
                builder.branch(inner_loop)

            builder.store(builder.add(i_val, ir.Constant(int_ty, 1)), i)
            builder.branch(outer_loop)
        builder.ret_void()

        llvm_ir = str(self.module)
        mod = binding.parse_assembly(llvm_ir)

        LinearJIT.optimize(mod, self.target_machine)

        self.engine.add_module(mod)
        self.engine.finalize_object()

        fn_ptr = self.engine.get_function_address(fn_name)

        self.fn_bwd = ctypes.CFUNCTYPE(
            None,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        )(fn_ptr)

        ctypes.CDLL(None)

    def _call(self, worker, N, num_threads):
        if self.options.parallel:
            chunk_size = (N + num_threads - 1) // num_threads
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i in range(num_threads):
                    futures.append(
                        executor.submit(
                            worker, i * chunk_size, min((i + 1) * chunk_size, N)
                        )
                    )
                wait(futures)
        else:
            worker(0, N)

    def call_forward(
        self, B, M, N, W_nnz, X, W_idx_N, W_idx_M, W_val, output, num_threads=4
    ):
        def worker(n_start, n_end):
            self.fn_fwd(
                B,
                M,
                n_start,
                n_end,
                W_nnz,
                ctypes.cast(X, ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(W_idx_N, ctypes.POINTER(ctypes.c_int)),
                ctypes.cast(W_idx_M, ctypes.POINTER(ctypes.c_int)),
                ctypes.cast(W_val, ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(output, ctypes.POINTER(ctypes.c_float)),
            )

        if not self.fn_fwd:
            self._jit_forward()

        self._call(worker, N, num_threads)

    def call_backward(
        self,
        B,
        M,
        N,
        W_nnz,
        X,
        W_idx_N,
        W_idx_M,
        W_val,
        dLdO,
        dLdX,
        dLdW_val,
        num_threads=4,
    ):
        def worker(n_start, n_end):
            self.fn_bwd(
                B,
                M,
                n_start,
                n_end,
                W_nnz,
                ctypes.cast(X, ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(W_idx_N, ctypes.POINTER(ctypes.c_int)),
                ctypes.cast(W_idx_M, ctypes.POINTER(ctypes.c_int)),
                ctypes.cast(W_val, ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(dLdO, ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(dLdX, ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(dLdW_val, ctypes.POINTER(ctypes.c_float)),
            )

        if not self.fn_bwd:
            self._jit_backward()

        self._call(worker, N, num_threads)
