from concurrent.futures import ThreadPoolExecutor, wait
import itertools
from threading import Thread

import ctypes
import time
import pulp
from llvmlite import ir, binding
import torch

from sparseprop.modules.jit_utils import JITOptions


# Batch size should be multiples of 8
class LinearJIT:
    def __init__(
        self,
        W_val,
        W_idx,
        W_shape,
        jit_options=JITOptions(),
        reg_tile_ops=None,
        name="jit_sparse_linear_forward",
    ):
        self.options = jit_options
        self.unroll_times = jit_options.batch_size // 8

        self.W_val = W_val
        self.W_idx_N, self.W_idx_M = W_idx
        self.N = W_shape[0]
        self.M = W_shape[1]
        if self.options.psc:
            self.codelets, self.trace, self.non_strided_N = self.find_strides(
                self.N, self.W_idx_N, self.W_idx_M
            )

        self.fn_fwd = None
        self.fn_bwd = None

        self.back_jit_thread = None

        if self.options.reg_tiling:
            self.reg_tile_groups, self.W_reg_offset = reg_tile_ops

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

        self.fma_float = ir.Function(
            self.module,
            ir.FunctionType(
                ir.FloatType(), [ir.FloatType(), ir.FloatType(), ir.FloatType()]
            ),
            name="llvm.fma.f32",
        )

        self.gather = ir.Function(
            self.module,
            ir.FunctionType(
                ir.VectorType(ir.FloatType(), 8),
                [
                    ir.VectorType(ir.FloatType(), 8),  # Passthrough
                    ir.PointerType(ir.FloatType()),  # base address pointer
                    ir.VectorType(ir.IntType(32), 8),  # indices
                    ir.VectorType(ir.FloatType(), 8),  # Mask
                    ir.IntType(8),  # scale (as i8)
                ],
            ),
            name="llvm.x86.avx2.gather.d.ps",
        )

        self.hadd = ir.Function(
            self.module,
            ir.FunctionType(
                ir.VectorType(ir.FloatType(), 4),
                [ir.VectorType(ir.FloatType(), 4), ir.VectorType(ir.FloatType(), 4)],
            ),
            name="llvm.x86.sse3.hadd.ps",
        )
        ############# Printf
        printf_ty = ir.FunctionType(
            ir.IntType(32), [ir.PointerType(ir.IntType(8))], var_arg=True
        )
        self.printf = ir.Function(self.module, printf_ty, name="printf")

        fmt_str = "%d %d\n\0"
        fmt_bytes = bytearray(fmt_str.encode("utf8"))
        fmt_type = ir.ArrayType(ir.IntType(8), len(fmt_bytes))

        self.global_fmt = ir.GlobalVariable(self.module, fmt_type, name="fstr")
        self.global_fmt.global_constant = True
        self.global_fmt.initializer = ir.Constant(fmt_type, fmt_bytes)
        ############# Printf

    def find_strides(self, n, row_idx, col_idx):
        assert self.options.psc, "PSC option needs to be provided!"
        trace = []
        for i in range(n):
            k = range(row_idx[i], row_idx[i + 1])
            trace.append(list(zip(k, [col_idx[j].item() for j in k])))

        sorted_indices = sorted(range(len(trace)), key=lambda i: len(trace[i]))
        orig_trace = trace.copy()
        trace = [trace[i] for i in sorted_indices]

        k = 0
        codelets = []
        visited = [
            False for _ in range(n)
        ]  # Keep track of indices already included in codelets
        while k < len(trace):
            curr_len = len(trace[k])
            if visited[k] or curr_len <= 1:
                k += 1
                continue
            target_offset = [
                (k2 - k1, W_idx2 - W_idx1)
                for (k1, W_idx1), (k2, W_idx2) in zip(trace[k], trace[k][1:])
            ]
            scale = None
            i = k + 1
            codelet = {}
            while i < len(trace) and len(trace[i]) == curr_len:
                outer_iter = [
                    (k2 - k1, W_idx2 - W_idx1)
                    for (k1, W_idx1), (k2, W_idx2) in zip(trace[k], trace[i])
                ]
                outer = all(t == outer_iter[0] for t in outer_iter)
                if outer and not scale:
                    scale = outer_iter[0]
                if outer and scale == outer_iter[0]:
                    curr_offset = [
                        (k2 - k1, W_idx2 - W_idx1)
                        for (k1, W_idx1), (k2, W_idx2) in zip(trace[i], trace[i][1:])
                    ]
                    # codelet is strided
                    if target_offset == curr_offset:
                        if not codelet:
                            codelet = {
                                "len": curr_len,
                                "x_offset": [trace[k][0][1]]
                                + [x for (_, x) in target_offset],
                                "w_offset": [trace[k][0][0]]
                                + [x for (x, _) in target_offset],
                                "w_scale": scale[0],
                                "x_scale": scale[1],
                                "o_idx": [sorted_indices[k]],
                                "o_scale": sorted_indices[i] - sorted_indices[k],
                            }
                            visited[k] = True

                        # Since relative order is maintained while sorting by len,
                        # We just check against the last element
                        if codelet["o_scale"] == (
                            sorted_indices[i] - codelet["o_idx"][-1]
                        ):
                            codelet["o_idx"].append(sorted_indices[i])
                            visited[i] = True
                            scale = (
                                scale[0] + outer_iter[0][0],
                                scale[1] + outer_iter[0][1],
                            )
                    else:
                        scale = None
                i += 1
            if codelet:
                print(codelet)
                codelets.append(codelet)
            k += 1
        return (
            codelets,
            orig_trace,
            [sorted_indices[i] for i, val in enumerate(visited) if not val],
        )

    @staticmethod
    def find_enum_freq(W_dense, t_i):
        n, m = W_dense.shape
        enum_freq = [0] * (2**t_i)
        enum_assignment = []
        for i in range(0, n - t_i + 1, t_i):
            enum_assignment.append([])
            for j in range(0, m):
                index = 0
                for k in range(0, t_i):
                    index <<= 1
                    if W_dense[i + k][j]:
                        index += 1
                enum_freq[index] += 1
                enum_assignment[-1].append(index)
        return enum_freq, enum_assignment

    @staticmethod
    def powerset(iterable):
        s = list(iterable)
        return list(
            itertools.chain.from_iterable(
                itertools.combinations(s, r) for r in range(len(s) + 1)
            )
        )

    @staticmethod
    def get_group_cost(indices):
        VAR_COST = 1  # Cost of loads of A and fma
        BASE_COST = 2  # cost of loads of B
        idx = 0
        for i in indices:
            idx |= i
        unique_nnz = bin(idx).count("1")
        return unique_nnz * VAR_COST + BASE_COST

    @staticmethod
    def solve_set_cover_pulp(
        elements_to_cover: set,
        available_subsets: list,
        subset_costs: list,
        max_num_subset: int = None,
        force_num_subsets: bool = False,
    ):
        num_subsets = len(available_subsets)

        prob = pulp.LpProblem("SetCoverProblem", pulp.LpMinimize)

        x = pulp.LpVariable.dicts(
            name="subset", indices=range(num_subsets), cat=pulp.LpBinary
        )

        prob += (
            pulp.lpSum([subset_costs[i] * x[i] for i in range(num_subsets)]),
            "Total_Cost",
        )

        for element in elements_to_cover:
            covering_subsets = [
                x[i] for i, subset in enumerate(available_subsets) if element in subset
            ]

            if covering_subsets:
                prob += pulp.lpSum(covering_subsets) == 1, f"CoverElement_{element}"

        if max_num_subset is not None:
            total_subsets_chosen = pulp.lpSum([x[i] for i in range(num_subsets)])
            if force_num_subsets:
                prob += (
                    total_subsets_chosen == max_num_subset,
                    "Force_Exact_Number_of_Subsets",
                )
            else:
                prob += total_subsets_chosen <= max_num_subset, "Max_Number_of_Subsets"

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] == "Optimal":
            chosen_indices = [i for i in range(num_subsets) if x[i].varValue > 0.9]
            chosen_subsets = [available_subsets[i] for i in chosen_indices]
            return chosen_subsets
        else:
            print(
                f"Could not find an optimal solution. Status: {pulp.LpStatus[prob.status]}"
            )
            return None

    @staticmethod
    def solve_sparse_jam(W_dense, row_tile):
        OVERHEAD = 0.7
        enum_freq, enum_assignment = LinearJIT.find_enum_freq(W_dense, row_tile)
        to_cover = {i for i, freq in enumerate(enum_freq) if freq > 0}
        all_sets = LinearJIT.powerset(range(1, len(enum_freq)))
        all_costs = [0] * len(all_sets)
        for i, g in enumerate(all_sets):
            freq = 0
            for idx in g:
                freq += enum_freq[idx]
            all_costs[i] = LinearJIT.get_group_cost(g) * freq + OVERHEAD

        included_groups = LinearJIT.solve_set_cover_pulp(
            to_cover, all_sets, all_costs, max_num_subset=5
        )
        column_vals = []
        column_pointers = []
        for i, assignments in enumerate(enum_assignment):
            mapping_dict = {}
            for j, idx in enumerate(assignments):
                if idx not in mapping_dict:
                    mapping_dict[idx] = []
                mapping_dict[idx].append(j)

            # Look at groups now:
            column_pointers.append([len(column_vals)])
            for g in included_groups:
                for idx in g:
                    if idx in mapping_dict:
                        column_vals += mapping_dict[idx]
                column_pointers[i].append(len(column_vals))
        tile_groups = [[] for _ in range(len(included_groups))]
        for i, g in enumerate(included_groups):
            idx = 0
            for j in g:
                idx |= j
            bin_str = bin(idx)[2:]
            bin_str = (row_tile - len(bin_str)) * "0" + bin_str
            tile_groups[i] = [j for j, bit in enumerate(bin_str) if bit == "1"]
        # Also get W_val
        # Also, get W_idx_N, W_idx_M to recreate later
        W_val = []
        to_dense_idx = {}
        W_idx_N, W_idx_M = [0], []
        recreate_idx = [[] for _ in range(len(W_dense))]
        W_indices = [[] for _ in range(len(W_dense))]
        for i, ptr in enumerate(column_pointers):
            for j in range(len(tile_groups)):
                for idx in range(ptr[j], ptr[j + 1]):
                    for i_idx in tile_groups[j]:
                        row_idx = i * row_tile + i_idx
                        col_idx = column_vals[idx]
                        W_indices[row_idx].append(col_idx)
                        recreate_idx[row_idx].append(len(W_val))
                        W_val.append(W_dense[row_idx][col_idx])
                        if W_val[-1]:
                            to_dense_idx[(row_idx, col_idx)] = len(W_val) - 1
        recreate_idx_from = [item for row in recreate_idx for item in row]
        recreate_idx_back = [0] * len(recreate_idx_from)
        for i, idx in enumerate(recreate_idx_from):
            recreate_idx_back[idx] = i
        for cols in W_indices:
            for i in cols:
                W_idx_M.append(i)
            W_idx_N.append(len(W_idx_M))
        # For parallel execution, we need to know the exact offset for W_val
        W_offset = [0]
        for ptr in column_pointers:
            acc = W_offset[-1]
            for i in range(len(ptr) - 1):
                acc += (ptr[i + 1] - ptr[i]) * len(tile_groups[i])
            W_offset.append(acc)
        # print(tile_groups, len(W_val), "\n", column_pointers, "\n", column_vals)
        return (
            (tile_groups, W_offset),
            torch.Tensor(W_val),
            (torch.Tensor(column_pointers).int(), torch.Tensor(column_vals).int()),
            (
                torch.Tensor(W_idx_N).int(),
                torch.Tensor(W_idx_M).int(),
                (
                    # To switch from and back to the special reg tile representation
                    torch.Tensor(recreate_idx_from).int(),
                    torch.Tensor(recreate_idx_back).int(),
                ),
            ),
            to_dense_idx,
        )

    def gen_fwd_codelet(
        self,
        codelet,
        builder,
        b_idx,
        B,
        X,
        W_val,
        O,
    ):
        i = builder.alloca(ir.IntType(32), name="i")
        builder.store(
            ir.Constant(ir.IntType(32), 0), i
        )  # index O with this, so O[o_idx[0] + i * o_scale]
        o_scale = ir.Constant(ir.IntType(32), codelet["o_scale"])
        w_scale = ir.Constant(ir.IntType(32), codelet["w_scale"])
        x_scale = ir.Constant(ir.IntType(32), codelet["x_scale"])
        m = ir.Constant(ir.IntType(32), self.M)
        o_offset = ir.Constant(ir.IntType(32), codelet["o_idx"][0])
        w_offset = 0
        x_offset = 0

        batch_loop = builder.append_basic_block("codelet_loop")
        builder.branch(batch_loop)
        builder.position_at_start(batch_loop)

        i_val = builder.load(i)

        cond = builder.icmp_signed(
            "<", i_val, ir.Constant(ir.IntType(32), len(codelet["o_idx"]))
        )
        with builder.if_then(cond):
            # Output is of dims N * Batch, this is for when we vectorize
            o_idx = builder.add(
                builder.mul(builder.add(builder.mul(i_val, o_scale), o_offset), B),
                b_idx,
            )
            for j in range(codelet["len"]):
                x_offset += codelet["x_offset"][j]
                w_offset += codelet["w_offset"][j]

                w_idx = builder.add(
                    builder.mul(i_val, w_scale),
                    ir.Constant(ir.IntType(32), w_offset),
                )
                w = builder.gep(W_val, [w_idx])

                x_idx = builder.add(
                    builder.mul(b_idx, m),
                    builder.add(
                        builder.mul(i_val, x_scale),
                        ir.Constant(ir.IntType(32), x_offset),
                    ),
                )
                x = builder.gep(X, [x_idx])

                o = builder.gep(O, [o_idx])
                fma = builder.call(
                    self.fma_float,
                    [builder.load(x), builder.load(w), builder.load(o)],
                )
                builder.store(fma, o)
            builder.store(builder.add(i_val, ir.Constant(ir.IntType(32), 1)), i)
            builder.branch(batch_loop)

    def gen_bwd_codelet(
        self, codelet, builder, b_idx, B, X, W_val, dLdO, dLdW_val, dLdX
    ):
        i = builder.alloca(ir.IntType(32), name="i")
        builder.store(
            ir.Constant(ir.IntType(32), 0), i
        )  # index O with this, so O[o_idx[0] + i * o_scale]
        outer_scale = ir.Constant(ir.IntType(32), codelet["o_scale"])
        inner_scale = ir.Constant(ir.IntType(32), codelet["w_scale"])
        sparse_scale = ir.Constant(ir.IntType(32), codelet["x_scale"])
        m = ir.Constant(ir.IntType(32), self.M)
        outer_offset = ir.Constant(ir.IntType(32), codelet["o_idx"][0])
        inner_offset = 0
        sparse_offset = 0

        batch_loop = builder.append_basic_block("codelet_loop")
        builder.branch(batch_loop)
        builder.position_at_start(batch_loop)

        i_val = builder.load(i)

        cond = builder.icmp_signed(
            "<", i_val, ir.Constant(ir.IntType(32), len(codelet["o_idx"]))
        )
        with builder.if_then(cond):
            # Output is of dims N * Batch, this is for when we vectorize
            outer_idx = builder.add(
                builder.mul(
                    builder.add(builder.mul(i_val, outer_scale), outer_offset), B
                ),
                b_idx,
            )
            for j in range(codelet["len"]):
                sparse_offset += codelet["x_offset"][j]
                inner_offset += codelet["w_offset"][j]

                # dLdX[W_col[j]] += W_val[j] * dLdO[i];
                inner_idx = builder.add(
                    builder.mul(i_val, inner_scale),
                    ir.Constant(ir.IntType(32), inner_offset),
                )
                w = builder.gep(W_val, [inner_idx])

                dldo = builder.gep(dLdO, [outer_idx])

                sparse_idx = builder.add(
                    builder.mul(
                        builder.add(
                            builder.mul(i_val, sparse_scale),
                            ir.Constant(ir.IntType(32), sparse_offset),
                        ),
                        B,
                    ),
                    b_idx,
                )
                output = builder.gep(dLdX, [sparse_idx])
                fma = builder.call(
                    self.fma_float,
                    [builder.load(dldo), builder.load(w), builder.load(output)],
                )
                builder.store(fma, output)

                # dLdW_val[j] += dLdO[i] * X[W_col[j]];
                x = builder.gep(X, [sparse_idx])
                output = builder.gep(dLdW_val, [inner_idx])
                fma = builder.call(
                    self.fma_float,
                    [builder.load(dldo), builder.load(x), builder.load(output)],
                )
                builder.store(fma, output)
            builder.store(builder.add(i_val, ir.Constant(ir.IntType(32), 1)), i)
            builder.branch(batch_loop)

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
    def vec_load_arr(
        builder, arr, idx, type=ir.VectorType(ir.FloatType(), 8), name=None
    ):
        if name is not None:
            vec = builder.alloca(type, name=name)
        else:
            vec = builder.alloca(type)
        builder.store(
            builder.load(
                builder.bitcast(builder.gep(arr, [idx]), ir.PointerType(type))
            ),
            vec,
        )
        return vec

    @staticmethod
    def vec_set(builder, vec, element):
        replaced = builder.insert_element(
            builder.load(vec),
            element,
            ir.Constant(ir.IntType(32), 0),
        )
        set = builder.shuffle_vector(
            replaced,
            replaced,
            ir.Constant(ir.VectorType(ir.IntType(32), 8), 0),
        )
        builder.store(set, vec)

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

    def generate_fwd_unrolled_mm(self, builder, B, indices, b_idx, X, W_val, O):
        # Output is of dims N * Batch
        for i in indices:
            o_idx = None
            for k in range(self.W_idx_N[i], self.W_idx_N[i + 1]):
                if not o_idx:
                    o_idx = builder.add(
                        builder.mul(ir.Constant(ir.IntType(32), i), B),
                        b_idx,
                    )
                w = builder.gep(W_val, [ir.Constant(ir.IntType(32), k)])
                x = builder.gep(
                    X,
                    [
                        builder.add(
                            builder.mul(b_idx, ir.Constant(ir.IntType(32), self.M)),
                            ir.Constant(ir.IntType(32), self.W_idx_M[k].item()),
                        )
                    ],
                )
                o = builder.gep(O, [o_idx])
                fma = builder.call(
                    self.fma_float,
                    [builder.load(x), builder.load(w), builder.load(o)],
                )
                builder.store(fma, o)

    def generate_bwd_unrolled_mm(
        self, builder, indices, b_idx, B, X, W_val, dLdO, dLdW_val, dLdX
    ):
        # Output is of dims N * Batch
        for i in indices:
            outer_idx = None
            for k in range(self.W_idx_N[i], self.W_idx_N[i + 1]):
                if not outer_idx:
                    outer_idx = builder.add(
                        builder.mul(ir.Constant(ir.IntType(32), i), B),
                        b_idx,
                    )
                # dLdX[W_col[j]] += W_val[j] * dLdO[i];
                inner_idx = ir.Constant(ir.IntType(32), k)
                w = builder.gep(W_val, [inner_idx])
                dldo = builder.gep(dLdO, [outer_idx])

                sparse_idx = builder.add(
                    builder.mul(
                        ir.Constant(ir.IntType(32), self.W_idx_M[k].item()),
                        B,
                    ),
                    b_idx,
                )
                output = builder.gep(dLdX, [sparse_idx])
                fma = builder.call(
                    self.fma_float,
                    [builder.load(dldo), builder.load(w), builder.load(output)],
                )
                builder.store(fma, output)

                # dLdW_val[j] += dLdO[i] * X[W_col[j]];
                x = builder.gep(X, [sparse_idx])
                output = builder.gep(dLdW_val, [inner_idx])
                fma = builder.call(
                    self.fma_float,
                    [builder.load(dldo), builder.load(x), builder.load(output)],
                )
                builder.store(fma, output)

    def _jit_forward_psc(
        self, builder, pstart, pend, B, M, N, W_nnz, X, W_idx_N, W_idx_M, W_val, O
    ):
        j = builder.alloca(ir.IntType(32), name="j")
        builder.store(pstart, j)

        batch_loop = builder.append_basic_block("batch_loop")
        builder.branch(batch_loop)
        builder.position_at_start(batch_loop)

        j_val = builder.load(j)

        cond = builder.icmp_signed("<", j_val, pend)
        with builder.if_then(cond):
            for codelet in self.codelets:
                self.gen_fwd_codelet(codelet, builder, j_val, B, X, W_val, O)
            self.generate_fwd_unrolled_mm(
                builder, B, self.non_strided_N, j_val, X, W_val, O
            )

            builder.store(builder.add(j_val, ir.Constant(ir.IntType(32), 1)), j)
            builder.branch(batch_loop)

    def _jit_forward_reg_tiled(
        self,
        builder,
        pstart,
        pend,
        B,
        M,
        N,
        W_nnz,
        X,
        col_ptr,
        col_idx,
        W_val,
        O,
        w_offset,
    ):
        t_i = ir.Constant(ir.IntType(32), self.options.reg_tile_size[0])
        t_j = ir.Constant(ir.IntType(32), self.options.reg_tile_size[1])

        i = builder.alloca(ir.IntType(32), name="i")
        builder.store(builder.mul(pstart, t_i), i)  # pstart * tile_size

        w_cnt = builder.alloca(ir.IntType(32), name="w_cnt")
        curr = builder.alloca(ir.IntType(32), name="curr")
        builder.store(w_offset, curr)
        next = builder.alloca(ir.IntType(32), name="next")

        N_loop = builder.append_basic_block("N_loop")
        builder.branch(N_loop)
        builder.position_at_start(N_loop)

        i_val = builder.load(i)

        cond = builder.icmp_signed(
            "<", i_val, builder.mul(pend, t_i)
        )  # pend * tile_size
        with builder.if_then(cond):
            j = builder.alloca(ir.IntType(32), name="j")
            builder.store(ir.Constant(ir.IntType(32), 0), j)

            batch_loop = builder.append_basic_block("B_loop")
            builder.branch(batch_loop)
            builder.position_at_start(batch_loop)

            j_val = builder.load(j)

            cond = builder.icmp_signed("<", j_val, B)
            with builder.if_then(cond):
                builder.store(builder.load(curr), w_cnt)
                out_indices = [
                    builder.add(
                        builder.mul(
                            builder.add(i_val, ir.Constant(ir.IntType(32), i_idx)), B
                        ),
                        builder.add(j_val, ir.Constant(ir.IntType(32), j_idx)),
                    )
                    for i_idx in range(self.options.reg_tile_size[0])
                    for j_idx in range(self.options.reg_tile_size[1])
                ]
                out = [builder.alloca(ir.FloatType()) for _ in out_indices]
                for o_idx, idx in enumerate(out_indices):
                    builder.store(builder.load(builder.gep(O, [idx])), out[o_idx])

                for cnt, g in enumerate(self.reg_tile_groups):
                    p = builder.alloca(ir.IntType(32))
                    idx_part = builder.mul(
                        builder.sdiv(i_val, t_i),
                        ir.Constant(ir.IntType(32), len(self.reg_tile_groups) + 1),
                    )
                    col_ptr_start = builder.load(
                        builder.gep(
                            col_ptr,
                            [builder.add(idx_part, ir.Constant(ir.IntType(32), cnt))],
                        )
                    )
                    col_ptr_end = builder.load(
                        builder.gep(
                            col_ptr,
                            [
                                builder.add(
                                    idx_part, ir.Constant(ir.IntType(32), cnt + 1)
                                )
                            ],
                        )
                    )
                    builder.store(col_ptr_start, p)

                    tile_loop = builder.append_basic_block("tile_loop")
                    builder.branch(tile_loop)
                    builder.position_at_start(tile_loop)

                    p_val = builder.load(p)
                    cond = builder.icmp_signed("<", p_val, col_ptr_end)
                    with builder.if_then(cond):
                        k_val = builder.load(builder.gep(col_idx, [p_val]))
                        for i_offset in g:
                            for j_offset in range(self.options.reg_tile_size[1]):
                                x_idx = builder.add(
                                    builder.mul(k_val, B),
                                    builder.add(
                                        j_val,
                                        ir.Constant(ir.IntType(32), j_offset),
                                    ),
                                )
                                x = builder.gep(X, [x_idx])
                                fma = builder.call(
                                    self.fma_float,
                                    [
                                        builder.load(x),
                                        builder.load(
                                            builder.gep(
                                                W_val,
                                                [builder.load(w_cnt)],
                                            )
                                        ),
                                        builder.load(
                                            out[
                                                i_offset * self.options.reg_tile_size[1]
                                                + j_offset
                                            ]
                                        ),
                                    ],
                                )
                                builder.store(
                                    fma,
                                    out[
                                        i_offset * self.options.reg_tile_size[1]
                                        + j_offset
                                    ],
                                )

                            builder.store(
                                builder.add(
                                    builder.load(w_cnt), ir.Constant(ir.IntType(32), 1)
                                ),
                                w_cnt,
                            )
                            # builder.call(
                            #     self.printf,
                            #     [
                            #         builder.bitcast(
                            #             self.global_fmt, ir.IntType(8).as_pointer()
                            #         ),
                            #         builder.load(w_cnt),
                            #     ],
                            # )
                        builder.store(
                            builder.add(p_val, ir.Constant(ir.IntType(32), 1)), p
                        )
                        builder.branch(tile_loop)

                for o_idx, idx in enumerate(out_indices):
                    builder.store(builder.load(out[o_idx]), builder.gep(O, [idx]))

                builder.store(builder.load(w_cnt), next)
                builder.store(builder.add(j_val, t_j), j)
                builder.branch(batch_loop)

            builder.store(builder.load(next), curr)
            builder.store(builder.add(i_val, t_i), i)
            builder.branch(N_loop)

    def _jit_backward_psc(
        self,
        builder,
        pstart,
        pend,
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
    ):
        j = builder.alloca(ir.IntType(32), name="j")
        builder.store(pstart, j)

        batch_loop = builder.append_basic_block("batch_loop")
        builder.branch(batch_loop)
        builder.position_at_start(batch_loop)

        j_val = builder.load(j)

        cond = builder.icmp_signed("<", j_val, pend)
        with builder.if_then(cond):
            for codelet in self.codelets:
                self.gen_bwd_codelet(
                    codelet, builder, j_val, B, X, W_val, dLdO, dLdW_val, dLdX
                )
            self.generate_bwd_unrolled_mm(
                builder, self.non_strided_N, j_val, B, X, W_val, dLdO, dLdW_val, dLdX
            )

            builder.store(builder.add(j_val, ir.Constant(ir.IntType(32), 1)), j)
            builder.branch(batch_loop)

    def _jit_forward_unrolled_scalar(
        self, builder, pstart, pend, B, M, N, W_nnz, X, W_idx_N, W_idx_M, W_val, O
    ):
        # Only call this with non transposed inputs!
        # As otherwise we would have bad spatial locality
        j = builder.alloca(ir.IntType(32), name="j")
        builder.store(pstart, j)

        batch_loop = builder.append_basic_block("batch_loop")
        builder.branch(batch_loop)
        builder.position_at_start(batch_loop)

        j_val = builder.load(j)

        cond = builder.icmp_signed("<", j_val, pend)
        with builder.if_then(cond):
            self.generate_fwd_unrolled_mm(builder, B, range(self.N), j_val, X, W_val, O)
            builder.store(builder.add(j_val, ir.Constant(ir.IntType(32), 1)), j)
            builder.branch(batch_loop)

    def _jit_forward_unrolled_vector(
        self, builder, pstart, pend, B, M, N, W_nnz, X, W_idx_N, W_idx_M, W_val, O
    ):
        # Only call this with non transposed inputs!
        # As otherwise we would have bad spatial locality
        mask = ir.Constant(ir.VectorType(ir.FloatType(), 8), -1)
        passthrough = ir.Constant(ir.VectorType(ir.FloatType(), 8), 0)
        scale = ir.Constant(ir.IntType(8), 4)
        vec_float_half_ty = ir.VectorType(ir.FloatType(), 4)
        out = None
        # fmt_ptr = builder.bitcast(self.global_fmt, ir.PointerType(ir.IntType(8)))

        idx_vec = builder.alloca(ir.VectorType(ir.IntType(32), 8))
        out = builder.alloca(ir.VectorType(ir.FloatType(), 8))
        builder.store(ir.Constant(ir.VectorType(ir.FloatType(), 8), 0), out)
        builder.store(ir.Constant(ir.VectorType(ir.IntType(32), 8), 0), idx_vec)

        j = builder.alloca(ir.IntType(32), name="j")
        builder.store(pstart, j)
        LinearJIT.vec_set(builder, idx_vec, pstart)

        batch_loop = builder.append_basic_block("batch_loop")
        builder.branch(batch_loop)
        builder.position_at_start(batch_loop)

        j_val = builder.load(j)
        cond = builder.icmp_signed("<", j_val, pend)
        with builder.if_then(cond):
            for i in range(self.N):
                k = self.W_idx_N[i].item()
                # Locality wrt to accesses to O is BAD here!!
                if k < self.W_idx_N[i + 1]:
                    o_idx = builder.add(
                        builder.mul(ir.Constant(ir.IntType(32), i), B),
                        j_val,
                    )
                else:
                    o_idx = None
                do_vec = k < self.W_idx_N[i + 1] - 7

                while k < self.W_idx_N[i + 1] - 7:
                    w = LinearJIT.vec_load_arr(
                        builder, W_val, ir.Constant(ir.IntType(32), k)
                    )
                    x_idx = builder.add(
                        builder.load(idx_vec),
                        ir.Constant(
                            ir.VectorType(ir.IntType(32), 8),
                            [self.W_idx_M[k + b].item() for b in range(0, 8)],
                        ),
                    )
                    x = builder.call(
                        self.gather,
                        [passthrough, X, x_idx, mask, scale],
                    )
                    # x = ir.Constant(ir.VectorType(ir.FloatType(), 8), 0)
                    fma = builder.call(
                        self.fma_intr,
                        [x, builder.load(w), builder.load(out)],
                    )
                    builder.store(fma, out)
                    k += 8

                # horizontal sum over O before we move to scalar
                if do_vec:
                    # _mm256_castps256_ps128(v);
                    low = builder.alloca(vec_float_half_ty)
                    builder.store(
                        builder.shuffle_vector(
                            builder.load(out),
                            builder.load(out),
                            ir.Constant(ir.VectorType(ir.IntType(32), 4), [0, 1, 2, 3]),
                        ),
                        low,
                    )
                    # mm256_extractf128_ps(v, 1);
                    high = builder.alloca(vec_float_half_ty)
                    builder.store(
                        builder.shuffle_vector(
                            builder.load(out),
                            builder.load(out),
                            ir.Constant(ir.VectorType(ir.IntType(32), 4), [4, 5, 6, 7]),
                        ),
                        high,
                    )
                    # _mm_add_ps(low, high);
                    low_high_add = builder.fadd(builder.load(low), builder.load(high))
                    # _mm_hadd_ps(sum, sum);
                    low_sum = builder.call(
                        self.hadd,
                        [low_high_add, low_high_add],
                    )
                    # _mm_hadd_ps(sum, sum);
                    full_sum = builder.call(
                        self.hadd,
                        [low_sum, low_sum],
                    )
                    # _mm_cvtss_f32(sum);
                    cvt = builder.extract_element(
                        full_sum, ir.Constant(ir.IntType(32), 0)
                    )
                    O_ele = builder.gep(O, [o_idx])
                    builder.store(
                        builder.fadd(builder.load(O_ele), cvt),
                        O_ele,
                    )
                    builder.store(ir.Constant(ir.VectorType(ir.FloatType(), 8), 0), out)

                while k < self.W_idx_N[i + 1]:
                    w = builder.gep(W_val, [ir.Constant(ir.IntType(32), k)])
                    x = builder.gep(
                        X,
                        [
                            builder.add(
                                builder.mul(j_val, ir.Constant(ir.IntType(32), self.M)),
                                ir.Constant(ir.IntType(32), self.W_idx_M[k].item()),
                            )
                        ],
                    )
                    o = builder.gep(O, [o_idx])
                    fma = builder.call(
                        self.fma_float,
                        [builder.load(x), builder.load(w), builder.load(o)],
                    )
                    builder.store(fma, o)
                    k += 1
            builder.store(
                builder.add(
                    builder.load(idx_vec),
                    ir.Constant(ir.VectorType(ir.IntType(32), 8), [self.M] * 8),
                ),
                idx_vec,
            )
            builder.store(builder.add(j_val, ir.Constant(ir.IntType(32), 1)), j)
            builder.branch(batch_loop)

    def _jit_forward_vec(
        self, builder, pstart, pend, B, M, N, W_nnz, X, W_idx_N, W_idx_M, W_val, O
    ):
        int_ty = ir.IntType(32)
        vec_float_ty = ir.VectorType(ir.FloatType(), 8)

        # for(int i = 0; i < N; i++){
        i = builder.alloca(int_ty, name="i")
        builder.store(pstart, i)

        outer_loop = builder.append_basic_block("outer_loop")
        builder.branch(outer_loop)
        builder.position_at_start(outer_loop)

        i_val = builder.load(i)

        cond = builder.icmp_signed("<", i_val, pend)
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
                LinearJIT.vec_set(builder, v, builder.load(builder.gep(W_val, [k_val])))

                self.fwd_block_loop(builder, B, idx, i_val, X, O, v)

                builder.store(builder.add(k_val, ir.Constant(int_ty, 1)), k)
                builder.branch(inner_loop)

            builder.store(builder.add(i_val, ir.Constant(int_ty, 1)), i)
            builder.branch(outer_loop)

    def _jit_forward(self, fn_name="sparse_fwd"):
        # void fn(int pstart, int pend, int B, int M, int N, int W_nnz, float* X, int* W_idx_N, int* W_idx_M, float* W_val, float* O)
        float_ptr_ty = ir.PointerType(ir.FloatType())
        int_ptr_ty = ir.PointerType(ir.IntType(32))
        int_ty = ir.IntType(32)
        func_args = [
            int_ty,
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
        ]
        if self.options.reg_tiling:
            func_args.append(int_ty)
        func_ty = ir.FunctionType(ir.VoidType(), func_args)

        func = ir.Function(self.module, func_ty, name=fn_name)

        block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        if self.options.unrolled_scalar:
            self._jit_forward_unrolled_vector(builder, *func.args)
        elif self.options.reg_tiling:
            self._jit_forward_reg_tiled(builder, *func.args)
        elif self.options.psc:
            self._jit_forward_psc(builder, *func.args)
        else:
            self._jit_forward_vec(builder, *func.args)
        builder.ret_void()

        llvm_ir = str(self.module)
        mod = binding.parse_assembly(llvm_ir)
        LinearJIT.optimize(mod, self.target_machine)

        self.engine.add_module(mod)
        self.engine.finalize_object()

        fn_ptr = self.engine.get_function_address(fn_name)

        func_ctype_args = [
            None,
            ctypes.c_int,
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
        ]
        if self.options.reg_tiling:
            func_ctype_args.append(ctypes.c_int)
        self.fn_fwd = ctypes.CFUNCTYPE(*func_ctype_args)(fn_ptr)

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

    def _jit_backward_vec(
        self,
        builder,
        pstart,
        pend,
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
    ):
        int_ty = ir.IntType(32)
        vec_float_ty = ir.VectorType(ir.FloatType(), 8)
        vec_float_half_ty = ir.VectorType(ir.FloatType(), 4)
        # for(int i = 0; i < N; i++){
        i = builder.alloca(int_ty, name="i")
        builder.store(pstart, i)

        outer_loop = builder.append_basic_block("outer_loop")
        builder.branch(outer_loop)
        builder.position_at_start(outer_loop)

        i_val = builder.load(i)

        cond = builder.icmp_signed("<", i_val, pend)
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
                LinearJIT.vec_set(builder, v, builder.load(sv))

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

    def _jit_backward(self, fn_name="sparse_bwd"):
        # void fn(int pstart, int pend, int B, int M, int N, int W_nnz, float* X, int* W_idx_N, int* W_idx_M,float* W_val, float* dLdO, float* dLdX, float* dLdW_val)
        float_ptr_ty = ir.PointerType(ir.FloatType())
        int_ptr_ty = ir.PointerType(ir.IntType(32))
        int_ty = ir.IntType(32)
        func_ty = ir.FunctionType(
            ir.VoidType(),
            [
                int_ty,
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

        block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        if self.options.psc:
            self._jit_backward_psc(builder, *func.args)
        else:
            # Backward with register tilling is inefficient as we compute over the zero elements of W
            # We probably don't want this and need to change the staorage format of X...
            self._jit_backward_vec(builder, *func.args)
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

    def _call(self, worker, loop_size, num_threads):
        if not self.options.parallel:
            worker(0, loop_size)
            return

        chunk_size = (loop_size + num_threads - 1) // num_threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                start = i * chunk_size
                end = min(start + chunk_size, loop_size)
                futures.append(executor.submit(worker, start, end))
            wait(futures)

    def call_forward(
        self, B, M, N, W_nnz, X, W_idx_N, W_idx_M, W_val, output, num_threads=4
    ):
        def worker(start, end):
            w_offset = None
            if self.options.reg_tiling:
                w_offset = self.W_reg_offset[start]
            self.fn_fwd(
                start,
                end,
                B,
                M,
                N,
                W_nnz,
                ctypes.cast(X, ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(W_idx_N, ctypes.POINTER(ctypes.c_int)),
                ctypes.cast(W_idx_M, ctypes.POINTER(ctypes.c_int)),
                ctypes.cast(W_val, ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(output, ctypes.POINTER(ctypes.c_float)),
                w_offset,
            )

        # start = time.time()
        if not self.fn_fwd:
            # Also spin up the backward jit-er
            # if not self.fn_bwd:
            #     self.back_jit_thread = Thread(target=self._jit_backward)
            #     self.back_jit_thread.start()
            self._jit_forward()
        # print("Jit time (approx):", time.time() - start)
        if self.options.unrolled_scalar:
            loop_size = B
        elif self.options.reg_tiling:
            # We multiply later to recover this if not parallel
            loop_size = N // self.options.reg_tile_size[0]
        else:
            loop_size = N
        self._call(worker, loop_size, num_threads)

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
        def worker(start, end):
            self.fn_bwd(
                start,
                end,
                B,
                M,
                N,
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
            # self.back_jit_thread.join()
            self._jit_backward()

        self._call(worker, B if self.options.unrolled_scalar else N, num_threads)
