import torch
from scipy.sparse import csr_matrix

from sparseprop.modules.functions import SparseLinearFunction
from sparseprop.modules.utils import to_csr_2d, from_csr_2d
from sparseprop.modules.linear_jit import LinearJIT, JITOptions

class SparseLinear(torch.nn.Module):
    def __init__(self, dense_weight, bias=None, jit_fn = False, jit_ops = JITOptions()):
        super(SparseLinear, self).__init__()
        self.N, self.M = dense_weight.shape
        self.jit_ops = jit_ops 

        W_reg_idx = None
        W_dense_idx = None
        reg_tile_ops = None
        if jit_fn and jit_ops.reg_tiling:
            reg_tile_ops, W_val, W_idx, W_reg_idx, W_dense_idx = LinearJIT.solve_sparse_jam(dense_weight, self.jit_ops.reg_tile_size[0])
        else:
            W_val, W_idx = to_csr_2d(dense_weight)
        self.W_val = torch.nn.Parameter(W_val)
        self.W_idx = W_idx
        self.W_reg_idx = W_reg_idx
        self.W_dense_idx = W_dense_idx

        self.sparse_linear_fn = SparseLinearFunction()
        assert bias is None or isinstance(bias, torch.nn.Parameter), f"bias is not a parameter but it's {type(bias)}"
        self.bias = bias

        self.jit = LinearJIT(W_val, W_idx, dense_weight.shape, jit_ops, reg_tile_ops) if jit_fn else None

    @staticmethod
    def from_dense(module):
        return SparseLinear(
            dense_weight=module.weight.data,
            bias=None if module.bias is None else torch.nn.Parameter(module.bias.data.clone())
        )

    # convert W_val/W_grad from reg tile format to csr, for correctness checks
    def from_reg_tile_to_csr(self, W_val):
        if self.jit_ops.reg_tiling:
            return W_val[self.W_reg_idx[2][0]]
        return W_val
    
    def get_dense_grad(self):
        assert self.jit_ops.reg_tiling
        dense = torch.zeros(self.N, self.M)
        for (i, j), w_i in self.W_dense_idx.items():
            dense[i][j] = self.W_val.grad[w_i]
        return dense
            
    def to_dense(self):
        dense_weight = from_csr_2d(
                self.W_val,
                self.W_idx,
                shape=(self.N, self.M)
            )

        linear = torch.nn.Linear(
            self.M,
            self.N,
            bias=self.bias is not None
        )

        with torch.no_grad():
            linear.weight.mul_(0)
            linear.weight.add_(dense_weight)

            if self.bias is not None:
                linear.bias.mul_(0)
                linear.bias.add_(self.bias)

        return linear

    @property
    def weight(self):
        return self.W_val
    
    def forward(self, input):
        if self.jit and self.jit_ops.unroll:
            self.jit.add_unrolling(input.reshape(-1, input.shape[-1]).shape[0])
        return self.sparse_linear_fn.apply(input, self.W_val, self.W_idx, self.bias, self.N, self.jit, self.W_reg_idx)

    @torch.no_grad()
    def apply_further_mask(self, new_mask):
        """
            This function is used when we need to further sparsify a sparse module, e.g., gradual pruning.
        """

        indptr, indices = self.W_idx
        dense_weight = torch.Tensor(csr_matrix((
            self.W_val.data, 
            indices, 
            indptr
        ), shape=(self.N, self.M)).toarray()).float()

        dense_mask = torch.Tensor(csr_matrix((
            new_mask, 
            indices, 
            indptr
        ), shape=(self.N, self.M)).toarray()).float()
        
        W_val, W_idx = to_csr_2d(dense_weight * dense_mask)
        self.W_val = torch.nn.Parameter(W_val)
        self.W_idx = W_idx

    def __repr__(self):
        nnz = len(self.W_val)
        numel = self.N * self.M
        return f"SparseLinear([{self.N}, {self.M}], sp={1. - nnz/numel:.2f}, nnz={nnz})"
