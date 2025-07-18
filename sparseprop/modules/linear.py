import torch
from scipy.sparse import csr_matrix

from sparseprop.modules.functions import SparseLinearFunction
from sparseprop.modules.utils import to_csr_2d, from_csr_2d
from sparseprop.modules.linear_jit import LinearJIT, JITOptions

class SparseLinear(torch.nn.Module):
    def __init__(self, dense_weight, bias=None, jit_fn = False, jit_ops = JITOptions()):
        super(SparseLinear, self).__init__()
        self.N, self.M = dense_weight.shape

        W_val, W_idx = to_csr_2d(dense_weight)
        self.W_val = torch.nn.Parameter(W_val)
        self.W_idx = W_idx

        self.sparse_linear_fn = SparseLinearFunction()
        assert bias is None or isinstance(bias, torch.nn.Parameter), f"bias is not a parameter but it's {type(bias)}"
        self.bias = bias

        self.jit_ops = jit_ops 
        self.jit = LinearJIT(jit_ops) if jit_fn else None

    @staticmethod
    def from_dense(module):
        return SparseLinear(
            dense_weight=module.weight.data,
            bias=None if module.bias is None else torch.nn.Parameter(module.bias.data.clone())
        )

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
        return self.sparse_linear_fn.apply(input, self.W_val, self.W_idx, self.bias, self.N, self.jit)

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
