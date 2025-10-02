import torch
from sparseprop.modules import SparseLinear
from sparseprop.modules.linear_jit import JITOptions
from sparseprop.utils import error
import math
from copy import deepcopy

if __name__ == "__main__":
    # torch.manual_seed(11)

    B = 2**6  # batch size
    N = 2**6  # input width
    M = 2**6  # input height
    sparsity = 0.9

    W = torch.randn(N, M)
    bias = torch.randn(N)
    mask = torch.rand_like(W) > sparsity
    # mask = [
    #     [1, 0, 1, 0, 0, 1, 0, 0],
    #     [0, 1, 0, 1, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 1, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 1, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0],
    # ]
    # mask = [
    #     [1, 0, 1, 0, 0, 1, 0, 0],
    #     [0, 1, 0, 1, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 1, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 1, 1, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    # ]
    # mask = [
    #     [1, 0, 1, 0, 1, 0, 1, 0],
    #     [0, 1, 0, 1, 0, 1, 0, 1],
    #     [1, 1, 1, 0, 1, 1, 1, 0],
    #     [0, 1, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 1, 0, 1, 0, 1, 0],
    #     [0, 1, 0, 1, 0, 1, 0, 1],
    #     [1, 1, 1, 0, 1, 1, 1, 0],
    #     [0, 1, 0, 1, 0, 1, 0, 1],
    # ]
    # mask = torch.tensor(mask, dtype=torch.float32)
    W *= mask
    Y_orig = torch.randn(B, N)

    X_orig = torch.randn(B, M)
    X_orig.requires_grad_()
    X_orig.retain_grad()

    torch_X = X_orig.clone()
    torch_X.retain_grad()
    torch_Y = Y_orig.clone()
    linear = torch.nn.Linear(M, N, bias=True)

    with torch.no_grad():
        linear.weight.mul_(0.0)
        linear.weight.add_(W)
        linear.bias.mul_(0.0)
        linear.bias.add_(bias)

    torch_O = linear(torch_X)
    torch.mean((torch_O - torch_Y) ** 2).backward()
    torch_X_grad = torch_X.grad
    torch_W_grad = linear.weight.grad[linear.weight != 0]
    torch_W_grad_dense = linear.weight.grad * mask

    to_compare = [
        [W, torch.nn.Parameter(deepcopy(bias))],
        [
            W,
            torch.nn.Parameter(deepcopy(bias)),
            True,
            JITOptions(do_parallel=True, do_only_scalar=True),
        ],
    ]
    module_names = ["Vanilla SparseLinear", "Jit SparseLinear"]

    for args, name in zip(to_compare, module_names):
        module = SparseLinear(*args)
        our_X = X_orig.clone()
        our_X.retain_grad()
        our_Y = Y_orig.clone()
        our_O = module(our_X)
        torch.mean((our_O - our_Y) ** 2).backward()
        our_X_grad = our_X.grad
        if module.jit_ops.reg_tiling:
            # Need to look at full dense grad otherwise things get a bit weird...
            # register tiling representation is very different from typical csr
            W_grad_err = error(module.get_dense_grad(), torch_W_grad_dense)
        else:
            W_grad_err = error(module.W_val.grad, torch_W_grad)

        print("-" * 20, name, "-" * 20)
        print("[Forward]\n O error:", error(our_O, torch_O))
        print(
            "[Backward]\n X grad error:",
            error(our_X_grad, torch_X_grad),
            "\n W grad error:",
            W_grad_err,
        )
        print(
            "[Backward]\n bias grad error:", error(module.bias.grad, linear.bias.grad)
        )
