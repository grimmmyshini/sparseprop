import torch
from sparseprop.modules import SparseLinear
from sparseprop.utils import error
import math
from copy import deepcopy

if __name__ == '__main__':
    torch.manual_seed(11)

    B = 128 # batch size
    M = 512 # input height
    N = 256 # input width
    sparsity = .9

    W = torch.randn(N, M)
    bias = torch.randn(N)
    mask = torch.rand_like(W) > sparsity
    W *= mask

    Y_orig = torch.randn(B, N)

    X_orig = torch.randn(B, M)
    X_orig.requires_grad_()
    X_orig.retain_grad()

    torch_X = X_orig.clone()
    torch_X.retain_grad()
    torch_Y = Y_orig.clone()
    linear = torch.nn.Linear(M, N, bias=True)

    print(linear.weight.shape, W.shape)
    with torch.no_grad():
        linear.weight.mul_(0.)
        linear.weight.add_(W)
        linear.bias.mul_(0.)
        linear.bias.add_(bias)

    torch_O = linear(torch_X)
    torch.mean((torch_O - torch_Y) ** 2).backward()
    torch_X_grad = torch_X.grad
    torch_W_grad = linear.weight.grad[linear.weight != 0]

    our_X = X_orig.clone()
    our_X.retain_grad()
    our_Y = Y_orig.clone()
    splinear = SparseLinear(W, bias=torch.nn.Parameter(deepcopy(bias)))
    our_O = splinear(our_X)
    torch.mean((our_O - our_Y) ** 2).backward()
    our_X_grad = our_X.grad
    our_W_grad = splinear.W_val.grad

    print('[Forward]\n O error:', error(our_O, torch_O))
    print('[Backward]\n X grad error:', error(our_X_grad, torch_X_grad), '\n W grad error:', error(our_W_grad, torch_W_grad))
    print('[Backward]\n bias grad error:', error(splinear.bias.grad, linear.bias.grad))

    our_X_jit = X_orig.clone()
    our_X_jit.retain_grad()
    our_Y_jit = Y_orig.clone()
    splinear_jit = SparseLinear(W, bias=torch.nn.Parameter(deepcopy(bias)), jit_fn=True)
    our_O_jit = splinear_jit(our_X_jit)
    torch.mean((our_O_jit - our_Y_jit) ** 2).backward()
    our_X_grad_jit = our_X_jit.grad
    our_W_grad_jit = splinear_jit.W_val.grad

    print('\n[Forward JIT]\n O error:', error(our_O_jit, torch_O))
    print('[Backward  JIT]\n X grad error:', error(our_X_grad_jit, torch_X_grad), '\n W grad error:', error(our_W_grad_jit, torch_W_grad))
    print('[Backward JIT]\n bias grad error:', error(splinear_jit.bias.grad, linear.bias.grad))