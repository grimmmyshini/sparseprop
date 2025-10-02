from matplotlib.patches import Patch
import torch
import matplotlib.pyplot as plt
from timeit import timeit
from tqdm import tqdm
import numpy as np
from math import log

from sparseprop.modules.functions import SparseLinearFunction
from sparseprop.modules.linear_jit import LinearJIT
from sparseprop.utils import JITOptions
from sparseprop.modules.utils import to_csr_2d, from_csr_2d

import os, sys


def init_weight(dense_weight, bias, jit_ops: JITOptions):
    W_reg_idx = None
    W_dense_idx = None
    reg_tile_ops = None
    W_idx = None
    if jit_ops and jit_ops.reg_tiling:
        reg_tile_ops, W_val, W_idx, W_reg_idx, W_dense_idx = LinearJIT.solve_sparse_jam(
            dense_weight, jit_ops.reg_tile_size[0]
        )
    else:
        W_val, W_idx = to_csr_2d(dense_weight)
    assert bias is None or isinstance(bias, torch.nn.Parameter), (
        f"bias is not a parameter but it's {type(bias)}"
    )
    return W_val, W_idx, W_reg_idx, W_dense_idx, reg_tile_ops


# Define a consistent style
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 200,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
    }
)

# Shared color palette (muted modern)
COLOR_PALETTE = {
    "jit": "#4C72B0",  # muted blue
    "register_tiling": "#DD8452",  # warm orange
    "psc": "#55A868",  # green
}
MODULE_COLORS = [
    "#1961D6",
    "#05A72B",
    "#BE393D",
    "#8172B3",
    "#64B5CD",
    "#CAA005",
    "#DD8DD6",
    "#6D3B0C",
    "#231C5F",
    "#04646B",
    "#F37500",
    "#E206B2",
]

# Font size for all plots.
fnt = 18


def plot(times, title, folder, sparsities, module_names, scales):
    os.makedirs(folder, exist_ok=True)
    for sparsity in sparsities:
        plt.figure(figsize=(10, 6))
        for i, mn in enumerate(module_names):
            plt.plot(
                scales,
                times[mn][sparsity],
                "-o",
                label=mn,
                color=MODULE_COLORS[i % len(MODULE_COLORS)],
                alpha=0.9,
                linewidth=3,
                markersize=7,
            )
        plt.xlabel("Input Size (log2(N))", fontsize=fnt)
        plt.ylabel("Runtime (Logarithmic, s)", fontsize=fnt)
        plt.legend(fontsize=fnt)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{title}_{sparsity}.pdf"))
        plt.close()


def plot_scatter(
    times_dict, module_names, sparsities, title, folder, remove_outliers=True
):
    os.makedirs(folder, exist_ok=True)

    for sparsity in sparsities:
        plt.figure(figsize=(10, 6))
        xs = np.arange(len(module_names))

        for i, module in enumerate(module_names):
            runtimes = np.array(times_dict[module][sparsity])

            if remove_outliers and len(runtimes) > 0:
                q1 = np.percentile(runtimes, 25)
                q3 = np.percentile(runtimes, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                runtimes = runtimes[
                    (runtimes >= lower_bound) & (runtimes <= upper_bound)
                ]

            # jitter so points don’t overlap
            jitter = np.random.uniform(-0.1, 0.1, size=len(runtimes))
            plt.scatter(
                xs[i] + jitter,
                runtimes,
                alpha=0.7,
                s=50,
                color=MODULE_COLORS[i % len(MODULE_COLORS)],
                edgecolor="black",
                linewidth=0.4,
            )

        plt.xticks(xs, module_names, rotation=45, fontsize=fnt)
        plt.ylabel("Runtime (s)", fontsize=fnt)
        plt.xlabel("Input Size (log2(N))", fontsize=fnt)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{title}_{sparsity}.pdf"))
        plt.close()


def plot_stacked(times, input_sizes, title, folder):
    os.makedirs(folder, exist_ok=True)

    modules = list(times.keys())
    num_modules = len(modules)
    num_sizes = len(input_sizes)

    bar_width = 0.5 / max(1, num_modules)
    xs = np.arange(num_sizes)

    plt.figure(figsize=(14, 6))

    hatches = ["..", "xx", "oo", "--", "++", "**"]

    module_patches = []

    for i, module in enumerate(modules):
        offsets = xs - (bar_width * num_modules / 2) + i * bar_width
        bottoms = np.zeros(num_sizes)

        for comp, color in COLOR_PALETTE.items():
            vals = times[module].get(comp, [0] * num_sizes)
            if len(vals) < num_sizes:
                vals = list(vals) + [0] * (num_sizes - len(vals))

            plt.bar(
                offsets,
                vals,
                bottom=bottoms,
                width=bar_width,
                color=color,
                alpha=0.9,
                hatch=hatches[i % len(hatches)],
                edgecolor="black",
                label=comp if i == 0 else None,
            )
            bottoms += np.array(vals)

        module_patches.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                hatch=hatches[i % len(hatches)],
                edgecolor="black",
                label=module,
            )
        )

    plt.xticks(xs, input_sizes, fontsize=fnt)
    plt.ylabel("Runtime (Logarithmic, s)", fontsize=fnt)
    plt.xlabel("Input Size (log2(N))", fontsize=fnt)

    comp_patches = [
        Patch(facecolor=color, edgecolor="black", label=comp)
        for comp, color in COLOR_PALETTE.items()
    ]
    comp_legend = plt.gca().legend(
        handles=comp_patches, title="Component", loc="upper left", fontsize=fnt
    )
    plt.gca().add_artist(comp_legend)
    plt.legend(
        handles=module_patches,
        title="Module",
        loc="upper left",
        bbox_to_anchor=(0.28, 1),
        ncol=2,
        fontsize=fnt,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{title}.pdf"))
    plt.close()


def benchmark_runtimes(
    B,
    N,
    input,
    W_val,
    W_idx,
    W_reg_idx,
    grad_out,
    W,
    module_options,
    reg_tile_ops,
    forward_times,
    backward_times,
    reps=500,
    do_back=False,
):
    ctx = torch.autograd.function.FunctionCtx()
    ctx.bench = True
    jit = None

    if module_options:
        jit = LinearJIT(W_val, W_idx, W.shape, module_options, reg_tile_ops, B=B)
        if module_options.unroll:
            jit.add_unrolling(input.reshape(-1, input.shape[-1]).shape[0])

    # Warmup
    for _ in range(reps):
        SparseLinearFunction.forward(ctx, input, W_val, W_idx, None, N, jit, W_reg_idx)
        if do_back:
            SparseLinearFunction.backward(ctx, grad_out)

    ft = timeit(
        lambda: SparseLinearFunction.forward(
            ctx, input, W_val, W_idx, None, N, jit, W_reg_idx
        ),
        number=reps,
    )
    forward_times.append(log(ft / reps))
    if do_back:
        bt = timeit(lambda: SparseLinearFunction.backward(ctx, grad_out), number=reps)
        backward_times.append(log(bt / reps))


def benchmark_overhead(
    B,
    N,
    input,
    W_val,
    W_idx,
    W,
    module_options: JITOptions,
    reg_tile_ops,
    forward_times,
    reps=10,
):
    jitt = 0
    psct = 0
    regt = 0
    for _ in range(reps):
        jit = LinearJIT(
            W_val,
            W_idx,
            W.shape,
            module_options,
            reg_tile_ops,
            B=B,
            immediate_jit=False,
        )
        if module_options.unroll:
            jit.add_unrolling(input.reshape(-1, input.shape[-1]).shape[0])

        if module_options.reg_tiling:
            regt += timeit(
                lambda: LinearJIT.solve_sparse_jam(W, module_options.reg_tile_size[0]),
                number=1,
            )
        elif module_options.psc:
            psct += timeit(lambda: jit.find_strides(N, W_idx[0], W_idx[1]), number=1)

        jitt += timeit(lambda: jit._jit_forward(), number=1) + timeit(
            lambda: jit._jit_backward(), number=1
        )

    forward_times["jit"].append(jitt / reps)
    if module_options.reg_tiling:
        forward_times["register_tiling"].append(regt / reps)
    elif module_options.psc:
        forward_times["psc"].append(psct / reps)


def do_regular_bench(
    scales, sparse, module_names, jit_ops, reps, patterned=False, do_back=False
):
    forward_times = {}
    backward_times = {}
    for module_name, module_options in zip(module_names, jit_ops):
        forward_times[module_name] = {}
        backward_times[module_name] = {}
        for sparsity in tqdm(sparse, desc=module_name):
            forward_times[module_name][sparsity] = []
            backward_times[module_name][sparsity] = []

            for scale in scales:
                diff = scale // 2
                M = 2 ** (scale + diff)
                N = 2 ** (scale - diff)
                B = 2**scale

                W = torch.randn(N, M)
                if patterned:
                    repeat = 4
                    pattern = torch.randn(repeat, M) > sparsity
                    mask = pattern
                    for _ in range(1, N // repeat):
                        mask = torch.cat((mask, pattern))
                else:
                    mask = torch.rand_like(W) > sparsity
                W *= mask
                W_val, W_idx, W_reg_idx, _, reg_tile_ops = init_weight(
                    W, bias=None, jit_ops=module_options
                )

                X = torch.randn(B, M)
                grad_out = torch.randn(B, N)

                benchmark_runtimes(
                    B,
                    N,
                    X,
                    W_val,
                    W_idx,
                    W_reg_idx,
                    grad_out,
                    W,
                    module_options,
                    reg_tile_ops,
                    forward_times[module_name][sparsity],
                    backward_times[module_name][sparsity],
                    reps=reps,
                    do_back=do_back,
                )
    return forward_times, backward_times


def do_sparse_bench(scale, sparsities, module_names, jit_ops, reps=500):
    M = 2**scale
    N = 2**scale
    B = 2**scale
    forward_times = {}
    backward_times = {}
    for module_name, module_options in zip(module_names, jit_ops):
        forward_times[module_name] = {}
        backward_times[module_name] = {}
        for sparsity in tqdm(sparsities, desc=module_name):
            forward_times[module_name][sparsity] = []
            backward_times[module_name][sparsity] = []
            for _ in range(reps):
                W = torch.randn(N, M)
                mask = torch.rand_like(W) > sparsity
                W *= mask
                W_val, W_idx, W_reg_idx, _, reg_tile_ops = init_weight(
                    W, bias=None, jit_ops=module_options
                )

                X = torch.randn(B, M)
                grad_out = torch.randn(B, N)

                benchmark_runtimes(
                    B,
                    N,
                    X,
                    W_val,
                    W_idx,
                    W_reg_idx,
                    grad_out,
                    W,
                    module_options,
                    reg_tile_ops,
                    forward_times[module_name][sparsity],
                    backward_times[module_name][sparsity],
                    reps=100,
                )
    return forward_times, backward_times


def do_overhead_bench(scales, sparsity, module_names, jit_ops, reps=10):
    forward_times = {}
    for module_name, module_options in zip(module_names, jit_ops):
        forward_times[module_name] = {"jit": [], "psc": [], "register_tiling": []}
        for scale in tqdm(scales, desc=module_name):
            M = 2**scale
            N = 2**scale
            B = 2**scale
            W = torch.randn(N, M)
            mask = torch.rand_like(W) > sparsity
            W *= mask
            W_val, W_idx, W_reg_idx, _, reg_tile_ops = init_weight(
                W, bias=None, jit_ops=module_options
            )
            X = torch.randn(B, M)

            benchmark_overhead(
                B,
                N,
                X,
                W_val,
                W_idx,
                W,
                module_options,
                reg_tile_ops,
                forward_times[module_name],
                reps,
            )
    return forward_times


if __name__ == "__main__":
    sparsities = [0.8, 0.9, 0.95]
    torch.manual_seed(11)

    assert len(sys.argv) > 1, "Need tag name for plots!!"
    tag = sys.argv[1]

    # PSC
    # module_names = [
    #     "SparseProp",
    #     "SparseJit",
    #     # "SparseJit_Unrolled",
    #     "SparseJit_PSC",
    # ]
    # jit_ops = [
    #     None,
    #     # JITOptions(do_parallel=True),
    #     # JITOptions(do_parallel=True, do_only_scalar=True),
    #     JITOptions(do_parallel=False, do_psc=True),
    # ]
    # forward_times, backward_times = do_regular_bench(
    #     scales, sparsities, module_names, jit_ops, reps=100000, patterned=True
    # )
    # plot(forward_times, tag + "_fwd_psc", "plots/linear", sparsities, module_names, scales)
    # plot(backward_times, tag + "_bwd_psc", "plots/linear", sparsities, module_names, scales)

    # FWD Times, short
    module_names = [
        "SparseProp",
        "SparseJit",
        "SparseJit_Unrolled",
        "SparseJit_PSC",
    ]
    jit_ops = [
        None,
        JITOptions(do_parallel=True),
        JITOptions(do_parallel=True, do_only_scalar=True),
        JITOptions(do_parallel=True, do_psc=True),
    ]
    scales = range(4, 9)
    forward_times, _ = do_regular_bench(
        scales, sparsities, module_names, jit_ops, reps=100
    )
    plot(forward_times, tag + "_fwd", "plots/linear", sparsities, module_names, scales)

    # BWD Times, short
    module_names = [
        "SparseProp",
        "SparseJit",
        "SparseJit_PSC",
    ]
    jit_ops = [
        None,
        JITOptions(do_parallel=True),
        JITOptions(do_parallel=True, do_psc=True),
    ]
    _, backward_times = do_regular_bench(
        scales, sparsities, module_names, jit_ops, reps=2000, do_back=True
    )
    plot(backward_times, tag + "_bwd", "plots/linear", sparsities, module_names, scales)

    # Stacked overhead plots
    module_names = [
        "SparseJit_PSC",
        "SparseJit",
        "SparseJit_Unrolled",  # Fully unrolled loops
        "SparseJit_RegisterTiled_2",
        "SparseJit_RegisterTiled_4",
    ]
    jit_ops = [
        JITOptions(do_parallel=True, do_psc=True),
        JITOptions(do_parallel=True),
        JITOptions(do_parallel=True, do_only_scalar=True),
        JITOptions(do_parallel=True, do_reg_tiling=True, reg_tile_size=(2, 16)),
        JITOptions(do_parallel=True, do_reg_tiling=True, reg_tile_size=(4, 16)),
    ]
    for sparsity in [0.8, 0.9, 0.99]:
        forward_overhead_times = do_overhead_bench(
            range(4, 9), sparsity, module_names, jit_ops
        )
        plot_stacked(
            forward_overhead_times,
            scales,
            tag + "_fwd_stacked_" + str(sparsity),
            "plots/linear",
        )

    # Scatter plot
    module_names = [
        "SparseProp",
        "SparseJit",
        "SparseJit_RegisterTiled_2",
        "SparseJit_RegisterTiled_4",
    ]
    jit_ops = [
        None,
        JITOptions(do_parallel=True),
        JITOptions(do_parallel=True, do_reg_tiling=True, reg_tile_size=(2, 16)),
        JITOptions(do_parallel=True, do_reg_tiling=True, reg_tile_size=(4, 16)),
    ]
    forward_sparse_times, backward_sparse_times = do_sparse_bench(
        10, sparsities, module_names, jit_ops, reps=150
    )
    plot_scatter(
        forward_sparse_times, module_names, sparsities, tag + "_scatter", "plots/linear"
    )
    plot_scatter(
        backward_sparse_times,
        module_names,
        sparsities,
        tag + "_scatter",
        "plots/linear",
    )

    # Longer benchmarks
    module_names = [
        "SparseJit_RegisterTiled_2",
        "SparseJit_RegisterTiled_4",
        "SparseProp",
        "SparseJit",
        # "SparseJit_Unrolled",
    ]
    jit_ops = [
        JITOptions(do_parallel=True, do_reg_tiling=True, reg_tile_size=(2, 16)),
        JITOptions(do_parallel=True, do_reg_tiling=True, reg_tile_size=(4, 16)),
        None,
        JITOptions(do_parallel=True),
        # JITOptions(do_parallel=True, do_only_scalar=True),
    ]
    scale_long = range(9, 13, 1)
    forward_times, _ = do_regular_bench(
        scale_long, sparsities, module_names, jit_ops, reps=5
    )
    plot(
        forward_times,
        tag + "_fwd_longer",
        "plots/linear",
        sparsities,
        module_names,
        scale_long,
    )
