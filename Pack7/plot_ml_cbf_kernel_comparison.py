from pathlib import Path

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "ml_cbf_pq_model.npz"

# Change these values to compare different polynomial kernels.
# Each tuple is (c1, c2) in K(y,z) = (c1 + c2*y^T*z)^degree.
KERNEL_PARAMS = [
    (0.8, 0.2),
    (0.8, 0.5),
    (0.8, 1.0),
    (1.5, 0.5),
]

DEFAULT_OUTPUT_STEM = "ml_cbf_pq_c1_c2_comparison"
DEFAULT_GRID_SIZE = 260
DEFAULT_RANDOM_SEED = 7


def poly_kernel(x, z, c1, c2, degree):
    return (c1 + c2 * np.asarray(x) @ np.asarray(z).T) ** degree


def denormalize_params(normal_params, bounds, log_scale):
    normal = np.asarray(normal_params, dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    out = np.empty_like(normal, dtype=float)

    linear = ~log_scale
    out[..., linear] = low[linear] + normal[..., linear] * (high[linear] - low[linear])

    log_low = np.log(low[log_scale])
    log_high = np.log(high[log_scale])
    out[..., log_scale] = np.exp(log_low + normal[..., log_scale] * (log_high - log_low))
    return out


def normalize_params(params, bounds, log_scale):
    params = np.asarray(params, dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    out = np.empty_like(params, dtype=float)

    linear = ~log_scale
    out[..., linear] = (params[..., linear] - low[linear]) / (high[linear] - low[linear])

    log_low = np.log(low[log_scale])
    log_high = np.log(high[log_scale])
    out[..., log_scale] = (np.log(params[..., log_scale]) - log_low) / (
        log_high - log_low
    )
    return np.clip(out, 0.0, 1.0)


def train_svm_smo(kernel_matrix, labels, C, seed, tol=1e-4, max_passes=35, max_iter=30000):
    """Small SMO solver for the binary kernel SVM dual problem."""
    rng = np.random.default_rng(seed)
    y = np.asarray(labels, dtype=float)
    n_samples = len(y)
    alpha = np.zeros(n_samples, dtype=float)
    bias = 0.0
    passes = 0
    iterations = 0

    while passes < max_passes and iterations < max_iter:
        changed = 0
        decision = (alpha * y) @ kernel_matrix + bias
        error = decision - y

        for i in rng.permutation(n_samples):
            error_i = error[i]
            violates_lower = y[i] * error_i < -tol and alpha[i] < C
            violates_upper = y[i] * error_i > tol and alpha[i] > 0.0
            if not (violates_lower or violates_upper):
                continue

            candidates = np.where(np.arange(n_samples) != i)[0]
            free = np.where(
                (alpha > 1e-8)
                & (alpha < C - 1e-8)
                & (np.arange(n_samples) != i)
            )[0]
            if len(free):
                candidates = free

            j = int(candidates[np.argmax(np.abs(error[candidates] - error_i))])
            error_j = error[j]
            alpha_i_old = alpha[i]
            alpha_j_old = alpha[j]

            if y[i] != y[j]:
                lower = max(0.0, alpha_j_old - alpha_i_old)
                upper = min(C, C + alpha_j_old - alpha_i_old)
            else:
                lower = max(0.0, alpha_i_old + alpha_j_old - C)
                upper = min(C, alpha_i_old + alpha_j_old)

            if abs(lower - upper) < 1e-12:
                continue

            eta = 2.0 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
            if eta >= -1e-12:
                continue

            alpha[j] = alpha_j_old - y[j] * (error_i - error_j) / eta
            alpha[j] = min(upper, max(lower, alpha[j]))

            if abs(alpha[j] - alpha_j_old) < 1e-6:
                alpha[j] = alpha_j_old
                continue

            alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])

            bias_1 = (
                bias
                - error_i
                - y[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i, i]
                - y[j] * (alpha[j] - alpha_j_old) * kernel_matrix[i, j]
            )
            bias_2 = (
                bias
                - error_j
                - y[i] * (alpha[i] - alpha_i_old) * kernel_matrix[i, j]
                - y[j] * (alpha[j] - alpha_j_old) * kernel_matrix[j, j]
            )

            if 0.0 < alpha[i] < C:
                bias = bias_1
            elif 0.0 < alpha[j] < C:
                bias = bias_2
            else:
                bias = 0.5 * (bias_1 + bias_2)

            changed += 1
            decision = (alpha * y) @ kernel_matrix + bias
            error = decision - y

        passes = passes + 1 if changed == 0 else 0
        iterations += 1

    return alpha, bias, iterations


def make_grid(bounds, log_scale, grid_size):
    p_min, p_max = bounds[0]
    q_min, q_max = bounds[1]
    q_grid = np.linspace(q_min, q_max, grid_size)
    p_grid = np.geomspace(p_min, p_max, grid_size)
    q_mesh, p_mesh = np.meshgrid(q_grid, p_grid)

    # Parameter order in the model is [p, q].
    raw_grid = np.c_[p_mesh.ravel(), q_mesh.ravel()]
    normal_grid = normalize_params(raw_grid, bounds, log_scale)
    return q_mesh, p_mesh, normal_grid


def parse_kernel_params(values):
    if not values:
        return KERNEL_PARAMS

    params = []
    for value in values:
        try:
            c1_text, c2_text = value.split(",", maxsplit=1)
            params.append((float(c1_text), float(c2_text)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid pair {value!r}; use the form c1,c2, for example 0.8,0.5"
            ) from exc

    return params


def plot_comparison(model_path, output_stem, kernel_params, grid_size, seed):
    model = np.load(model_path, allow_pickle=True)
    train_normal = model["train_normal_params"].astype(float)
    train_params = model["train_params"].astype(float)
    labels = model["labels"].astype(float)
    bounds = model["param_bounds"].astype(float)
    log_scale = model["param_log_scale"].astype(bool)
    degree = int(model["svm_degree"][0])
    C = float(model["svm_C"][0])

    p_values = train_params[:, 0]
    q_values = train_params[:, 1]
    feasible = labels > 0
    infeasible = labels < 0

    q_mesh, p_mesh, normal_grid = make_grid(bounds, log_scale, grid_size)
    rows = math.ceil(len(kernel_params) / 2)
    cols = 2 if len(kernel_params) > 1 else 1

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5.7 * cols, 3.9 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    summary = []
    for ax, (c1, c2) in zip(axes.flat, kernel_params):
        kernel_matrix = poly_kernel(train_normal, train_normal, c1, c2, degree)
        alpha, bias, iterations = train_svm_smo(
            kernel_matrix,
            labels,
            C=C,
            seed=seed,
        )

        train_score = (alpha * labels) @ kernel_matrix + bias
        prediction = np.where(train_score >= 0.0, 1.0, -1.0)
        accuracy = float(np.mean(prediction == labels))
        support_count = int(np.sum(alpha > 1e-6))

        grid_kernel = poly_kernel(normal_grid, train_normal, c1, c2, degree)
        grid_score = (grid_kernel @ (alpha * labels) + bias).reshape(q_mesh.shape)

        ax.contourf(
            q_mesh,
            p_mesh,
            grid_score,
            levels=[-1e9, 0.0, 1e9],
            colors=["#f8d7da", "#d7ecff"],
            alpha=0.48,
        )
        ax.contour(q_mesh, p_mesh, grid_score, levels=[0.0], colors="black", linewidths=1.6)
        ax.scatter(
            q_values[infeasible],
            p_values[infeasible],
            s=30,
            c="#d62728",
            marker="x",
            linewidths=1.2,
            label="Infeasible",
        )
        ax.scatter(
            q_values[feasible],
            p_values[feasible],
            s=30,
            c="#1f77b4",
            marker="o",
            edgecolors="white",
            linewidths=0.5,
            label="Feasible",
        )

        ax.set_title(f"c1={c1:g}, c2={c2:g} | acc={accuracy:.1%}, SV={support_count}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.23)
        summary.append((c1, c2, accuracy, support_count, iterations))

    for ax in axes.flat[len(kernel_params) :]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("q")
    for ax in axes[:, 0]:
        ax.set_ylabel("p")

    handles, labels_text = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_text,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(f"Effect of polynomial-kernel parameters, degree {degree}", y=1.055)
    fig.tight_layout()

    png_path = BASE_DIR / f"{output_stem}.png"
    pdf_path = BASE_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path, summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate ML-CBF polynomial-kernel SVM comparison plots."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--kernel",
        action="append",
        default=None,
        help="Kernel pair c1,c2. Can be repeated, e.g. --kernel 0.8,0.5 --kernel 1.5,0.5",
    )
    args = parser.parse_args()

    kernel_params = parse_kernel_params(args.kernel)
    png_path, pdf_path, summary = plot_comparison(
        model_path=args.model,
        output_stem=args.output_stem,
        kernel_params=kernel_params,
        grid_size=args.grid_size,
        seed=args.seed,
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    for c1, c2, accuracy, support_count, iterations in summary:
        print(
            f"c1={c1:g}, c2={c2:g}, "
            f"accuracy={accuracy:.4f}, support_vectors={support_count}, iterations={iterations}"
        )


if __name__ == "__main__":
    main()
