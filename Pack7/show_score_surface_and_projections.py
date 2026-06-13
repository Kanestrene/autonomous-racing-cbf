from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "ml_cbf_pq_model3.npz"
FIGSIZE_3D = (10.8, 7.3)
FIGSIZE_2D = (12.2, 4.8)
SCORE_MIN = -100.0
SCORE_TOP_MARGIN = 15.0
MIN_INTERVAL_POINT_GAP = 0.9
LP_BETA = 2.0
LP_STEP_SIZE = 0.35
LP_VELOCITY_FRACTION = 0.08
LP_MAX_ITERATIONS = 120
INTERPOLATION_STYLES = (
    ("rbf", "Gaussian RBF", "#ff7f00", ":", 2.8),
)


def normalize_params(params, bounds, log_scale):
    params = np.asarray(params, dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    out = np.empty_like(params, dtype=float)

    linear = ~log_scale
    out[..., linear] = (params[..., linear] - low[linear]) / (high[linear] - low[linear])

    if np.any(log_scale):
        positive_low = np.maximum(low[log_scale], 1e-12)
        positive_params = np.maximum(params[..., log_scale], 1e-12)
        log_low = np.log(positive_low)
        log_high = np.log(high[log_scale])
        out[..., log_scale] = (np.log(positive_params) - log_low) / (
            log_high - log_low
        )

    return np.clip(out, 0.0, 1.0)


def rbf_interpolation_surface(train_normal, train_scores, bounds, log_scale, grid_size, bandwidth):
    p_min, p_max = bounds[0]
    q_min, q_max = bounds[1]
    p_min_plot = max(float(p_min), 1e-6)

    q_grid = np.linspace(q_min, q_max, grid_size)
    p_grid = np.geomspace(p_min_plot, p_max, grid_size)
    q_mesh, p_mesh, score_grid = rbf_surface_on_grid(
        train_normal=train_normal,
        train_scores=train_scores,
        q_grid=q_grid,
        p_grid=p_grid,
        bounds=bounds,
        log_scale=log_scale,
        bandwidth=bandwidth,
    )
    return q_grid, p_grid, q_mesh, p_mesh, score_grid


def rbf_surface_on_grid(train_normal, train_scores, q_grid, p_grid, bounds, log_scale, bandwidth):
    sigma = max(float(bandwidth), 1e-6)
    ridge = 1e-8

    train_normal = np.asarray(train_normal, dtype=float)
    train_scores = np.asarray(train_scores, dtype=float)
    q_mesh, p_mesh = np.meshgrid(q_grid, p_grid)
    raw_grid = np.c_[p_mesh.ravel(), q_mesh.ravel()]
    normal_grid = normalize_params(raw_grid, bounds, log_scale)

    dist2 = np.sum(
        (train_normal[:, None, :] - train_normal[None, :, :]) ** 2,
        axis=2,
    )
    kernel = np.exp(-0.5 * dist2 / sigma**2)
    coef = np.linalg.solve(kernel + ridge * np.eye(len(train_normal)), train_scores)

    grid_dist2 = np.sum(
        (normal_grid[:, None, :] - train_normal[None, :, :]) ** 2,
        axis=2,
    )
    grid_kernel = np.exp(-0.5 * grid_dist2 / sigma**2)
    score_grid = (grid_kernel @ coef).reshape(q_mesh.shape)
    return q_mesh, p_mesh, score_grid


def build_rbf_predictor(train_normal, train_scores, bandwidth):
    sigma = max(float(bandwidth), 1e-6)
    train_normal = np.asarray(train_normal, dtype=float)
    train_scores = np.asarray(train_scores, dtype=float)

    dist2 = np.sum(
        (train_normal[:, None, :] - train_normal[None, :, :]) ** 2,
        axis=2,
    )
    kernel = np.exp(-0.5 * dist2 / sigma**2)
    coef = np.linalg.solve(kernel + 1e-8 * np.eye(len(train_normal)), train_scores)

    def predict(normal_params):
        normal_params = np.atleast_2d(np.asarray(normal_params, dtype=float))
        query_dist2 = np.sum(
            (normal_params[:, None, :] - train_normal[None, :, :]) ** 2,
            axis=2,
        )
        query_kernel = np.exp(-0.5 * query_dist2 / sigma**2)
        return query_kernel @ coef

    return predict


def model_scalar(model, names, default):
    for name in names:
        if name in model.files:
            values = np.asarray(model[name]).reshape(-1)
            if len(values):
                return float(values[0])
    return float(default)


def build_svm_h_predictor(model, bounds, log_scale):
    try:
        from sklearn.svm import SVC
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to solve the filtered-RBF LP.") from exc

    train_normal = model["train_normal_params"].astype(float)
    labels = model["labels"].astype(int)
    if len(np.unique(labels)) < 2:
        raise RuntimeError("The LP requires both feasible and infeasible samples to reconstruct H.")

    c1 = model_scalar(model, ("kernel_c1", "svm_kernel_c1", "c1"), 0.8)
    c2 = model_scalar(model, ("kernel_c2", "svm_kernel_c2", "c2"), 0.5)
    degree = int(round(model_scalar(model, ("kernel_degree", "svm_kernel_degree", "degree"), 7)))
    svm_c = model_scalar(model, ("svm_c", "classifier_c"), 1.0)

    def polynomial_kernel(left, right):
        return (c1 + c2 * (left @ right.T)) ** degree

    classifier = SVC(C=svm_c, kernel=polynomial_kernel)
    classifier.fit(train_normal, labels)

    def predict(raw_params):
        normal = normalize_params(np.atleast_2d(raw_params), bounds, log_scale)
        return classifier.decision_function(normal)

    return predict


def finite_difference_gradient(function, params, bounds):
    params = np.asarray(params, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    span = bounds[:, 1] - bounds[:, 0]
    gradient = np.zeros_like(params)

    for idx in range(len(params)):
        step = max(float(span[idx]) * 1e-4, 1e-8)
        low = params.copy()
        high = params.copy()
        low[idx] = max(float(bounds[idx, 0]), float(params[idx] - step))
        high[idx] = min(float(bounds[idx, 1]), float(params[idx] + step))
        width = high[idx] - low[idx]
        if width > 0.0:
            gradient[idx] = (float(function(high)) - float(function(low))) / width

    return gradient


def solve_box_lp(objective, constraint_gradient, constraint_offset, velocity_limits):
    velocity_limits = np.asarray(velocity_limits, dtype=float)
    lower = -velocity_limits
    upper = velocity_limits
    candidates = [
        np.array([x_value, y_value], dtype=float)
        for x_value in (lower[0], upper[0])
        for y_value in (lower[1], upper[1])
    ]

    a_value, b_value = np.asarray(constraint_gradient, dtype=float)
    if abs(b_value) > 1e-12:
        for x_value in (lower[0], upper[0]):
            y_value = (-float(constraint_offset) - a_value * x_value) / b_value
            if lower[1] <= y_value <= upper[1]:
                candidates.append(np.array([x_value, y_value], dtype=float))
    if abs(a_value) > 1e-12:
        for y_value in (lower[1], upper[1]):
            x_value = (-float(constraint_offset) - b_value * y_value) / a_value
            if lower[0] <= x_value <= upper[0]:
                candidates.append(np.array([x_value, y_value], dtype=float))

    feasible = [
        velocity
        for velocity in candidates
        if float(np.dot(constraint_gradient, velocity) + constraint_offset) >= -1e-10
    ]
    if not feasible:
        return np.zeros(2, dtype=float)
    return min(feasible, key=lambda velocity: float(np.dot(objective, velocity)))


def solve_filtered_rbf_lp(model, params, normal_params, scores, keep_indices, bounds, log_scale, bandwidth):
    filtered_params = params[keep_indices]
    filtered_scores = scores[keep_indices]
    filtered_normal = normal_params[keep_indices]
    score_predictor = build_rbf_predictor(filtered_normal, filtered_scores, bandwidth)
    h_predictor = build_svm_h_predictor(model, bounds, log_scale)

    search_bounds = np.column_stack(
        [np.min(filtered_params, axis=0), np.max(filtered_params, axis=0)]
    )
    span = np.maximum(search_bounds[:, 1] - search_bounds[:, 0], 1e-12)
    velocity_limits = LP_VELOCITY_FRACTION * span
    current = filtered_params[int(np.argmax(filtered_scores))].astype(float).copy()

    def estimated_score(raw_params):
        normal = normalize_params(np.atleast_2d(raw_params), bounds, log_scale)
        return float(score_predictor(normal)[0])

    def h_value(raw_params):
        return float(h_predictor(np.atleast_2d(raw_params))[0])

    iterations = 0
    for iterations in range(1, LP_MAX_ITERATIONS + 1):
        grad_d = finite_difference_gradient(lambda point: -estimated_score(point), current, search_bounds)
        grad_h = finite_difference_gradient(h_value, current, search_bounds)
        h_current = h_value(current)
        velocity = solve_box_lp(
            objective=grad_d,
            constraint_gradient=grad_h,
            constraint_offset=LP_BETA * h_current,
            velocity_limits=velocity_limits,
        )
        next_params = np.clip(current + LP_STEP_SIZE * velocity, search_bounds[:, 0], search_bounds[:, 1])
        if np.linalg.norm((next_params - current) / span) < 1e-6:
            break
        current = next_params

    return {
        "params": current,
        "score": estimated_score(current),
        "h": h_value(current),
        "iterations": iterations,
    }


def select_samples(model, use_all_points):
    train_params = model["train_params"].astype(float)
    train_normal = model["train_normal_params"].astype(float)
    scores = model["scores"].astype(float)
    labels = model["labels"].astype(int)

    mask = np.ones_like(labels, dtype=bool) if use_all_points else labels > 0
    return train_params[mask], train_normal[mask], scores[mask], labels[mask]


def filter_surface_points(normal_params, scores, bins=7):
    normal_params = np.asarray(normal_params, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.zeros((0,), dtype=int)

    coords = np.clip(normal_params, 0.0, 1.0)
    bin_ids = np.floor(coords * int(bins)).astype(int)
    bin_ids = np.clip(bin_ids, 0, int(bins) - 1)

    best_by_bin = {}
    for idx, key in enumerate(map(tuple, bin_ids)):
        old_idx = best_by_bin.get(key)
        if old_idx is None or scores[idx] > scores[old_idx]:
            best_by_bin[key] = idx

    keep = set(best_by_bin.values())
    keep.add(int(np.argmax(scores)))
    return np.array(sorted(keep), dtype=int)


def filter_max_per_abscissa(x_values, scores, bins=12, log_x=False, x_limits=None):
    x_values = np.asarray(x_values, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if log_x:
        x_work_all = np.log10(np.maximum(x_values, 1e-12))
    else:
        x_work_all = x_values.copy()

    finite = np.isfinite(x_work_all) & np.isfinite(scores)
    x_work = x_work_all[finite]
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices) == 0:
        return np.zeros((0,), dtype=int)

    if x_limits is None:
        x_min = float(np.min(x_work))
        x_max = float(np.max(x_work))
    else:
        lo, hi = x_limits
        if log_x:
            x_min = float(np.log10(max(float(lo), 1e-12)))
            x_max = float(np.log10(max(float(hi), 1e-12)))
        else:
            x_min = float(lo)
            x_max = float(hi)
    if x_max <= x_min:
        return np.array([int(finite_indices[int(np.argmax(scores[finite]))])], dtype=int)

    edges = np.linspace(x_min, x_max, int(bins) + 1)
    keep = set()
    for bin_id in range(int(bins)):
        left = edges[bin_id]
        right = edges[bin_id + 1]
        if bin_id == int(bins) - 1:
            in_bin = (x_work >= left) & (x_work <= right)
        else:
            in_bin = (x_work >= left) & (x_work < right)
        if np.any(in_bin):
            local_indices = finite_indices[in_bin]
            local_scores = scores[local_indices]
            keep.add(int(local_indices[int(np.argmax(local_scores))]))

    keep.add(int(np.argmax(scores)))
    min_gap = MIN_INTERVAL_POINT_GAP * (x_max - x_min) / max(int(bins), 1)
    if min_gap > 0.0 and len(keep) > 1:
        merged = []
        for idx in sorted(keep, key=lambda item: scores[item], reverse=True):
            if all(abs(x_work_all[idx] - x_work_all[old_idx]) >= min_gap for old_idx in merged):
                merged.append(int(idx))
        keep = set(merged)
        keep.add(int(np.argmax(scores)))

    return np.array(sorted(keep), dtype=int)


def pchip_slopes(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    count = len(x_values)
    h = np.diff(x_values)
    delta = np.diff(y_values) / h

    if count == 2:
        return np.array([delta[0], delta[0]], dtype=float)

    slopes = np.zeros(count, dtype=float)
    for idx in range(1, count - 1):
        if delta[idx - 1] * delta[idx] <= 0.0:
            slopes[idx] = 0.0
        else:
            w1 = 2.0 * h[idx] + h[idx - 1]
            w2 = h[idx] + 2.0 * h[idx - 1]
            slopes[idx] = (w1 + w2) / (w1 / delta[idx - 1] + w2 / delta[idx])

    slopes[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(slopes[0]) != np.sign(delta[0]):
        slopes[0] = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(slopes[0]) > abs(3.0 * delta[0]):
        slopes[0] = 3.0 * delta[0]

    slopes[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(slopes[-1]) != np.sign(delta[-1]):
        slopes[-1] = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(slopes[-1]) > abs(3.0 * delta[-1]):
        slopes[-1] = 3.0 * delta[-1]

    return slopes


def evaluate_pchip(x_values, y_values, x_query):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    slopes = pchip_slopes(x_values, y_values)

    segment_ids = np.searchsorted(x_values, x_query, side="right") - 1
    segment_ids = np.clip(segment_ids, 0, len(x_values) - 2)
    h = x_values[segment_ids + 1] - x_values[segment_ids]
    t = (x_query - x_values[segment_ids]) / h

    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2

    return (
        h00 * y_values[segment_ids]
        + h10 * h * slopes[segment_ids]
        + h01 * y_values[segment_ids + 1]
        + h11 * h * slopes[segment_ids + 1]
    )


def evaluate_polynomial(x_values, y_values, x_query):
    degree = len(x_values) - 1
    center = 0.5 * (float(x_values[0]) + float(x_values[-1]))
    scale = max(0.5 * (float(x_values[-1]) - float(x_values[0])), 1e-12)
    coeffs = np.polyfit((x_values - center) / scale, y_values, degree)
    return np.polyval(coeffs, (x_query - center) / scale)


def evaluate_gaussian_rbf(x_values, y_values, x_query):
    span = max(float(x_values[-1] - x_values[0]), 1e-12)
    spacing = np.diff(x_values)
    sigma = max(float(np.median(spacing)), span / max(len(x_values) - 1, 1), 1e-12)

    dist2 = (x_values[:, None] - x_values[None, :]) ** 2
    kernel = np.exp(-0.5 * dist2 / sigma**2)
    coeffs = np.linalg.solve(kernel + 1e-10 * np.eye(len(x_values)), y_values)

    query_dist2 = (x_query[:, None] - x_values[None, :]) ** 2
    query_kernel = np.exp(-0.5 * query_dist2 / sigma**2)
    return query_kernel @ coeffs


def evaluate_interpolation_method(x_values, y_values, x_query, method):
    if method == "linear":
        return np.interp(x_query, x_values, y_values)
    if method == "pchip":
        return evaluate_pchip(x_values, y_values, x_query)
    if method == "polynomial":
        return evaluate_polynomial(x_values, y_values, x_query)
    if method == "rbf":
        return evaluate_gaussian_rbf(x_values, y_values, x_query)
    raise ValueError(f"Unknown interpolation method: {method}")


def interpolation_curve_through_points(x_values, scores, indices, x_grid, log_x=False, method="pchip"):
    x_values = np.asarray(x_values, dtype=float)
    scores = np.asarray(scores, dtype=float)
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    if log_x:
        x_selected = np.log10(np.maximum(x_values[indices], 1e-12))
        x_grid_work = np.log10(np.maximum(x_grid, 1e-12))
    else:
        x_selected = x_values[indices]
        x_grid_work = np.asarray(x_grid, dtype=float)
    y_selected = scores[indices]

    finite = np.isfinite(x_selected) & np.isfinite(y_selected)
    x_selected = x_selected[finite]
    y_selected = y_selected[finite]
    if len(x_selected) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    unique_x = np.unique(x_selected)
    unique_y = np.array(
        [np.max(y_selected[x_selected == x_value]) for x_value in unique_x],
        dtype=float,
    )
    if len(unique_x) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    curve_x_work = np.unique(
        np.r_[
            np.linspace(float(unique_x[0]), float(unique_x[-1]), len(x_grid_work)),
            unique_x,
        ]
    )
    curve_y = evaluate_interpolation_method(unique_x, unique_y, curve_x_work, method)

    if log_x:
        curve_x = 10.0**curve_x_work
    else:
        curve_x = curve_x_work
    return curve_x, curve_y


def interpolation_curves_through_points(x_values, scores, indices, x_grid, log_x=False):
    curves = []
    for method, label, color, linestyle, linewidth in INTERPOLATION_STYLES:
        curve_x, curve_y = interpolation_curve_through_points(
            x_values,
            scores,
            indices,
            x_grid,
            log_x=log_x,
            method=method,
        )
        if len(curve_x):
            curves.append(
                {
                    "x": curve_x,
                    "y": curve_y,
                    "label": label,
                    "color": color,
                    "linestyle": linestyle,
                    "linewidth": linewidth,
                }
            )
    return curves


def upper_envelope_curve(x_values, scores, x_grid, bins=10, quantile=0.85):
    x_values = np.asarray(x_values, dtype=float)
    scores = np.asarray(scores, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    finite = np.isfinite(x_values) & np.isfinite(scores)
    x_values = x_values[finite]
    scores = scores[finite]
    if len(x_values) < 2:
        return x_grid, np.full_like(x_grid, np.nan, dtype=float)

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    edges = np.linspace(x_min, x_max, int(bins) + 1)
    centers = []
    envelope = []

    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (x_values >= left) & (x_values <= right)
        if np.any(in_bin):
            centers.append(0.5 * (left + right))
            envelope.append(float(np.quantile(scores[in_bin], quantile)))

    if len(centers) < 2:
        order = np.argsort(x_values)
        centers = x_values[order]
        envelope = scores[order]
    else:
        best_idx = int(np.argmax(scores))
        centers.append(float(x_values[best_idx]))
        envelope.append(float(scores[best_idx]))
        order = np.argsort(centers)
        centers = np.asarray(centers, dtype=float)[order]
        envelope = np.asarray(envelope, dtype=float)[order]

    curve = np.interp(x_grid, centers, envelope)
    grid_step = max(float(np.mean(np.diff(x_grid))), 1e-12)
    sigma = max((x_max - x_min) * 0.11, grid_step)
    radius = max(2, int(np.ceil(3.0 * sigma / grid_step)))
    kernel_x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x * grid_step / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(curve, (radius, radius), mode="edge")
    curve = np.convolve(padded, kernel, mode="same")[radius:-radius]

    valid_grid = (x_grid >= x_min) & (x_grid <= x_max)
    return x_grid[valid_grid], curve[valid_grid]


def set_p_axis_ticks(ax, p_min, p_max, log_positions=False, max_ticks=6):
    p_min = max(float(p_min), 1e-12)
    p_max = max(float(p_max), p_min)
    exponents = range(int(np.floor(np.log10(p_min))), int(np.ceil(np.log10(p_max))) + 1)
    candidates = [
        multiplier * 10.0**exponent
        for exponent in exponents
        for multiplier in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
    ]
    ticks = np.array([p_min, *candidates, p_max], dtype=float)
    ticks = np.unique(ticks[(ticks >= p_min) & (ticks <= p_max)])
    if len(ticks) > max_ticks:
        indices = np.linspace(0, len(ticks) - 1, max_ticks).round().astype(int)
        ticks = ticks[indices]

    positions = np.log10(ticks) if log_positions else ticks
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{value:.3g}" for value in ticks])


def plot_3d(q_mesh, p_mesh, score_grid, params, scores, best_idx, title, lp_result=None):
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    p_log_mesh = np.log10(np.maximum(p_mesh, 1e-12))
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    plotted_score_grid = np.clip(score_grid, score_min, score_max)
    surface = ax.plot_surface(
        q_mesh,
        p_log_mesh,
        plotted_score_grid,
        cmap="viridis",
        alpha=0.72,
        linewidth=0,
        antialiased=True,
    )

    p_values = params[:, 0]
    q_values = params[:, 1]
    p_log_values = np.log10(np.maximum(p_values, 1e-12))

    best_p = float(p_values[best_idx])
    best_q = float(q_values[best_idx])
    best_score = float(scores[best_idx])
    ax.scatter(
        [best_q],
        [np.log10(max(best_p, 1e-12))],
        [best_score],
        c="#ffbf00",
        marker="*",
        s=240,
        edgecolors="black",
        linewidths=0.9,
        depthshade=False,
    )
    ax.scatter(
        q_values,
        p_log_values,
        scores,
        c="#1f77b4",
        marker="o",
        s=34,
        edgecolors="white",
        linewidths=0.5,
        depthshade=False,
    )
    if lp_result is not None:
        lp_p, lp_q = lp_result["params"]
        lp_score = float(lp_result["score"])
        lp_score_plot = float(np.clip(lp_score, score_min, score_max))
        ax.scatter(
            [lp_q],
            [np.log10(max(float(lp_p), 1e-12))],
            [lp_score_plot],
            c="#e41a1c",
            marker="X",
            s=100,
            edgecolors="black",
            linewidths=0.7,
            depthshade=False,
            zorder=8,
        )
        ax.text(
            float(lp_q),
            np.log10(max(float(lp_p), 1e-12)),
            lp_score_plot - 7.0,
            f"LP: p={lp_p:.2f}, q={lp_q:.2f}",
            fontsize=9,
        )
    ax.text(
        best_q,
        np.log10(max(best_p, 1e-12)),
        best_score + 5.0,
        f"p={best_p:.2f}, q={best_q:.2f}",
        fontsize=9,
    )

    set_p_axis_ticks(ax, np.min(p_mesh), np.max(p_mesh), log_positions=True)

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            markeredgecolor="white",
            markersize=8,
            label="Samples",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#ffbf00",
            markeredgecolor="black",
            markersize=13,
            label="Max score",
        ),
    ]
    if lp_result is not None:
        legend_items.append(
            Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                markerfacecolor="#e41a1c",
                markeredgecolor="black",
                markersize=9,
                label="Filtered-RBF LP result",
            )
        )

    ax.set_xlabel("q", labelpad=10)
    ax.set_ylabel("p", labelpad=10)
    ax.set_zlabel("interpolated score", labelpad=10)
    ax.set_title(title)
    ax.view_init(elev=27, azim=-58)
    ax.set_xlim(float(np.min(q_mesh)), float(np.max(q_mesh)))
    ax.set_ylim(float(np.min(p_log_mesh)), float(np.max(p_log_mesh)))
    ax.set_zlim(score_min, score_max)
    ax.legend(handles=legend_items, loc="upper right")
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.08, label="interpolated score")
    fig.tight_layout()
    return fig


def plot_top_view(q_mesh, p_mesh, score_grid, params, scores, best_idx, title, lp_result=None):
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    plotted_score_grid = np.clip(score_grid, score_min, score_max)

    contour = ax.contourf(
        q_mesh,
        p_mesh,
        plotted_score_grid,
        levels=24,
        cmap="viridis",
    )
    p_values = params[:, 0]
    q_values = params[:, 1]
    ax.scatter(
        q_values,
        p_values,
        c="#1f77b4",
        marker="o",
        s=34,
        edgecolors="white",
        linewidths=0.5,
        label="Samples",
        zorder=5,
    )

    best_p = float(p_values[best_idx])
    best_q = float(q_values[best_idx])
    ax.scatter(
        [best_q],
        [best_p],
        c="#ffbf00",
        marker="*",
        s=240,
        edgecolors="black",
        linewidths=0.9,
        label="Max score",
        zorder=6,
    )
    if lp_result is not None:
        lp_p, lp_q = lp_result["params"]
        ax.scatter(
            [lp_q],
            [lp_p],
            c="#e41a1c",
            marker="X",
            s=100,
            edgecolors="black",
            linewidths=0.7,
            label="Filtered-RBF LP result",
            zorder=7,
        )

    ax.set_yscale("log")
    ax.set_xlim(float(np.min(q_mesh)), float(np.max(q_mesh)))
    ax.set_ylim(float(np.min(p_mesh)), float(np.max(p_mesh)))
    set_p_axis_ticks(ax, np.min(p_mesh), np.max(p_mesh))
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True)
    fig.colorbar(contour, ax=ax, label="interpolated score")
    fig.tight_layout()
    return fig


def save_surface_views_pdf(q_mesh, p_mesh, score_grid, params, scores, best_idx, lp_result=None):
    fig = plt.figure(figsize=(15.8, 6.3))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0))
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_top = fig.add_subplot(grid[0, 1])

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    color_min = score_min
    color_max = score_max
    if np.isclose(color_min, color_max):
        color_min -= 0.5
        color_max += 0.5
    plotted_score_grid = np.clip(score_grid, score_min, score_max)
    p_log_mesh = np.log10(np.maximum(p_mesh, 1e-12))
    p_values = params[:, 0]
    q_values = params[:, 1]
    p_log_values = np.log10(np.maximum(p_values, 1e-12))
    best_p = float(p_values[best_idx])
    best_q = float(q_values[best_idx])
    best_score = float(scores[best_idx])

    surface = ax3d.plot_surface(
        q_mesh,
        p_log_mesh,
        plotted_score_grid,
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
        alpha=0.72,
        linewidth=0,
        antialiased=True,
    )
    ax3d.scatter(
        q_values,
        p_log_values,
        scores,
        c="#1f77b4",
        marker="o",
        s=30,
        edgecolors="white",
        linewidths=0.5,
        depthshade=False,
        label="Samples",
    )
    ax3d.scatter(
        [best_q],
        [np.log10(max(best_p, 1e-12))],
        [best_score],
        c="#ffbf00",
        marker="*",
        s=220,
        edgecolors="black",
        linewidths=0.9,
        depthshade=False,
        label="Max score",
    )
    if lp_result is not None:
        lp_p, lp_q = lp_result["params"]
        lp_score = float(np.clip(lp_result["score"], score_min, score_max))
        ax3d.scatter(
            [lp_q],
            [np.log10(max(float(lp_p), 1e-12))],
            [lp_score],
            c="#e41a1c",
            marker="X",
            s=95,
            edgecolors="black",
            linewidths=0.7,
            depthshade=False,
            label="Filtered-RBF LP result",
        )

    set_p_axis_ticks(ax3d, np.min(p_mesh), np.max(p_mesh), log_positions=True)
    ax3d.set_xlim(float(np.min(q_mesh)), float(np.max(q_mesh)))
    ax3d.set_ylim(float(np.min(p_log_mesh)), float(np.max(p_log_mesh)))
    ax3d.set_zlim(score_min, score_max)
    ax3d.set_xlabel("q")
    ax3d.set_ylabel("p")
    ax3d.set_zlabel("score")
    ax3d.set_title("RBF score surface")
    ax3d.view_init(elev=25, azim=-58)

    contour = ax_top.contourf(
        q_mesh,
        p_mesh,
        plotted_score_grid,
        levels=np.linspace(color_min, color_max, 25),
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
    )
    ax_top.scatter(
        q_values,
        p_values,
        c="#1f77b4",
        marker="o",
        s=34,
        edgecolors="white",
        linewidths=0.5,
        label="Samples",
        zorder=5,
    )
    ax_top.scatter(
        [best_q],
        [best_p],
        c="#ffbf00",
        marker="*",
        s=220,
        edgecolors="black",
        linewidths=0.9,
        label="Max score",
        zorder=6,
    )
    if lp_result is not None:
        lp_p, lp_q = lp_result["params"]
        ax_top.scatter(
            [lp_q],
            [lp_p],
            c="#e41a1c",
            marker="X",
            s=95,
            edgecolors="black",
            linewidths=0.7,
            label="Filtered-RBF LP result",
            zorder=7,
        )
    ax_top.set_yscale("log")
    ax_top.set_xlim(float(np.min(q_mesh)), float(np.max(q_mesh)))
    ax_top.set_ylim(float(np.min(p_mesh)), float(np.max(p_mesh)))
    set_p_axis_ticks(ax_top, np.min(p_mesh), np.max(p_mesh))
    ax_top.set_xlabel("q")
    ax_top.set_ylabel("p")
    ax_top.set_title("Top view")
    ax_top.legend(loc="lower right", fontsize=11)

    fig.subplots_adjust(left=0.03, right=0.92, bottom=0.08, top=0.93, wspace=0.18)
    fig.colorbar(
        surface,
        ax=[ax3d, ax_top],
        shrink=0.84,
        pad=0.04,
        label="interpolated score",
    )
    fig.savefig(BASE_DIR / "ml_cbf_rbf_score_surface_views.pdf", bbox_inches="tight")
    return fig


def plot_projections(q_grid, p_grid, score_grid, params, scores, best_idx, title):
    p_mean = np.mean(score_grid, axis=1)
    p_q10 = np.percentile(score_grid, 10, axis=1)
    p_q90 = np.percentile(score_grid, 90, axis=1)
    p_min_curve = np.min(score_grid, axis=1)
    p_max_curve = np.max(score_grid, axis=1)

    q_mean = np.mean(score_grid, axis=0)
    q_q10 = np.percentile(score_grid, 10, axis=0)
    q_q90 = np.percentile(score_grid, 90, axis=0)
    q_min_curve = np.min(score_grid, axis=0)
    q_max_curve = np.max(score_grid, axis=0)

    p_values = params[:, 0]
    q_values = params[:, 1]
    best_p = float(p_values[best_idx])
    best_q = float(q_values[best_idx])
    best_score = float(scores[best_idx])
    score_max = float(np.max(scores)) + SCORE_TOP_MARGIN

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_2D)

    ax = axes[0]
    ax.fill_between(p_grid, p_min_curve, p_max_curve, color="#9ecae1", alpha=0.23, label="Surface min-max")
    ax.fill_between(p_grid, p_q10, p_q90, color="#3182bd", alpha=0.20, label="Surface 10-90%")
    ax.plot(p_grid, p_mean, color="black", linewidth=2.0, label="Projected mean")
    ax.scatter([best_p], [best_score], c="#ffbf00", marker="*", s=230, edgecolors="black", linewidths=0.9, label="Max score", zorder=5)
    ax.scatter(p_values, scores, c="#1f77b4", s=42, edgecolors="white", linewidths=0.6, label="Samples", zorder=6)
    ax.set_xscale("log")
    ax.set_xlim(float(np.min(p_grid)), float(np.max(p_grid)))
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_title("Projection onto (p, score)")
    ax.set_ylim(SCORE_MIN, score_max)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    ax = axes[1]
    ax.fill_between(q_grid, q_min_curve, q_max_curve, color="#9ecae1", alpha=0.23, label="Surface min-max")
    ax.fill_between(q_grid, q_q10, q_q90, color="#3182bd", alpha=0.20, label="Surface 10-90%")
    ax.plot(q_grid, q_mean, color="black", linewidth=2.0, label="Projected mean")
    ax.scatter([best_q], [best_score], c="#ffbf00", marker="*", s=230, edgecolors="black", linewidths=0.9, label="Max score", zorder=5)
    ax.scatter(q_values, scores, c="#1f77b4", s=42, edgecolors="white", linewidths=0.6, label="Samples", zorder=6)
    ax.set_xlim(float(np.min(q_grid)), float(np.max(q_grid)))
    ax.set_xlabel("q")
    ax.set_ylabel("score")
    ax.set_title("Projection onto (q, score)")
    ax.set_ylim(SCORE_MIN, score_max)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_point_only_projections(q_grid, p_grid, params, scores, best_idx, title):
    p_values = params[:, 0]
    q_values = params[:, 1]
    best_p = float(p_values[best_idx])
    best_q = float(q_values[best_idx])
    best_score = float(scores[best_idx])
    score_max = float(np.max(scores)) + SCORE_TOP_MARGIN
    p_curve_x, p_curve_score = upper_envelope_curve(
        np.log10(np.maximum(p_values, 1e-12)),
        scores,
        np.log10(np.maximum(p_grid, 1e-12)),
    )
    q_curve_x, q_curve_score = upper_envelope_curve(q_values, scores, q_grid)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_2D)

    ax = axes[0]
    ax.scatter(
        [best_p],
        [best_score],
        c="#ffbf00",
        marker="*",
        s=230,
        edgecolors="black",
        linewidths=0.9,
        label="Max score",
        zorder=5,
    )
    ax.scatter(
        p_values,
        scores,
        c="#1f77b4",
        s=42,
        edgecolors="white",
        linewidths=0.6,
        label="Samples",
        zorder=6,
    )
    ax.plot(
        10.0**p_curve_x,
        p_curve_score,
        color="#e41a1c",
        linewidth=3.0,
        label="Upper envelope",
        zorder=4,
    )
    ax.set_xscale("log")
    ax.set_xlim(float(np.min(p_grid)), float(np.max(p_grid)))
    ax.set_ylim(SCORE_MIN, score_max)
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_title("Points only: score vs p")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    ax = axes[1]
    ax.scatter(
        [best_q],
        [best_score],
        c="#ffbf00",
        marker="*",
        s=230,
        edgecolors="black",
        linewidths=0.9,
        label="Max score",
        zorder=5,
    )
    ax.scatter(
        q_values,
        scores,
        c="#1f77b4",
        s=42,
        edgecolors="white",
        linewidths=0.6,
        label="Samples",
        zorder=6,
    )
    ax.plot(
        q_curve_x,
        q_curve_score,
        color="#e41a1c",
        linewidth=3.0,
        label="Upper envelope",
        zorder=4,
    )
    ax.set_xlim(float(np.min(q_grid)), float(np.max(q_grid)))
    ax.set_ylim(SCORE_MIN, score_max)
    ax.set_xlabel("q")
    ax.set_ylabel("score")
    ax.set_title("Points only: score vs q")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_filtered_surface_points(
    q_grid,
    p_grid,
    params,
    scores,
    keep_indices,
    bounds,
    log_scale,
    score_bandwidth,
    lp_result,
    title,
):
    filtered_params = params[keep_indices]
    filtered_scores = scores[keep_indices]
    all_normal = normalize_params(params, bounds, log_scale)
    all_p_values = params[:, 0]
    all_q_values = params[:, 1]
    p_values = filtered_params[:, 0]
    q_values = filtered_params[:, 1]
    surface_q_grid = np.linspace(float(np.min(all_q_values)), float(np.max(all_q_values)), len(q_grid))
    surface_p_grid = np.geomspace(
        max(float(np.min(all_p_values)), 1e-12),
        float(np.max(all_p_values)),
        len(p_grid),
    )
    q_mesh, p_mesh, filtered_score_grid = rbf_surface_on_grid(
        train_normal=all_normal,
        train_scores=scores,
        q_grid=surface_q_grid,
        p_grid=surface_p_grid,
        bounds=bounds,
        log_scale=log_scale,
        bandwidth=score_bandwidth,
    )
    best_idx = int(np.argmax(filtered_scores))
    best_p = float(p_values[best_idx])
    best_q = float(q_values[best_idx])
    best_score = float(filtered_scores[best_idx])
    sample_score_min = float(np.min(scores))
    sample_score_max = float(np.max(scores))
    plotted_score_grid = np.clip(filtered_score_grid, sample_score_min, sample_score_max)
    projection_score_max = sample_score_max + SCORE_TOP_MARGIN

    fig = plt.figure(figsize=(14.6, 4.9))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_p = fig.add_subplot(1, 3, 2)
    ax_q = fig.add_subplot(1, 3, 3)

    all_p_log_values = np.log10(np.maximum(all_p_values, 1e-12))
    ax3d.plot_surface(
        q_mesh,
        np.log10(np.maximum(p_mesh, 1e-12)),
        plotted_score_grid,
        cmap="viridis",
        alpha=0.58,
        linewidth=0,
        antialiased=True,
    )
    ax3d.scatter(
        [best_q],
        [np.log10(max(best_p, 1e-12))],
        [best_score],
        c="#ffbf00",
        marker="*",
        s=230,
        edgecolors="black",
        linewidths=0.9,
        depthshade=False,
        label="Max score",
    )
    ax3d.scatter(
        all_q_values,
        all_p_log_values,
        scores,
        c="#1f77b4",
        marker="o",
        s=38,
        edgecolors="white",
        linewidths=0.6,
        depthshade=False,
        label="All points",
    )
    if lp_result is not None:
        lp_p, lp_q = lp_result["params"]
        ax3d.scatter(
            [lp_q],
            [np.log10(max(float(lp_p), 1e-12))],
            [lp_result["score"]],
            c="#e41a1c",
            marker="X",
            s=90,
            edgecolors="black",
            linewidths=0.7,
            depthshade=False,
            label="Filtered-RBF LP result",
        )
    p_ticks = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 15.0])
    p_ticks = p_ticks[(p_ticks >= float(np.min(p_grid))) & (p_ticks <= float(np.max(p_grid)))]
    if len(p_ticks):
        ax3d.set_yticks(np.log10(p_ticks))
        ax3d.set_yticklabels([f"{v:g}" for v in p_ticks])
    ax3d.set_xlim(float(np.min(q_grid)), float(np.max(q_grid)))
    ax3d.set_ylim(np.log10(float(np.min(p_grid))), np.log10(float(np.max(p_grid))))
    ax3d.set_zlim(sample_score_min, sample_score_max)
    ax3d.set_xlabel("q", labelpad=8)
    ax3d.set_ylabel("p", labelpad=8)
    ax3d.set_zlabel("score", labelpad=8)
    ax3d.set_title("RBF surface from all available points")
    ax3d.view_init(elev=25, azim=-58)
    ax3d.legend(loc="upper right")

    ax_p.scatter(
        [best_p],
        [best_score],
        c="#ffbf00",
        marker="*",
        s=230,
        edgecolors="black",
        linewidths=0.9,
        label="Max score",
        zorder=5,
    )
    ax_p.scatter(
        p_values,
        filtered_scores,
        c="#1f77b4",
        s=42,
        edgecolors="white",
        linewidths=0.6,
        label="Filtered points",
        zorder=6,
    )
    if lp_result is not None:
        lp_p, _ = lp_result["params"]
        ax_p.scatter(
            [lp_p],
            [lp_result["score"]],
            c="#e41a1c",
            marker="X",
            s=90,
            edgecolors="black",
            linewidths=0.7,
            label="Filtered-RBF LP result",
            zorder=7,
        )
    ax_p.set_xscale("log")
    ax_p.set_xlim(float(np.min(p_grid)), float(np.max(p_grid)))
    ax_p.set_ylim(SCORE_MIN, projection_score_max)
    ax_p.set_xlabel("p")
    ax_p.set_ylabel("score")
    ax_p.set_title("Filtered: score vs p")
    ax_p.grid(True, alpha=0.25)
    ax_p.legend(loc="lower right", frameon=True)

    ax_q.scatter(
        [best_q],
        [best_score],
        c="#ffbf00",
        marker="*",
        s=230,
        edgecolors="black",
        linewidths=0.9,
        label="Max score",
        zorder=5,
    )
    ax_q.scatter(
        q_values,
        filtered_scores,
        c="#1f77b4",
        s=42,
        edgecolors="white",
        linewidths=0.6,
        label="Filtered points",
        zorder=6,
    )
    if lp_result is not None:
        _, lp_q = lp_result["params"]
        ax_q.scatter(
            [lp_q],
            [lp_result["score"]],
            c="#e41a1c",
            marker="X",
            s=90,
            edgecolors="black",
            linewidths=0.7,
            label="Filtered-RBF LP result",
            zorder=7,
        )
    ax_q.set_xlim(float(np.min(q_grid)), float(np.max(q_grid)))
    ax_q.set_ylim(SCORE_MIN, projection_score_max)
    ax_q.set_xlabel("q")
    ax_q.set_ylabel("score")
    ax_q.set_title("Filtered: score vs q")
    ax_q.grid(True, alpha=0.25)
    ax_q.legend(loc="lower right", frameon=True)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_max_per_abscissa_points(q_grid, p_grid, params, scores, p_indices, q_indices, title):
    p_values = params[:, 0]
    q_values = params[:, 1]
    best_idx = int(np.argmax(scores))
    best_p = float(p_values[best_idx])
    best_q = float(q_values[best_idx])
    best_score = float(scores[best_idx])
    score_max = float(np.max(scores)) + SCORE_TOP_MARGIN
    p_curves = interpolation_curves_through_points(
        p_values,
        scores,
        p_indices,
        p_grid,
        log_x=True,
    )
    q_curves = interpolation_curves_through_points(
        q_values,
        scores,
        q_indices,
        q_grid,
        log_x=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_2D)

    ax = axes[0]
    ax.scatter(
        p_values,
        scores,
        c="#b7c4d4",
        s=28,
        edgecolors="white",
        linewidths=0.4,
        label="Other samples",
    )
    for curve in p_curves:
        ax.plot(
            curve["x"],
            curve["y"],
            color=curve["color"],
            linestyle=curve["linestyle"],
            linewidth=curve["linewidth"],
            label=curve["label"],
            zorder=4,
        )
    ax.scatter(
        [best_p],
        [best_score],
        c="#ffbf00",
        marker="*",
        s=230,
        edgecolors="black",
        linewidths=0.9,
        label="Global max",
        zorder=5,
    )
    ax.scatter(
        p_values[p_indices],
        scores[p_indices],
        c="#1f77b4",
        s=54,
        edgecolors="white",
        linewidths=0.7,
        label="Max per p interval",
        zorder=6,
    )
    ax.set_xscale("log")
    ax.set_xlim(float(np.min(p_grid)), float(np.max(p_grid)))
    ax.set_ylim(SCORE_MIN, score_max)
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_title("Max score in each p interval")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=8)

    ax = axes[1]
    ax.scatter(
        q_values,
        scores,
        c="#b7c4d4",
        s=28,
        edgecolors="white",
        linewidths=0.4,
        label="Other samples",
    )
    for curve in q_curves:
        ax.plot(
            curve["x"],
            curve["y"],
            color=curve["color"],
            linestyle=curve["linestyle"],
            linewidth=curve["linewidth"],
            label=curve["label"],
            zorder=4,
        )
    ax.scatter(
        [best_q],
        [best_score],
        c="#ffbf00",
        marker="*",
        s=230,
        edgecolors="black",
        linewidths=0.9,
        label="Global max",
        zorder=5,
    )
    ax.scatter(
        q_values[q_indices],
        scores[q_indices],
        c="#1f77b4",
        s=54,
        edgecolors="white",
        linewidths=0.7,
        label="Max per q interval",
        zorder=6,
    )
    ax.set_xlim(float(np.min(q_grid)), float(np.max(q_grid)))
    ax.set_ylim(SCORE_MIN, score_max)
    ax.set_xlabel("q")
    ax.set_ylabel("score")
    ax.set_title("Max score in each q interval")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=8)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "ml_cbf_gaussian_rbf_max_per_interval_2d.pdf", bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Show the ML-CBF score surface and 2D projections without saving files."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--grid-size", type=int, default=140)
    parser.add_argument("--surface-bins", type=int, default=7)
    parser.add_argument("--all-points", action="store_true", help="Use all samples instead of only feasible samples.")
    parser.add_argument("--skip-filtered-rbf-lp", action="store_true", help="Skip the LP based on the filtered RBF score.")
    args = parser.parse_args()

    model = np.load(args.model, allow_pickle=True)
    bounds = model["param_bounds"].astype(float)
    log_scale = model["param_log_scale"].astype(bool)
    score_bandwidth = float(model["score_bandwidth"][0])

    params, normal_params, scores, _ = select_samples(model, args.all_points)
    if len(scores) == 0:
        raise RuntimeError("No samples available for plotting.")

    q_grid, p_grid, q_mesh, p_mesh, score_grid = rbf_interpolation_surface(
        train_normal=normal_params,
        train_scores=scores,
        bounds=bounds,
        log_scale=log_scale,
        grid_size=args.grid_size,
        bandwidth=score_bandwidth,
    )

    best_idx = int(np.argmax(scores))
    sample_name = "all samples" if args.all_points else "feasible samples"
    surface_indices = filter_surface_points(
        normal_params,
        scores,
        bins=args.surface_bins,
    )
    lp_result = None
    if not args.skip_filtered_rbf_lp:
        lp_result = solve_filtered_rbf_lp(
            model=model,
            params=params,
            normal_params=normal_params,
            scores=scores,
            keep_indices=surface_indices,
            bounds=bounds,
            log_scale=log_scale,
            bandwidth=score_bandwidth,
        )
    p_interval_indices = filter_max_per_abscissa(
        params[:, 0],
        scores,
        bins=args.surface_bins,
        log_x=True,
        x_limits=(float(np.min(p_grid)), float(np.max(p_grid))),
    )
    q_interval_indices = filter_max_per_abscissa(
        params[:, 1],
        scores,
        bins=args.surface_bins,
        log_x=False,
        x_limits=(float(np.min(q_grid)), float(np.max(q_grid))),
    )

    plot_3d(
        q_mesh=q_mesh,
        p_mesh=p_mesh,
        score_grid=score_grid,
        params=params,
        scores=scores,
        best_idx=best_idx,
        title=f"RBF-interpolated score surface from {sample_name}",
        lp_result=lp_result,
    )
    plot_top_view(
        q_mesh=q_mesh,
        p_mesh=p_mesh,
        score_grid=score_grid,
        params=params,
        scores=scores,
        best_idx=best_idx,
        title=f"Top view of the RBF score surface ({sample_name})",
        lp_result=lp_result,
    )
    save_surface_views_pdf(
        q_mesh=q_mesh,
        p_mesh=p_mesh,
        score_grid=score_grid,
        params=params,
        scores=scores,
        best_idx=best_idx,
        lp_result=lp_result,
    )
    plot_projections(
        q_grid=q_grid,
        p_grid=p_grid,
        score_grid=score_grid,
        params=params,
        scores=scores,
        best_idx=best_idx,
        title=f"2D projections of the RBF score surface ({sample_name})",
    )
    plot_point_only_projections(
        q_grid=q_grid,
        p_grid=p_grid,
        params=params,
        scores=scores,
        best_idx=best_idx,
        title=f"Point-only lateral projections ({sample_name})",
    )
    plot_filtered_surface_points(
        q_grid=q_grid,
        p_grid=p_grid,
        params=params,
        scores=scores,
        keep_indices=surface_indices,
        bounds=bounds,
        log_scale=log_scale,
        score_bandwidth=score_bandwidth,
        lp_result=lp_result,
        title=f"Filtered surface points ({sample_name})",
    )
    plot_max_per_abscissa_points(
        q_grid=q_grid,
        p_grid=p_grid,
        params=params,
        scores=scores,
        p_indices=p_interval_indices,
        q_indices=q_interval_indices,
        title=f"Maximum point per abscissa interval ({sample_name})",
    )

    best_params = params[best_idx]
    print(f"Samples used: {len(scores)} ({sample_name})")
    print(
        "Best point: "
        f"p={best_params[0]:.6f}, q={best_params[1]:.6f}, score={scores[best_idx]:.6f}"
    )
    print(f"Filtered surface points: {len(surface_indices)}")
    print(f"Max-per-p-interval points: {len(p_interval_indices)}")
    print(f"Max-per-q-interval points: {len(q_interval_indices)}")
    if lp_result is not None:
        lp_p, lp_q = lp_result["params"]
        print(
            "Filtered-RBF LP result: "
            f"p={lp_p:.6f}, q={lp_q:.6f}, "
            f"estimated score={lp_result['score']:.6f}, "
            f"H={lp_result['h']:.6f}, "
            f"iterations={lp_result['iterations']}"
        )
    plt.show()


if __name__ == "__main__":
    main()
