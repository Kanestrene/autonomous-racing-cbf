from pathlib import Path

import numpy as np

import ml_cbf
import qp
from controller import build_spline_path, omega_to_delta, rate_limit, wrap_to_pi


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OBSTACLE_PATH_OFFSET_M = 5.0

DEFAULT_BASELINE_PARAMS = {
    "class_k_p": 3.0,
    "class_k_q": 1.0,
    "alpha": 3.0,
    "margin": 0.05,
    "lookahead_l": 0.20,
    "barrier_lookahead_l": 0.10,
    "Wv": 100000.0,
    "Ww": 1.0,
    "p_slack": 50.0,
    "eps_clf": 2.0,
}

DEFAULT_WAYPOINTS = [
    (3.0, 3.0),
    (2.6, 3.5),
    (2.2, 4.2),
    (2.0, 5.0),
    (2.0, 6.2),
    (2.0, 7.4),
    (2.0, 8.8),
    (2.0, 10.2),
    (2.0, 11.6),
    (2.0, 13.0),
    (2.2, 13.6),
    (2.6, 14.5),
    (3.1, 14.8),
    (3.7, 15.0),
    (4.2, 15.0),
    (4.8, 14.9),
    (5.3, 14.6),
    (5.6, 14.1),
    (5.7, 13.5),
    (5.6, 12.6),
    (5.5, 11.6),
    (5.5, 10.6),
    (5.5, 9.8),
    (5.7, 9.2),
    (6.0, 8.7),
    (6.6, 8.4),
    (7.4, 8.4),
    (8.2, 8.5),
    (8.8, 8.9),
    (9.1, 9.6),
    (9.3, 10.4),
    (9.5, 11.6),
    (9.7, 12.6),
    (9.9, 13.4),
    (10.2, 14.0),
    (10.8, 14.6),
    (11.6, 15.0),
    (12.6, 15.0),
    (13.6, 15.0),
    (14.6, 14.8),
    (15.4, 14.4),
    (16.0, 13.6),
    (16.0, 12.4),
    (16.0, 11.2),
    (16.0, 10.0),
    (16.0, 8.8),
    (16.0, 7.6),
    (16.0, 6.4),
    (16.0, 5.0),
    (15.5, 3.6),
    (14.2, 2.6),
    (12.0, 2.0),
    (10.8, 2.0),
    (9.6, 2.0),
    (8.4, 2.0),
    (7.2, 2.0),
    (6.0, 2.0),
    (4.0, 2.5),
    (3.0, 3.0),
]


def build_problem(
    ds=0.01,
    variation_id=0,
    obstacle_path_offset_m=DEFAULT_OBSTACLE_PATH_OFFSET_M,
):
    px, py, pyaw, s = build_spline_path(DEFAULT_WAYPOINTS, ds=ds)
    obstacles = make_obstacles(
        px,
        py,
        pyaw,
        variation_id=variation_id,
        path_offset_m=obstacle_path_offset_m,
    )
    inner_bar = np.loadtxt(BASE_DIR / "barreira_suavizada_interna.txt")
    outer_bar = np.loadtxt(BASE_DIR / "barreira_suavizada_externa.txt")
    return px, py, pyaw, s, obstacles, inner_bar, outer_bar


def shift_path_indices(px, py, idxs, offset_m):
    path_s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))))
    path_length = float(path_s[-1])
    if path_length <= 0.0:
        return np.asarray(idxs, dtype=int)

    shifted_s = np.mod(path_s[idxs] + float(offset_m), path_length)
    return np.searchsorted(path_s, shifted_s, side="left")


def make_obstacles(px, py, pyaw, n_obs=5, variation_id=0, path_offset_m=0.0):
    n_path = len(px)
    idxs = np.linspace(0, n_path - 1, n_obs + 2, dtype=int)[1:-1]
    idxs = shift_path_indices(px, py, idxs, path_offset_m)

    obstacles = []
    for k, idx in enumerate(idxs):
        nx = -np.sin(pyaw[idx])
        ny = np.cos(pyaw[idx])
        side = (-1) ** k
        offset = 0.30

        obstacles.append(
            {
                "x": float(px[idx] + side * offset * nx),
                "y": float(py[idx] + side * offset * ny),
                "r": 0.35,
            }
        )

    return obstacles


def start_state_for_variation(variation_id):
    dx = 0.08 * np.sin(variation_id)
    dy = 0.08 * np.cos(0.6 * variation_id)
    dyaw = np.deg2rad(4.0 * np.sin(0.4 * variation_id))
    return 2.0 + dx, 6.0 + dy, np.deg2rad(90.0) + dyaw, 0.0


def min_distance_to_polyline(x, y, poly):
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2:
        return np.inf

    if np.hypot(*(poly[0] - poly[-1])) > 1e-9:
        poly = np.vstack([poly, poly[0]])

    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    ap = np.array([x, y], dtype=float) - a
    denom = np.sum(ab * ab, axis=1)
    denom = np.maximum(denom, 1e-12)
    t = np.sum(ap * ab, axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    d = np.linalg.norm(closest - np.array([x, y]), axis=1)
    return float(np.min(d))


def min_obstacle_clearance(x, y, obstacles, ellipse_ab, margin):
    if not obstacles:
        return 10.0

    robot_radius = max(float(ellipse_ab[0]), float(ellipse_ab[1]))
    clearances = [
        np.hypot(x - obs["x"], y - obs["y"]) - float(obs.get("r", 0.0)) - robot_radius - margin
        for obs in obstacles
    ]
    return float(min(clearances))


def min_barrier_clearance(x, y, inner_bar, outer_bar, ellipse_ab, margin):
    robot_radius = max(float(ellipse_ab[0]), float(ellipse_ab[1]))
    d_inner = min_distance_to_polyline(x, y, inner_bar)
    d_outer = min_distance_to_polyline(x, y, outer_bar)
    return float(min(d_inner, d_outer) - robot_radius - margin)


def score_episode(metrics):
    safety_margin = min(metrics["min_obstacle_clearance"], metrics["min_barrier_clearance"])
    mean_abs_dv = float(metrics.get("mean_abs_dv", 0.0))
    total_abs_dv = float(metrics.get("total_abs_dv", 0.0))
    progress = float(metrics["progress_ratio"])
    score = 70.0 * progress

    if metrics["completed"]:
        score += 100.0

    #score += 180.0 * np.clip(safety_margin, -0.5, 1.0)
    #score -= 140.0 * max(0.0, -safety_margin) ** 2
    #score -= 130.0 * float(metrics["mean_abs_cte"])
    #score -= 1000.0 * mean_abs_dv
    #score -= 10.0 * total_abs_dv
    #score -= 3.0 * int(metrics["qp_failures"])

    #score += 18.0 * np.clip(safety_margin, -0.5, 1.0)
    score -= 3.0 * total_abs_dv
    score -= 100.0 * max(0.0, -safety_margin) ** 2
    score -= 200.0 * float(metrics["mean_abs_cte"])
    score -= 3.0 * int(metrics["qp_failures"])

    if metrics["collided"]:
        score -= 80.0

    return float(score)


def run_episode(
    params=None,
    ml_cbf_model=None,
    variation_id=0,
    horizon=50.0,
    dt=0.02,
    record_stride=10,
    record_history=False,
    solver_preference=("quadprog", "daqp"),
):
    params = {**DEFAULT_BASELINE_PARAMS, **(params or {})}
    px, py, pyaw, s, obstacles, inner_bar, outer_bar = build_problem(variation_id=variation_id)
    n_path = len(px)

    x, y, yaw, v = start_state_for_variation(variation_id)

    v_ref = 2.0
    w_max = 2.5
    ellipse_ab = (0.30, 0.20)

    L = 0.26
    delta = 0.0
    delta_max = np.deg2rad(25.0)
    delta_rate_max = np.deg2rad(300.0)

    last_near = 0
    prev_near_idx = None
    lap_progress_idx = 0.0
    qp_failures = 0
    ctes = []
    v_safe_values = []
    contexts = []
    selected_candidates = []
    active_params_history = []
    history = {"x": [], "y": [], "yaw": [], "v": [], "w": []}

    min_obs_clear = np.inf
    min_bar_clear = np.inf
    collided = False

    steps = int(horizon / dt)
    for k in range(steps):
        v_nom = v_ref
        w_nom = 0.0
        robot_state = (x, y, yaw)

        u_safe, clf_info = qp.cbf_clf_qp_filter(
            u_nom=(v_nom, w_nom),
            robot_state=robot_state,
            obstacles=obstacles,
            px=px,
            py=py,
            pyaw=pyaw,
            s=s,
            last_path_idx=last_near,
            ellipse_ab=ellipse_ab,
            margin=params["margin"],
            lookahead_l=params["lookahead_l"],
            barrier_lookahead_l=params["barrier_lookahead_l"],
            alpha=params["alpha"],
            class_k_p=params["class_k_p"],
            class_k_q=params["class_k_q"],
            eps_clf=params["eps_clf"],
            q_clf=(1.0, 10.0, 0.01),
            W=(params["Wv"], params["Ww"]),
            p_slack=params["p_slack"],
            v_ref=v_ref,
            v_bounds=(0.0, 2.0),
            w_bounds=(-w_max, w_max),
            ml_cbf_selector=ml_cbf_model,
            solver_preference=solver_preference,
        )

        if clf_info.get("qp_failed", False):
            qp_failures += 1

        active = clf_info.get("active_cbf_params", {})
        active_margin = float(active.get("margin", params["margin"]))
        if "ml_cbf_candidate" in active:
            selected_candidates.append(int(active["ml_cbf_candidate"]))
            active_params_history.append(dict(active))

        if k % max(1, record_stride) == 0:
            contexts.append(
                ml_cbf.make_context_features(
                    robot_state=robot_state,
                    u_nom=(v_nom, w_nom),
                    obstacles=obstacles,
                    clf_info=clf_info,
                    ellipse_ab=ellipse_ab,
                )
            )

        v_safe, w_safe = u_safe
        v_safe_values.append(float(v_safe))
        last_near = clf_info["idx"]
        ctes.append(abs(float(clf_info["ey"])))

        if prev_near_idx is None:
            prev_near_idx = last_near
        else:
            delta_idx = last_near - prev_near_idx
            if delta_idx < -n_path / 2:
                delta_idx += n_path
            elif delta_idx > n_path / 2:
                delta_idx -= n_path
            lap_progress_idx += max(0.0, float(delta_idx))
            prev_near_idx = last_near

        kappa_max = np.tan(delta_max) / L
        w_max_speed = abs(v_safe) * kappa_max
        w_safe = float(np.clip(w_safe, -w_max_speed, w_max_speed))

        delta_cmd = omega_to_delta(w_safe, v_safe, L, v_min=0.2)
        delta_cmd = float(np.clip(delta_cmd, -delta_max, delta_max))
        delta = rate_limit(delta_cmd, delta, du_max=delta_rate_max * dt)

        x += float(v_safe) * np.cos(yaw) * dt
        y += float(v_safe) * np.sin(yaw) * dt
        yaw = wrap_to_pi(yaw + (float(v_safe) / L) * np.tan(delta) * dt)
        v = float(v_safe)

        obs_clear = min_obstacle_clearance(x, y, obstacles, ellipse_ab, active_margin)
        bar_clear = min_barrier_clearance(x, y, inner_bar, outer_bar, ellipse_ab, active_margin)
        min_obs_clear = min(min_obs_clear, obs_clear)
        min_bar_clear = min(min_bar_clear, bar_clear)

        if record_history:
            history["x"].append(float(x))
            history["y"].append(float(y))
            history["yaw"].append(float(yaw))
            history["v"].append(float(v_safe))
            history["w"].append(float(w_safe))

        if min(obs_clear, bar_clear) < -0.35:
            collided = True
            break

        if lap_progress_idx >= (n_path - 1):
            break

    progress_ratio = min(1.0, lap_progress_idx / max(1.0, float(n_path - 1)))
    completed = progress_ratio >= 1.0
    if len(v_safe_values) > 1:
        abs_dv = np.abs(np.diff(np.asarray(v_safe_values, dtype=float)))
        mean_abs_dv = float(np.mean(abs_dv))
        total_abs_dv = float(np.sum(abs_dv))
    else:
        mean_abs_dv = 0.0
        total_abs_dv = 0.0

    metrics = {
        "completed": bool(completed),
        "collided": bool(collided),
        "progress_ratio": float(progress_ratio),
        "steps": int(k + 1),
        "qp_failures": int(qp_failures),
        "mean_abs_cte": float(np.mean(ctes)) if ctes else np.inf,
        "max_abs_cte": float(np.max(ctes)) if ctes else np.inf,
        "mean_abs_dv": mean_abs_dv,
        "total_abs_dv": total_abs_dv,
        "min_obstacle_clearance": float(min_obs_clear),
        "min_barrier_clearance": float(min_bar_clear),
        "contexts": np.asarray(contexts, dtype=float),
        "selected_candidates": selected_candidates,
        "active_params_history": active_params_history,
        "last_active_params": active_params_history[-1] if active_params_history else {},
        "history": history,
        "obstacles": obstacles,
        "inner_bar": inner_bar,
        "outer_bar": outer_bar,
        "px": px,
        "py": py,
    }
    metrics["score"] = score_episode(metrics)
    return metrics
