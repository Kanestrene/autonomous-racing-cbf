import numpy as np
from qpsolvers import solve_qp
import cbf 
from shared_config import (
    BARRIER_INNER_M,
    BARRIER_OUTER_M,
    BARRIER_ROUND_CORNER_RADIUS_M,
    BARRIER_ROUND_CORNER_SAMPLES,
)


_BARRIER_CACHE = None


def build_rounded_polyline(points, corner_radius=0.08, corner_samples=10, closed=True):
    n = len(points)
    if n < 2:
        return list(points)
    if n == 2:
        return list(points)

    pts = [np.array(p, dtype=float) for p in points]

    if not closed:
        out_open = [tuple(pts[0])]
        for i in range(1, n - 1):
            p_prev = pts[i - 1]
            p_curr = pts[i]
            p_next = pts[i + 1]
            v_in = p_curr - p_prev
            v_out = p_next - p_curr
            len_in = float(np.linalg.norm(v_in))
            len_out = float(np.linalg.norm(v_out))
            if len_in < 1e-9 or len_out < 1e-9:
                out_open.append(tuple(p_curr))
                continue
            u_in = v_in / len_in
            u_out = v_out / len_out
            d = min(corner_radius, 0.45 * len_in, 0.45 * len_out)
            p_start = p_curr - u_in * d
            p_end = p_curr + u_out * d
            out_open.append(tuple(p_start))
            for t in np.linspace(0.0, 1.0, corner_samples + 2)[1:-1]:
                pt = ((1.0 - t) ** 2) * p_start + 2.0 * (1.0 - t) * t * p_curr + (t ** 2) * p_end
                out_open.append((float(pt[0]), float(pt[1])))
            out_open.append(tuple(p_end))
        out_open.append(tuple(pts[-1]))
        return out_open

    out = []
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]
        v_in = p_curr - p_prev
        v_out = p_next - p_curr
        len_in = float(np.linalg.norm(v_in))
        len_out = float(np.linalg.norm(v_out))
        if len_in < 1e-9 or len_out < 1e-9:
            continue
        u_in = v_in / len_in
        u_out = v_out / len_out
        d = min(corner_radius, 0.45 * len_in, 0.45 * len_out)
        p_start = p_curr - u_in * d
        p_end = p_curr + u_out * d
        out.append((float(p_start[0]), float(p_start[1])))
        for t in np.linspace(0.0, 1.0, corner_samples + 2)[1:-1]:
            pt = ((1.0 - t) ** 2) * p_start + 2.0 * (1.0 - t) * t * p_curr + (t ** 2) * p_end
            out.append((float(pt[0]), float(pt[1])))
        out.append((float(p_end[0]), float(p_end[1])))

    return out


def get_dense_barriers():
    global _BARRIER_CACHE

    if _BARRIER_CACHE is None:
        inner = build_rounded_polyline(
            BARRIER_INNER_M,
            corner_radius=BARRIER_ROUND_CORNER_RADIUS_M,
            corner_samples=BARRIER_ROUND_CORNER_SAMPLES,
            closed=True,
        )
        outer = build_rounded_polyline(
            BARRIER_OUTER_M,
            corner_radius=BARRIER_ROUND_CORNER_RADIUS_M,
            corner_samples=BARRIER_ROUND_CORNER_SAMPLES,
            closed=True,
        )
        _BARRIER_CACHE = (
            np.asarray(inner, dtype=float),
            np.asarray(outer, dtype=float),
        )

    return _BARRIER_CACHE


def cbf_qp_filter(u_nom, robot_state, obstacles,
                  ellipse_ab=(0.30, 0.20),
                  margin=0.05, lookahead_l=0.35, alpha=2.0,
                  W=(20.0, 1.0),
                  v_bounds=(0.0, 1.5), w_bounds=(-2.5, 2.5),
                  solver_preference=("quadprog", "daqp")):
    """
    Resolve:
      min (u-u_nom)^T W (u-u_nom)
      s.t. G u <= h   (CBF + bounds)
    u = [v, w]
    """
    v_nom, w_nom = u_nom
    x, y, th = robot_state

    # custo: (u-u_nom)^T W (u-u_nom)  -> 1/2 u^T P u + q^T u
    Wv, Ww = W
    P = 2.0 * np.diag([Wv, Ww])
    q = -2.0 * np.array([Wv * v_nom, Ww * w_nom], dtype=float)

    # CBF constraints
    G_obs, h_obs = cbf.cbf_rows_for_circle_obstacles(
        x, y, th, obstacles,
        ellipse_ab=ellipse_ab, margin=margin,
        lookahead_l=lookahead_l, alpha=alpha
    )

    inner, outer = get_dense_barriers()

    G_barrier, h_barrier = cbf.cbf_rows_for_barriers(
        x, y, th,
        barrier_inner=inner,
        barrier_outer=outer,
        ellipse_ab=ellipse_ab,
        margin=margin,
        lookahead_l=0.01,
        alpha=alpha,
        max_segments=10
    )

    # bounds (box) em G u <= h
    vmin, vmax = v_bounds
    wmin, wmax = w_bounds
    G_box = np.array([
        [ 1.0,  0.0],   #  v <= vmax
        [-1.0,  0.0],   # -v <= -vmin  -> v >= vmin
        [ 0.0,  1.0],   #  w <= wmax
        [ 0.0, -1.0],   # -w <= -wmin  -> w >= wmin
    ])
    h_box = np.array([vmax, -vmin, wmax, -wmin], dtype=float)

    # juntar tudo
    G_parts = [G_box]
    h_parts = [h_box]

    if G_obs.size != 0:
        G_parts.insert(0, G_obs)
        h_parts.insert(0, h_obs)

    if G_barrier.size != 0:
        G_parts.insert(-1, G_barrier)
        h_parts.insert(-1, h_barrier)

    G = np.vstack(G_parts)
    h = np.concatenate(h_parts)

    # resolver QP
    u = None
    for s in solver_preference:
        try:
            u = solve_qp(P, q, G, h, solver=s)
            if u is not None:
                break
        except Exception:
            pass

    # fallback se falhar
    if u is None or np.any(np.isnan(u)):
        return np.array([vmin, 0.0], dtype=float)

    return u
