import numpy as np
from qpsolvers import solve_qp
import cbf 
import clf2
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

    out = []
    for i in range(n):
        if not closed and i in (0, n - 1):
            out.append(tuple(pts[i]))
            continue

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
        lookahead_l=0.02,
        alpha=3,
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

def cbf_clf_qp_filter(
    u_nom,
    robot_state,
    obstacles,
    px, py, pyaw, s,
    last_path_idx=0,
    ellipse_ab=(0.30, 0.20),
    margin=0.05,
    lookahead_l=0.35,
    alpha=2.0,               # CBF
    eps_clf=1.0,             # CLF
    q_clf=(1.0, 4.0, 2.0),   # qx, qy, qpsi
    W=(20.0, 1.0),           # pesos para v,w
    p_slack=1000.0,          # peso da slack delta
    v_ref=1.5,
    v_bounds=(0.0, 1.5),
    kappa_max=1,
    solver_preference=("quadprog", "daqp")
):
    """
    Resolve:
      min (u-u_nom)^T W (u-u_nom) + p_slack * delta^2

    sujeito a:
      CBFs
      CLF: Vdot + eps_clf V <= delta
      bounds em v,w
      delta >= 0
    """
    v_nom, w_nom = u_nom
    x, y, th = robot_state

    # ----------------------------------
    # custo em z = [v, w, delta]
    # ----------------------------------
    Wv, Ww = W
    P = 2.0 * np.diag([Wv, Ww, p_slack])
    q = -2.0 * np.array([Wv * v_nom, Ww * w_nom, 0.0], dtype=float)

    # ----------------------------------
    # CBF obstáculos circulares
    # retorna G u <= h com u=[v,w]
    # ----------------------------------
    G_obs_2, h_obs = cbf.cbf_rows_for_circle_obstacles(
        x, y, th, obstacles,
        ellipse_ab=ellipse_ab,
        margin=margin,
        lookahead_l=lookahead_l,
        alpha=alpha
    )

    if G_obs_2.size == 0:
        G_obs = np.zeros((0, 3))
        h_obs = np.zeros((0,))
    else:
        G_obs = np.hstack([G_obs_2, np.zeros((G_obs_2.shape[0], 1))])

    # ----------------------------------
    # CBF barreiras
    # ----------------------------------
    inner, outer = get_dense_barriers()

    G_bar_2, h_bar = cbf.cbf_rows_for_barriers(
        x, y, th,
        barrier_inner=inner,
        barrier_outer=outer,
        ellipse_ab=ellipse_ab,
        margin=margin,
        lookahead_l=0.1,
        alpha=alpha,
        max_segments=10
    )

    if G_bar_2.size == 0:
        G_bar = np.zeros((0, 3))
        h_bar = np.zeros((0,))
    else:
        G_bar = np.hstack([G_bar_2, np.zeros((G_bar_2.shape[0], 1))])

    # ----------------------------------
    # CLF
    # ----------------------------------
    G_clf, h_clf, clf_info = clf2.clf_row_path_tracking(
        px, py, pyaw, s,
        robot_state=(x, y, th),
        last_idx=last_path_idx,
        v_ref=v_ref,
        qx=q_clf[0],
        qy=q_clf[1],
        qpsi=q_clf[2],
        eps=eps_clf
    )

    # ----------------------------------
    # bounds
    # ----------------------------------
    vmin, vmax = v_bounds
    
    wmax = abs(vmax) * kappa_max

    G_box = np.array([
        [ 1.0,  0.0,  0.0],   # v <= vmax
        [-1.0,  0.0,  0.0],   # v >= vmin
        [ 0.0,  1.0,  0.0],   # w <= wmax
        [ 0.0, -1.0,  0.0],   # w >= wmin
        [ 0.0,  0.0, -1.0],   # delta >= 0
    ], dtype=float)

    h_box = np.array([
        vmax,
        -vmin,
        wmax,
        wmax,
        0.0
    ], dtype=float)

    # ----------------------------------
    # juntar tudo
    # ----------------------------------
    G = np.vstack([G_obs, G_bar, G_clf, G_box])
    h = np.concatenate([h_obs, h_bar, h_clf, h_box])
    #G = np.vstack([G_obs, G_clf, G_box])
    #h = np.concatenate([h_obs, h_clf, h_box])

    z = None
    for sname in solver_preference:
        try:
            z = solve_qp(P, q, G, h, solver=sname)
            if z is not None:
                break
        except Exception:
            pass

    if z is None or np.any(np.isnan(z)):
        return np.array([vmin, 0.0], dtype=float), clf_info

    return z[:2], clf_info
