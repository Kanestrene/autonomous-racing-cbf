import numpy as np
from pathlib import Path
from qpsolvers import solve_qp
import cbf2 as cbf


def cbf_qp_filter(u_nom, robot_state, obstacles,
                  ellipse_ab=(0.30, 0.20),
                  margin=0.05, lookahead_l=0.35, alpha=2.0,
                  W=(20.0, 1.0),
                  v_bounds=(0.0, 1.5), w_bounds=(-2.5, 2.5),
                  solver_preference=("quadprog", "daqp"),
                  use_barriers=True):
    """
    Dynamic-obstacle CBF-QP.

    u = [v, w]. Obstacles can include vx/vy; missing velocities are static.
    """
    v_nom, w_nom = u_nom
    x, y, th = robot_state

    Wv, Ww = W
    P = 2.0 * np.diag([Wv, Ww])
    q = -2.0 * np.array([Wv * v_nom, Ww * w_nom], dtype=float)

    G_obs, h_obs = cbf.cbf_rows_for_circle_obstacles(
        x, y, th, obstacles,
        ellipse_ab=ellipse_ab,
        margin=margin,
        lookahead_l=lookahead_l,
        alpha=alpha,
    )

    if use_barriers:
        base_dir = Path(__file__).resolve().parent
        inner = np.loadtxt(base_dir / "barreira_suavizada_interna.txt")
        outer = np.loadtxt(base_dir / "barreira_suavizada_externa.txt")

        G_barrier, h_barrier = cbf.cbf_rows_for_barriers(
            x, y, th,
            barrier_inner=inner,
            barrier_outer=outer,
            ellipse_ab=ellipse_ab,
            margin=margin,
            lookahead_l=0.01,
            alpha=4,
            max_segments=10,
        )
    else:
        G_barrier = np.zeros((0, 2), dtype=float)
        h_barrier = np.zeros((0,), dtype=float)

    vmin, vmax = v_bounds
    wmin, wmax = w_bounds
    G_box = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ])
    h_box = np.array([vmax, -vmin, wmax, -wmin], dtype=float)

    G_parts = []
    h_parts = []
    for G_i, h_i in ((G_obs, h_obs), (G_barrier, h_barrier), (G_box, h_box)):
        if G_i.size == 0:
            continue
        G_parts.append(G_i)
        h_parts.append(h_i)

    G = np.vstack(G_parts)
    h = np.concatenate(h_parts)

    u = None
    for s in solver_preference:
        try:
            u = solve_qp(P, q, G, h, solver=s)
            if u is not None:
                break
        except Exception:
            pass

    if u is None or np.any(np.isnan(u)):
        return np.array([vmin, 0.0], dtype=float)

    return u
