import numpy as np
from qpsolvers import solve_qp
import cbf 


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

    inner = np.loadtxt("barreira_suavizada_interna.txt")
    outer = np.loadtxt("barreira_suavizada_externa.txt")

    '''

    G_barrier, h_barrier = cbf.cbf_rows_for_barriers(
        x, y, th,
        barrier_inner=inner,
        barrier_outer=outer,
        ellipse_ab=ellipse_ab,
        margin=margin,
        lookahead_l=0.1,
        alpha=alpha,
        max_segments=10
    )

    '''

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
    if G_obs.size == 0:
        G = G_box
        h = h_box
    else:
        #G = np.vstack([G_obs, G_barrier, G_box])
        #h = np.concatenate([h_obs, h_barrier, h_box])
        G = np.vstack([G_obs, G_box])
        h = np.concatenate([h_obs, h_box])

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
