import time
import numpy as np
from qpsolvers import solve_qp
import cbf 
import clf


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
        G = np.vstack([G_obs, G_barrier, G_box])
        h = np.concatenate([h_obs, h_barrier, h_box])

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

def _build_cbf_clf_constraints(
    robot_state,
    obstacles,
    px, py, pyaw, s,
    last_path_idx,
    ellipse_ab,
    margin,
    lookahead_l,
    alpha,
    eps_clf,
    q_clf,
    v_ref,
    v_bounds,
    w_bounds,
    barrier_inner,
    barrier_outer,
    seg_hint=None,
):
    """
    Restrições G z <= h em z = [v, w, delta]:
      CBFs (obstáculos + barreiras), CLF (Vdot + eps_clf V <= delta),
      bounds em v,w e delta >= 0
    """
    x, y, th = robot_state

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
    inner = barrier_inner if barrier_inner is not None else np.loadtxt("barreira_suavizada_interna.txt")
    outer = barrier_outer if barrier_outer is not None else np.loadtxt("barreira_suavizada_externa.txt")

    G_bar_2, h_bar = cbf.cbf_rows_for_barriers(
        x, y, th,
        barrier_inner=inner,
        barrier_outer=outer,
        ellipse_ab=ellipse_ab,
        margin=margin,
        lookahead_l=0.1,
        alpha=alpha,
        max_segments=10,
        seg_hint=seg_hint,
    )

    if G_bar_2.size == 0:
        G_bar = np.zeros((0, 3))
        h_bar = np.zeros((0,))
    else:
        G_bar = np.hstack([G_bar_2, np.zeros((G_bar_2.shape[0], 1))])

    # ----------------------------------
    # CLF
    # ----------------------------------
    G_clf, h_clf, clf_info = clf.clf_row_path_tracking(
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
    wmin, wmax = w_bounds

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
        -wmin,
        0.0
    ], dtype=float)

    # ----------------------------------
    # juntar tudo
    # ----------------------------------
    G = np.vstack([G_obs, G_bar, G_clf, G_box])
    h = np.concatenate([h_obs, h_bar, h_clf, h_box])
    #G = np.vstack([G_obs, G_clf, G_box])
    #h = np.concatenate([h_obs, h_clf, h_box])

    return G, h, clf_info


def _solve(P, q, G, h, solver_preference, info):
    z = None
    t0 = time.perf_counter()
    for sname in solver_preference:
        try:
            z = solve_qp(P, q, G, h, solver=sname)
            if z is not None:
                info["solver"] = sname
                break
        except Exception:
            pass
    info["qp_time"] = time.perf_counter() - t0  # segundos

    if z is None or np.any(np.isnan(z)):
        info["infeasible"] = True
        return None

    info["infeasible"] = False
    return z


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
    w_bounds=(-2.5, 2.5),
    solver_preference=("quadprog", "daqp"),
    barrier_inner=None,
    barrier_outer=None,
    seg_hint=None,           # dict: janela de segmentos das barreiras
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

    # ----------------------------------
    # custo em z = [v, w, delta]
    # ----------------------------------
    Wv, Ww = W
    P = 2.0 * np.diag([Wv, Ww, p_slack])
    q = -2.0 * np.array([Wv * v_nom, Ww * w_nom, 0.0], dtype=float)

    G, h, clf_info = _build_cbf_clf_constraints(
        robot_state, obstacles, px, py, pyaw, s, last_path_idx,
        ellipse_ab, margin, lookahead_l, alpha, eps_clf, q_clf,
        v_ref, v_bounds, w_bounds, barrier_inner, barrier_outer,
        seg_hint=seg_hint,
    )

    z = _solve(P, q, G, h, solver_preference, clf_info)
    if z is None:
        clf_info["delta"] = np.nan
        return np.array([v_bounds[0], 0.0], dtype=float), clf_info

    clf_info["delta"] = z[2]
    return z[:2], clf_info


def cbf_clf_qp_paper_cost(
    u_pre,
    robot_state,
    obstacles,
    px, py, pyaw, s,
    dt,
    last_path_idx=0,
    ellipse_ab=(0.30, 0.20),
    margin=0.05,
    lookahead_l=0.35,
    alpha=2.0,               # CBF
    eps_clf=1.0,             # CLF path tracking
    q_clf=(1.0, 4.0, 2.0),   # qx, qy, qpsi
    H=(1.0, 1.0),            # paper: 1/2 u^T H u
    Q=(10.0, 10.0),          # paper: (u-u_pre)^T Q (u-u_pre)
    p_slack=1000.0,          # peso da slack delta (CLF path)
    gamma_v=5.0,             # CLF velocidade
    p_v=1000.0,              # peso da slack delta_v (CLF velocidade)
    v_ref=1.5,
    v_bounds=(0.0, 1.5),
    w_bounds=(-2.5, 2.5),
    solver_preference=("quadprog", "daqp"),
    barrier_inner=None,
    barrier_outer=None,
    seg_hint=None,           # dict: janela de segmentos das barreiras
):
    """
    Custo do paper (Huang et al. 2023, eq. 24a) + CLF de velocidade:
      min 1/2 u^T H u + p_slack delta^2 + p_v delta_v^2 + (u-u_pre)^T Q (u-u_pre)

    sujeito a:
      mesmas restrições de cbf_clf_qp_filter (CBFs, CLF path, bounds, delta >= 0)
      CLF velocidade: V_v = 1/2 (v - v_ref)^2
        Vdot_v ~ (v_pre - v_ref)(v - v_pre)/dt   (v é entrada -> diferenças finitas)
        Vdot_v + gamma_v V_v(v_pre) <= delta_v,  delta_v >= 0
    z = [v, w, delta, delta_v]
    """
    u_pre = np.asarray(u_pre, dtype=float)
    v_pre = u_pre[0]

    # ----------------------------------
    # custo em z = [v, w, delta, delta_v]
    # ----------------------------------
    Hm = np.diag(H)
    Qm = np.diag(Q)
    P = np.zeros((4, 4))
    P[:2, :2] = Hm + 2.0 * Qm
    P[2, 2] = 2.0 * p_slack
    P[3, 3] = 2.0 * p_v
    q = np.zeros(4)
    q[:2] = -2.0 * Qm @ u_pre

    G3, h3, clf_info = _build_cbf_clf_constraints(
        robot_state, obstacles, px, py, pyaw, s, last_path_idx,
        ellipse_ab, margin, lookahead_l, alpha, eps_clf, q_clf,
        v_ref, v_bounds, w_bounds, barrier_inner, barrier_outer,
        seg_hint=seg_hint,
    )

    # ----------------------------------
    # CLF velocidade
    # ----------------------------------
    ev = v_pre - v_ref
    V_v = 0.5 * ev**2
    G_v = np.array([
        [ev / dt, 0.0, 0.0, -1.0],   # (v_pre-v_ref)(v-v_pre)/dt + gamma_v V_v <= delta_v
        [0.0,     0.0, 0.0, -1.0],   # delta_v >= 0
    ], dtype=float)
    h_v = np.array([ev * v_pre / dt - gamma_v * V_v, 0.0], dtype=float)

    G = np.vstack([np.hstack([G3, np.zeros((G3.shape[0], 1))]), G_v])
    h = np.concatenate([h3, h_v])

    clf_info["V_v"] = V_v
    z = _solve(P, q, G, h, solver_preference, clf_info)
    if z is None:
        clf_info["delta"] = np.nan
        clf_info["delta_v"] = np.nan
        return np.array([v_bounds[0], 0.0], dtype=float), clf_info

    clf_info["delta"] = z[2]
    clf_info["delta_v"] = z[3]
    return z[:2], clf_info
