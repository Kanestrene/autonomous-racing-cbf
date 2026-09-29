"""
CLF-CBF-QP de Huang et al., "Obstacle Avoidance for Unicycle-Modelled Mobile
Robots with Time-varying Control Barrier Functions", IECON 2023.

Estado (x, y, th): (x, y) = eixo traseiro. Centro do robô c = (x + l cos th, y + l sin th).

  min_{u,delta}  1/2 u^T H u + p delta^2 + (u - u_pre)^T Q (u - u_pre)      (24a)
  s.t.  LgV u + gamma V <= delta                                            (24b)
        Lgh_i u + dh_i/dt + alpha h_i >= 0                                  (24c)
        u in U,  delta in R                                                 (24d,e)
"""
import time
import numpy as np
from qpsolvers import solve_qp

from cbf import _nearest_segments
from clf import nearest_path_point_closed


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _cbf_row(dx, dy, th, l, r_safe, alpha, dh_dt=0.0):
    """
    h = dx^2 + dy^2 - r_safe^2 (eq. 16), com [dx,dy] = c - o.
    Lgh = [2(dx cos + dy sin), 2l(-dx sin + dy cos)] (eq. 18).
    Lgh u + dh_dt + alpha h >= 0  ->  -Lgh u <= alpha h + dh_dt
    """
    h_val = dx * dx + dy * dy - r_safe * r_safe
    a_v = 2.0 * (dx * np.cos(th) + dy * np.sin(th))
    a_w = 2.0 * l * (-dx * np.sin(th) + dy * np.cos(th))
    return [-a_v, -a_w, 0.0], alpha * h_val + dh_dt, h_val


def paper_goal_on_path(px, py, pyaw, s, robot_state, l, last_idx, d_goal):
    """Objetivo móvel: ponto do caminho d_goal metros (em arco) à frente do centro."""
    x, y, th = robot_state
    xc = x + l * np.cos(th)
    yc = y + l * np.sin(th)
    idx, _ = nearest_path_point_closed(px, py, xc, yc, last_idx=last_idx)

    ds = s[1] - s[0]
    n = len(px)
    g = (idx + int(round(d_goal / ds))) % n
    return (px[g], py[g], pyaw[g]), idx


def paper_clf_cbf_qp(
    u_pre,
    robot_state,
    goal,
    obstacles,
    barrier_inner,
    barrier_outer,
    l=0.15,
    r_robot=0.30,
    margin=0.05,
    alpha=1.5,
    gamma=0.5,
    p_slack=1000.0,
    H=(1.0, 1.0),
    Q=(10.0, 10.0),
    P_clf=((1.0, 0.0, 0.5),
           (0.0, 1.0, 0.5),
           (0.5, 0.5, 1.0)),
    v_bounds=(0.0, 2.0),
    w_bounds=(-2.5, 2.5),
    max_segments=10,
    seg_hint=None,           # dict: janela de segmentos das paredes
    seg_window=100,
    goal_frame=True,
    kappa_max=None,
    obs_vel=None,
    solver_preference=("quadprog", "daqp"),
):
    x, y, th = robot_state
    xg, yg, thg = goal
    xc = x + l * np.cos(th)
    yc = y + l * np.sin(th)

    # ----------------------------------
    # custo (24a) em z = [v, w, delta]
    # ----------------------------------
    Hm = np.diag(H)
    Qm = np.diag(Q)
    u_pre = np.asarray(u_pre, dtype=float)

    P = np.zeros((3, 3))
    P[:2, :2] = Hm + 2.0 * Qm
    P[2, 2] = 2.0 * p_slack
    q = np.zeros(3)
    q[:2] = -2.0 * Qm @ u_pre

    G_list, h_list, h_vals = [], [], []

    # ----------------------------------
    # CLF (11): V = e^T P e
    # ----------------------------------
    Pc = np.asarray(P_clf, dtype=float)
    ex, ey = x - xg, y - yg
    rot = thg if goal_frame else 0.0
    # goal_frame: erro (x,y) no referencial do objetivo (equivale a P rodado por thg),
    # para os termos cruzados não dependerem da orientação da pista
    c, s_ = np.cos(rot), np.sin(rot)
    e = np.array([c * ex + s_ * ey, -s_ * ex + c * ey, wrap_to_pi(th - thg)])
    V = float(e @ Pc @ e)
    g_mat = np.array([[np.cos(th - rot), 0.0],
                      [np.sin(th - rot), 0.0],
                      [0.0, 1.0]])
    LgV = 2.0 * e @ Pc @ g_mat
    G_list.append([LgV[0], LgV[1], -1.0])
    h_list.append(-gamma * V)

    # ----------------------------------
    # CBF obstáculos (16), time-varying se obs_vel
    # ----------------------------------
    for i, obs in enumerate(obstacles):
        dx = xc - obs["x"]
        dy = yc - obs["y"]
        dh_dt = 0.0
        if obs_vel is not None:
            vox, voy = obs_vel[i]
            dh_dt = -2.0 * (dx * vox + dy * voy)
        row, rhs, hv = _cbf_row(dx, dy, th, l, r_robot + obs["r"] + margin, alpha, dh_dt)
        G_list.append(row)
        h_list.append(rhs)
        h_vals.append(hv)

    # ----------------------------------
    # CBF paredes: mesma forma, o = ponto mais próximo do segmento
    # (mesma seleção dos segmentos mais próximos que cbf.cbf_rows_for_barriers)
    # ----------------------------------
    for key, poly in enumerate((barrier_inner, barrier_outer)):
        poly = np.asarray(poly, dtype=float)
        if np.hypot(poly[0, 0] - poly[-1, 0], poly[0, 1] - poly[-1, 1]) > 1e-9:
            poly = np.vstack([poly, poly[0]])
        A = poly[:-1]
        B = poly[1:]

        q_sel, _ = _nearest_segments(xc, yc, A, B, max_segments,
                                     seg_hint=seg_hint, key=key, seg_window=seg_window)
        for qx, qy in q_sel:
            dx = xc - qx
            dy = yc - qy
            row, rhs, hv = _cbf_row(dx, dy, th, l, r_robot + margin, alpha)
            G_list.append(row)
            h_list.append(rhs)
            h_vals.append(hv)

    # ----------------------------------
    # bounds (24d), delta livre (24e)
    # ----------------------------------
    vmin, vmax = v_bounds
    wmin, wmax = w_bounds
    G_list += [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    h_list += [vmax, -vmin, wmax, -wmin]
    if kappa_max is not None:
        # carro (Ackermann): |w| <= kappa_max * v, afim em u (não roda no sítio)
        G_list += [[-kappa_max, 1.0, 0.0], [-kappa_max, -1.0, 0.0]]
        h_list += [0.0, 0.0]

    G = np.array(G_list, dtype=float)
    h = np.array(h_list, dtype=float)

    info = {"V": V, "e": e, "h_min": min(h_vals) if h_vals else np.inf}

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
    info["qp_time"] = time.perf_counter() - t0

    if z is None or np.any(np.isnan(z)):
        info["infeasible"] = True
        info["delta"] = np.nan
        return np.array([vmin, 0.0], dtype=float), info

    info["infeasible"] = False
    info["delta"] = z[2]
    return z[:2], info
