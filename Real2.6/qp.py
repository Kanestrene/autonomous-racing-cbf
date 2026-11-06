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

def _avoidance_bias(x, y, th, obstacles, clf_info, ellipse_ab, margin,
                    activation_dist, hold_extra, kappa_max, v_ref):
    """
    Desempate para obstaculos de frente.

    Com o obstaculo alinhado com o carro a CBF so consegue ser satisfeita a travar
    (a_w ~ 0), e a CLF puxa o carro para o caminho, que passa pelo obstaculo:
    o carro trava ate parar (deadlock). Aqui escolhe-se um lado e devolve-se um
    w nominal de desvio; a CBF continua a garantir a seguranca.

    - obstaculo no corredor a frente (|lat| < r_safe): roda para o lado livre
    - ja fora do corredor mas ainda a frente (|lat| < r_safe + hold_extra): segue
      direito (w=0), para a CLF nao o puxar de volta antes de passar o obstaculo
    Devolve (w_bias, ativo).
    """
    a, b = ellipse_ab
    r_robot = max(a, b)
    c, s_ = np.cos(th), np.sin(th)

    best = None
    for obs in obstacles:
        dx = obs["x"] - x
        dy = obs["y"] - y
        fwd = c * dx + s_ * dy          # distancia a frente do carro
        lat = -s_ * dx + c * dy         # >0: obstaculo a esquerda
        r_safe = obs["r"] + r_robot + margin
        if 0.0 < fwd < r_safe + activation_dist and abs(lat) < r_safe + hold_extra:
            if best is None or fwd < best[0]:
                best = (fwd, lat, obs, r_safe)

    if best is None:
        return 0.0, False

    _, lat, obs, r_safe = best
    if abs(lat) >= r_safe:
        return 0.0, True                # ja tem espaco: mantem a direcao ate passar
    if abs(lat) > 0.01:
        side = -np.sign(lat)            # roda para o lado oposto ao obstaculo
    else:
        # obstaculo mesmo a frente: contorna pelo lado contrario ao desvio do
        # obstaculo em relacao ao caminho (fica mais perto do caminho)
        psi_r = clf_info["psi_r"]
        e_obs = -np.sin(psi_r) * (obs["x"] - clf_info["xr"]) + np.cos(psi_r) * (obs["y"] - clf_info["yr"])
        side = -np.sign(e_obs) if abs(e_obs) > 1e-3 else 1.0

    return float(side * kappa_max * v_ref), True


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
    obstacle_ellipse_ab=None,   # tamanho do robo so para os obstaculos (None = ellipse_ab)
    obstacle_margin=None,       # margem so para os obstaculos (None = margin)
    body_cbf=True,              # CBF extra no centro do robo (protege o corpo, nao so o ponto a frente)
    avoid_bias=True,            # desempate para obstaculos de frente
    avoid_activation_dist=0.25, # distancia (m) a frente a partir da qual o desvio e ativado
    avoid_hold_extra=0.08,      # folga lateral (m) extra em que o carro mantem a direcao
    avoid_p_slack=0.01,         # peso da slack da CLF durante o desvio (deixa o carro sair do caminho)
    obs_lookahead_slack=1e4,    # peso da folga da CBF no ponto a frente (None = rigida)
    solver_preference=("quadprog", "daqp")
):
    """
    Resolve:
      min (u-u_nom)^T W (u-u_nom) + p_slack * delta^2 + obs_lookahead_slack * sigma^2

    sujeito a:
      CBFs: obstaculos no centro (rigida) e no ponto a frente (com folga sigma), barreiras
      CLF: Vdot + eps_clf V <= delta
      v_min <= v <= v_max
      |w| <= kappa_max * v   (o carro so roda se andar: viragem fisicamente possivel)
      delta >= 0

    Devolve (u, info); info inclui "qp_ok", "obs_clearance", "avoid" e "sigma".
    """
    v_nom, w_nom = u_nom
    x, y, th = robot_state

    obs_ab = ellipse_ab if obstacle_ellipse_ab is None else obstacle_ellipse_ab
    obs_margin = margin if obstacle_margin is None else obstacle_margin

    # ----------------------------------
    # CLF (primeiro: a escolha do lado do desvio usa o ponto de referencia)
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
    # desempate para obstaculos de frente
    # ----------------------------------
    avoiding = False
    if avoid_bias and obstacles:
        w_bias, avoiding = _avoidance_bias(
            x, y, th, obstacles, clf_info, obs_ab, obs_margin,
            avoid_activation_dist, avoid_hold_extra, kappa_max, v_ref,
        )
        if avoiding:
            w_nom = w_bias
            p_slack = min(p_slack, avoid_p_slack)

    # ----------------------------------
    # custo em z = [v, w, delta, sigma]
    #   delta: folga da CLF
    #   sigma: folga da CBF dos obstaculos no ponto a frente (so se obs_lookahead_slack)
    # ----------------------------------
    soft_lookahead = obs_lookahead_slack is not None
    p_sigma = obs_lookahead_slack if soft_lookahead else 1.0

    Wv, Ww = W
    P = 2.0 * np.diag([Wv, Ww, p_slack, p_sigma])
    q = -2.0 * np.array([Wv * v_nom, Ww * w_nom, 0.0, 0.0], dtype=float)

    def _rows(G2, h2, sigma_coef=0.0):
        """[a_v, a_w] -> linhas em z=[v, w, delta, sigma]."""
        if G2.size == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        n = G2.shape[0]
        return np.hstack([G2, np.zeros((n, 1)), np.full((n, 1), sigma_coef)]), h2

    # ----------------------------------
    # CBF obstáculos circulares
    # retorna G u <= h com u=[v,w]
    # ----------------------------------
    # no ponto a frente: antecipa o desvio. Com |w| <= kappa_max*v o ponto a frente pode
    # ficar sem saida (QP inviavel e o carro para), por isso pode ser flexivel (sigma).
    G_obs_2, h_obs = cbf.cbf_rows_for_circle_obstacles(
        x, y, th, obstacles,
        ellipse_ab=obs_ab,
        margin=obs_margin,
        lookahead_l=lookahead_l,
        alpha=alpha
    )
    G_obs, h_obs = _rows(G_obs_2, h_obs, sigma_coef=-1.0 if soft_lookahead else 0.0)

    # no centro do robo: rigida, protege o corpo
    G_body_2, h_body = cbf.cbf_rows_for_circle_obstacles(
        x, y, th, obstacles,
        ellipse_ab=obs_ab,
        margin=obs_margin,
        lookahead_l=0.0,
        alpha=alpha
    )
    if body_cbf:
        G_body, h_body = _rows(G_body_2, h_body)
    else:
        G_body, h_body = _rows(np.zeros((0, 2)), np.zeros((0,)))

    # folga (m) entre o robo e o obstaculo mais proximo; < 0 = dentro da zona de seguranca
    clf_info["obs_clearance"] = min(
        (float(np.hypot(x - o["x"], y - o["y"]) - (o["r"] + max(obs_ab) + obs_margin)) for o in obstacles),
        default=float("inf"),
    )
    clf_info["avoid"] = avoiding

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
    G_bar, h_bar = _rows(G_bar_2, h_bar)

    G_clf = np.hstack([G_clf, np.zeros((G_clf.shape[0], 1))])

    # ----------------------------------
    # bounds
    # ----------------------------------
    vmin, vmax = v_bounds

    G_box = np.array([
        [ 1.0,        0.0,  0.0,  0.0],   # v <= vmax
        [-1.0,        0.0,  0.0,  0.0],   # v >= vmin
        [-kappa_max,  1.0,  0.0,  0.0],   # w <=  kappa_max * v
        [-kappa_max, -1.0,  0.0,  0.0],   # w >= -kappa_max * v
        [ 0.0,        0.0, -1.0,  0.0],   # delta >= 0
        [ 0.0,        0.0,  0.0, -1.0],   # sigma >= 0
    ], dtype=float)

    h_box = np.array([
        vmax,
        -vmin,
        0.0,
        0.0,
        0.0,
        0.0,
    ], dtype=float)

    # ----------------------------------
    # juntar tudo
    # ----------------------------------
    G = np.vstack([G_obs, G_body, G_bar, G_clf, G_box])
    h = np.concatenate([h_obs, h_body, h_bar, h_clf, h_box])

    z = None
    for sname in solver_preference:
        try:
            z = solve_qp(P, q, G, h, solver=sname)
            if z is not None:
                break
        except Exception:
            pass

    if z is None or np.any(np.isnan(z)):
        # QP inviavel: para (antes era silencioso; agora fica assinalado no info)
        clf_info["qp_ok"] = False
        clf_info["sigma"] = float("nan")
        return np.array([vmin, 0.0], dtype=float), clf_info

    clf_info["qp_ok"] = True
    clf_info["sigma"] = float(z[3])
    return z[:2], clf_info
