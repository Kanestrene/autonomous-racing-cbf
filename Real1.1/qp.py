import numpy as np
from pathlib import Path
try:
    from qpsolvers import solve_qp
except ImportError:
    solve_qp = None
import cbf 


def _fallback_linear_solution(
    nu_nom, w_nom, G, h, nu_low, nu_high, wmin, wmax, Wnu=1.0, Ww=1.0
):
    """Procura uma solucao viavel simples para G[nu,w] <= h."""
    if nu_low > nu_high:
        return None

    nu_nom_clip = np.clip(nu_nom, nu_low, nu_high)
    nu_samples = np.unique(np.concatenate([
        np.array([nu_nom_clip, nu_low, nu_high], dtype=float),
        np.linspace(nu_low, nu_high, 101),
    ]))

    best_u = None
    best_cost = np.inf

    for nu in nu_samples:
        w_low = wmin
        w_high = wmax
        feasible = True

        for (a_nu, a_w), hi in zip(G, h):
            rhs = hi - a_nu * nu
            if abs(a_w) < 1e-12:
                if a_nu * nu > hi + 1e-9:
                    feasible = False
                    break
            elif a_w > 0.0:
                w_high = min(w_high, rhs / a_w)
            else:
                w_low = max(w_low, rhs / a_w)

            if w_low > w_high + 1e-9:
                feasible = False
                break

        if not feasible:
            continue

        w = np.clip(w_nom, w_low, w_high)
        cost = Wnu * (nu - nu_nom) ** 2 + Ww * (w - w_nom) ** 2
        if cost < best_cost:
            best_cost = cost
            best_u = np.array([nu, w], dtype=float)

    return best_u


def cbf_qp_filter(u_nom, robot_state, obstacles,
                  ellipse_ab=(0.30, 0.20),
                  margin=0.05, lookahead_l=None, alpha=2.0,
                  barrier_margin=None, barrier_alpha=None,
                  alpha1=None, alpha2=None, dt=0.02,
                  W=(20.0, 1.0),
                  nu_bounds=(-2.0, 2.0),
                  v_bounds=(0.0, 1.5), w_bounds=(-2.5, 2.5),
                  wheelbase=None,
                  delta_bounds=None,
                  delta_current=None,
                  delta_rate_max=None,
                  solver_preference=("quadprog", "daqp", "osqp", "clarabel", "scs")):
    """
    Resolve:
      min (u-u_nom)^T W (u-u_nom)
      s.t. G u <= h   (CBF + bounds)
    u = [nu, w]

    A velocidade e atualizada fora do QP por:
      v_next = v_current + nu*dt

    Se wheelbase/delta_bounds forem dados, tambem impoe:
      w = v_next/L * tan(delta)
    com delta dentro dos limites fisicos e de rate limit.
    """
    nu_nom, w_nom = u_nom
    if barrier_margin is None:
        barrier_margin = margin
    if barrier_alpha is None:
        barrier_alpha = alpha

    if len(robot_state) >= 4:
        x, y, th, v_current = robot_state[:4]
    else:
        x, y, th = robot_state
        v_current = 0.0

    # custo: (u-u_nom)^T W (u-u_nom)  -> 1/2 u^T P u + q^T u
    Wnu, Ww = W
    P = 2.0 * np.diag([Wnu, Ww])
    q = -2.0 * np.array([Wnu * nu_nom, Ww * w_nom], dtype=float)

    # CBF constraints
    G_obs, h_obs = cbf.cbf_rows_for_circle_obstacles(
        x, y, th, obstacles,
        ellipse_ab=ellipse_ab, margin=margin,
        lookahead_l=lookahead_l, alpha=alpha,
        alpha1=alpha1, alpha2=alpha2,
        v_current=v_current, dt=dt
    )

    base_dir = Path(__file__).resolve().parent
    inner = np.loadtxt(base_dir / "barreira_suavizada_interna.txt")
    outer = np.loadtxt(base_dir / "barreira_suavizada_externa.txt")

    G_barrier, h_barrier = cbf.cbf_rows_for_barriers(
        x, y, th,
        barrier_inner=inner,
        barrier_outer=outer,
        ellipse_ab=ellipse_ab,
        margin=barrier_margin,
        lookahead_l=0.05,
        alpha=3,
        alpha1=10.0, alpha2=10.0,
        v_current=v_current, dt=dt,
        max_segments=10
    )

    # bounds (box) em G u <= h, u=[nu,w]
    dt = max(float(dt), 1e-9)
    numin, numax = nu_bounds
    vmin, vmax = v_bounds
    wmin, wmax = w_bounds
    nu_low = max(numin, (vmin - v_current) / dt)
    nu_high = min(numax, (vmax - v_current) / dt)
    G_box = np.array([
        [ 1.0,  0.0],   #  nu <= numax
        [-1.0,  0.0],   # -nu <= -numin  -> nu >= numin
        [ dt,   0.0],   #  v_current + nu*dt <= vmax
        [-dt,   0.0],   # -v_current - nu*dt <= -vmin
        [ 0.0,  1.0],   #  w <= wmax
        [ 0.0, -1.0],   # -w <= -wmin  -> w >= wmin
    ])
    h_box = np.array([
        numax,
        -numin,
        vmax - v_current,
        v_current - vmin,
        wmax,
        -wmin,
    ], dtype=float)

    G_steer = np.zeros((0, 2), dtype=float)
    h_steer = np.zeros((0,), dtype=float)
    if wheelbase is not None and delta_bounds is not None:
        L = max(float(wheelbase), 1e-9)
        delta_min, delta_max = delta_bounds

        if delta_current is not None and delta_rate_max is not None:
            delta_step = abs(float(delta_rate_max)) * dt
            delta_min = max(delta_min, delta_current - delta_step)
            delta_max = min(delta_max, delta_current + delta_step)

        k_low = np.tan(delta_min) / L
        k_high = np.tan(delta_max) / L

        # w <= k_high*(v_current + nu*dt)
        # w >= k_low *(v_current + nu*dt)
        G_steer = np.array([
            [-k_high * dt,  1.0],
            [ k_low  * dt, -1.0],
        ], dtype=float)
        h_steer = np.array([
            k_high * v_current,
            -k_low * v_current,
        ], dtype=float)

    # juntar tudo
    G_parts = []
    h_parts = []
    for Gi, hi in (
        (G_obs, h_obs),
        (G_barrier, h_barrier),
        (G_box, h_box),
        (G_steer, h_steer),
    ):
        if Gi.size > 0:
            G_parts.append(Gi)
            h_parts.append(hi)

    G = np.vstack(G_parts)
    h = np.concatenate(h_parts)

    # resolver QP
    u = None
    if solve_qp is not None:
        for s in solver_preference:
            try:
                u = solve_qp(P, q, G, h, solver=s)
                if u is not None:
                    break
            except Exception:
                pass

    u_fallback = _fallback_linear_solution(
        nu_nom, w_nom, G, h, nu_low, nu_high, wmin, wmax, Wnu=Wnu, Ww=Ww
    )

    # fallback se falhar, ou se o solver devolver algo pior/fora das restricoes
    if u is None or np.any(np.isnan(u)):
        if u_fallback is not None:
            return u_fallback

        if nu_low <= nu_high:
            return np.array([nu_low, 0.0], dtype=float)
        return np.array([np.clip(0.0, numin, numax), 0.0], dtype=float)

    solver_violation = np.max(G @ u - h)
    if solver_violation > 1e-6 and u_fallback is not None:
        return u_fallback

    if u_fallback is not None:
        solver_cost = Wnu * (u[0] - nu_nom) ** 2 + Ww * (u[1] - w_nom) ** 2
        fallback_cost = (
            Wnu * (u_fallback[0] - nu_nom) ** 2
            + Ww * (u_fallback[1] - w_nom) ** 2
        )
        if fallback_cost + 1e-6 < solver_cost:
            return u_fallback

    return u
