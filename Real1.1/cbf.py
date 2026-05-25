import numpy as np


def ellipse_radius_in_direction(a, b, ux, uy):
    # r(u) = 1 / sqrt((ux/a)^2 + (uy/b)^2)
    denom = (ux / max(1e-9, a)) ** 2 + (uy / max(1e-9, b)) ** 2
    return 1.0 / np.sqrt(max(1e-12, denom))


def _resolve_alphas(alpha, alpha1, alpha2):
    if alpha1 is None:
        alpha1 = alpha
    if alpha2 is None:
        alpha2 = alpha
    return float(alpha1), float(alpha2)


def _ihocbf_row_for_point(dx, dy, th, v_current, d_safe, alpha1, alpha2):
    """
    iHOCBF sem lookahead para u=[nu,w].

    Modelo usado no QP:
        x_dot = v cos(th)
        y_dot = v sin(th)
        th_dot = w

    b = dx^2 + dy^2 - d_safe^2
    b_dot = 2 v_current A
    b_ddot = 2 A nu + 2 v_current^2 + 2 v_current B w
    """
    v_current = float(v_current)

    c = np.cos(th)
    s = np.sin(th)
    A = dx * c + dy * s
    B = -dx * s + dy * c
    b_val = dx * dx + dy * dy - d_safe * d_safe

    coeff_nu = 2.0 * A
    coeff_w = 2.0 * v_current * B
    known_terms = (
        2.0 * v_current * v_current
        + 2.0 * (alpha1 + alpha2) * v_current * A
        + alpha1 * alpha2 * b_val
    )

    # coeff_nu*nu + coeff_w*w + known_terms >= 0
    # => -coeff_nu*nu - coeff_w*w <= known_terms
    G_row = [-coeff_nu, -coeff_w]
    h_row = known_terms
    return G_row, h_row


def cbf_rows_for_circle_obstacles(
    x,
    y,
    th,
    obstacles,
    ellipse_ab=(0.30, 0.20),
    margin=0.05,
    lookahead_l=None,
    alpha=2.0,
    v_current=0.0,
    dt=0.02,
    alpha1=None,
    alpha2=None,
):
    """
    Retorna G, h para G u <= h, u=[nu,w].
    Cada obstaculo gera 1 restricao iHOCBF.

    O parametro lookahead_l fica apenas por compatibilidade; nao e usado.
    """
    del lookahead_l
    alpha1, alpha2 = _resolve_alphas(alpha, alpha1, alpha2)
    a, b = ellipse_ab

    G_list = []
    h_list = []

    for obs in obstacles:
        ox, oy, ro = obs["x"], obs["y"], obs["r"]

        dx = x - ox
        dy = y - oy
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            ux, uy = np.cos(th), np.sin(th)
        else:
            ux, uy = dx / dist, dy / dist

        r_robot = ellipse_radius_in_direction(a, b, ux, uy)
        d_safe = ro + r_robot + margin
        G_row, h_row = _ihocbf_row_for_point(
            dx, dy, th, v_current, d_safe, alpha1, alpha2
        )

        G_list.append(G_row)
        h_list.append(h_row)

    return np.array(G_list, dtype=float), np.array(h_list, dtype=float)


def _closest_point_on_segment(px, py, ax, ay, bx, by):
    """Retorna (qx,qy,t) ponto mais proximo no segmento AB do ponto P."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom < 1e-12:
        return ax, ay, 0.0
    t = (apx * abx + apy * aby) / denom
    t = np.clip(t, 0.0, 1.0)
    qx = ax + t * abx
    qy = ay + t * aby
    return qx, qy, t


def cbf_rows_for_barriers(
    x,
    y,
    th,
    barrier_inner,
    barrier_outer,
    ellipse_ab=(0.30, 0.20),
    margin=0.05,
    lookahead_l=None,
    alpha=3.0,
    max_segments=40,
    v_current=0.0,
    dt=0.02,
    alpha1=None,
    alpha2=None,
):
    """
    iHOCBF para 2 barreiras (interna e externa), dadas como arrays Nx2.
    Retorna G, h para G u <= h, u=[nu,w].

    Sem ponto lookahead: a barreira e calculada a partir do centro do robo.
    Para cada barreira escolhe os segmentos mais proximos e cria uma restricao
    por segmento escolhido.
    """
    del lookahead_l
    alpha1, alpha2 = _resolve_alphas(alpha, alpha1, alpha2)
    a, b = ellipse_ab

    def add_constraints_from_poly(poly):
        G_list = []
        h_list = []

        poly = np.asarray(poly, dtype=float)
        if poly.shape[0] < 2:
            return G_list, h_list

        if np.hypot(poly[0, 0] - poly[-1, 0], poly[0, 1] - poly[-1, 1]) > 1e-9:
            poly2 = np.vstack([poly, poly[0]])
        else:
            poly2 = poly

        seg_a = poly2[:-1]
        seg_b = poly2[1:]

        d2 = np.empty(len(seg_a), dtype=float)
        q_cache = np.empty((len(seg_a), 2), dtype=float)

        for i, ((ax, ay), (bx, by)) in enumerate(zip(seg_a, seg_b)):
            qx, qy, _ = _closest_point_on_segment(x, y, ax, ay, bx, by)
            q_cache[i, 0] = qx
            q_cache[i, 1] = qy
            dx = x - qx
            dy = y - qy
            d2[i] = dx * dx + dy * dy

        m = min(max(1, int(max_segments)), len(seg_a))
        idxs = np.argpartition(d2, m - 1)[:m]

        for i in idxs:
            qx, qy = q_cache[i, 0], q_cache[i, 1]
            dx = x - qx
            dy = y - qy
            dist = np.hypot(dx, dy)

            if dist < 1e-9:
                ux, uy = np.cos(th), np.sin(th)
            else:
                ux, uy = dx / dist, dy / dist

            r_robot = ellipse_radius_in_direction(a, b, ux, uy)
            d_safe = r_robot + margin
            G_row, h_row = _ihocbf_row_for_point(
                dx, dy, th, v_current, d_safe, alpha1, alpha2
            )

            G_list.append(G_row)
            h_list.append(h_row)

        return G_list, h_list

    G_list_all, h_list_all = [], []

    for poly in (barrier_inner, barrier_outer):
        Gi, hi = add_constraints_from_poly(poly)
        G_list_all.extend(Gi)
        h_list_all.extend(hi)

    if len(G_list_all) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    return np.array(G_list_all, dtype=float), np.array(h_list_all, dtype=float)
