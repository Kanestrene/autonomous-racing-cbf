import numpy as np
from matplotlib.patches import Circle, Ellipse
from scipy.interpolate import CubicSpline

def ellipse_radius_in_direction(a, b, ux, uy):
    # r(u) = 1 / sqrt((ux/a)^2 + (uy/b)^2)
    denom = (ux / max(1e-9, a))**2 + (uy / max(1e-9, b))**2
    return 1.0 / np.sqrt(max(1e-12, denom))

def cbf_rows_for_circle_obstacles(x, y, th, obstacles,
                                 ellipse_ab=(0.30, 0.20),
                                 margin=0.05, lookahead_l=0.35,
                                 alpha=2.0):
    """
    Retorna G, h para G u <= h, u=[v,w].
    Cada obstáculo gera 1 restrição CBF.

    obstáculos: lista de dicts {"x":..., "y":..., "r":...}
    """
    a, b = ellipse_ab
    l = lookahead_l

    # ponto à frente
    px = x + l*np.cos(th)
    py = y + l*np.sin(th)

    G_list = []
    h_list = []

    for obs in obstacles:
        ox, oy, ro = obs["x"], obs["y"], obs["r"]

        # direção centro do robô -> obstáculo para inflar pela elipse
        dx_c = x - ox
        dy_c = y - oy
        dist_c = np.hypot(dx_c, dy_c)
        if dist_c < 1e-6:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx_c/dist_c, dy_c/dist_c

        r_robot = ellipse_radius_in_direction(a, b, ux, uy)
        r_safe = ro + r_robot + margin

        # barreira no ponto à frente
        dx = px - ox
        dy = py - oy
        h_val = (dx*dx + dy*dy) - (r_safe*r_safe)

        # p_dot = [ v cos(th) - l w sin(th),
        #           v sin(th) + l w cos(th) ]
        # \dot h = 2 [dx,dy]·p_dot = a_v v + a_w w
        a_v = 2.0*(dx*np.cos(th) + dy*np.sin(th))
        a_w = 2.0*l*(-dx*np.sin(th) + dy*np.cos(th))

        # CBF: a_v v + a_w w + alpha*h >= 0
        # => a_v v + a_w w >= -alpha*h
        # => -(a_v v + a_w w) <= alpha*h   (formato G u <= h)
        G_list.append([-a_v, -a_w])
        h_list.append(alpha*h_val)

    return np.array(G_list, dtype=float), np.array(h_list, dtype=float)

def _closest_point_on_segment(px, py, ax, ay, bx, by):
    """Retorna (qx,qy,t) ponto mais próximo no segmento AB do ponto P."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx*abx + aby*aby
    if denom < 1e-12:
        # segmento degenerado
        return ax, ay, 0.0
    t = (apx*abx + apy*aby) / denom
    t = np.clip(t, 0.0, 1.0)
    qx = ax + t*abx
    qy = ay + t*aby
    return qx, qy, t

def _closest_points_on_segments(px, py, A, B):
    """
    Versão vetorizada de _closest_point_on_segment para todos os segmentos AB.
    Retorna (Q, d2): pontos mais próximos (N,2) e distâncias ao quadrado (N,).
    """
    abx = B[:, 0] - A[:, 0]
    aby = B[:, 1] - A[:, 1]
    apx = px - A[:, 0]
    apy = py - A[:, 1]
    denom = abx*abx + aby*aby
    degenerate = denom < 1e-12
    t = (apx*abx + apy*aby) / np.where(degenerate, 1.0, denom)
    t = np.where(degenerate, 0.0, np.clip(t, 0.0, 1.0))
    qx = A[:, 0] + t*abx
    qy = A[:, 1] + t*aby
    dx = px - qx
    dy = py - qy
    return np.stack([qx, qy], axis=1), dx*dx + dy*dy

def _nearest_segments(px, py, A, B, max_segments, seg_hint=None, key=None, seg_window=100):
    """
    Escolhe os 'max_segments' segmentos AB mais próximos do ponto p.
    Retorna (Q, d2) só dos segmentos escolhidos.

    Se seg_hint (dict) for dado, procura só numa janela de +-seg_window segmentos
    à volta do segmento mais próximo do passo anterior (seg_hint[key]) e atualiza-o.
    Procura completa: na 1a chamada, ou se o mínimo cair no bordo da janela
    (o robô saiu da janela -> re-ancorar).
    """
    n = len(A)
    use_window = seg_hint is not None and key in seg_hint and 2*seg_window + 1 < n
    if use_window:
        idx = (seg_hint[key] + np.arange(-seg_window, seg_window + 1)) % n
        Q, d2 = _closest_points_on_segments(px, py, A[idx], B[idx])
        j = int(np.argmin(d2))
        if j == 0 or j == len(idx) - 1:
            use_window = False
    if not use_window:
        idx = None
        Q, d2 = _closest_points_on_segments(px, py, A, B)
        j = int(np.argmin(d2))

    if seg_hint is not None:
        seg_hint[key] = j if idx is None else int(idx[j])

    m = min(max_segments, len(d2))
    sel = np.argpartition(d2, m-1)[:m]
    return Q[sel], d2[sel]

def cbf_rows_for_barriers(x, y, th,
                            barrier_inner, barrier_outer,
                            ellipse_ab=(0.30, 0.20),
                            margin=0.05,
                            lookahead_l=0.35,
                            alpha=2.0,
                            max_segments=40,
                            seg_hint=None,
                            seg_window=100):
    """
    CBF para 2 barreiras (interna e externa), dadas como arrays Nx2 (fechados ou não).
    Retorna G, h para G u <= h, u=[v,w].

    - Usa ponto lookahead p = [x + l cos(th), y + l sin(th)]
    - Para cada barreira, escolhe os 'max_segments' segmentos mais próximos e
        cria uma restrição por segmento escolhido.
    - seg_hint (dict, opcional): procura só numa janela de +-seg_window segmentos
        à volta do mais próximo do passo anterior (ver _nearest_segments).
    """

    a, b = ellipse_ab
    l = lookahead_l

    # ponto à frente
    px = x + l*np.cos(th)
    py = y + l*np.sin(th)

    # raio efetivo do robô na direção "p -> barreira" (aprox pelo vetor p-q)
    # (vamos calcular por restrição)

    def add_constraints_from_poly(poly, key):
        G_list = []
        h_list = []

        poly = np.asarray(poly, dtype=float)
        if poly.shape[0] < 2:
            return G_list, h_list

        # garante fechado para segmentos (se não estiver)
        if np.hypot(poly[0,0]-poly[-1,0], poly[0,1]-poly[-1,1]) > 1e-9:
            poly2 = np.vstack([poly, poly[0]])
        else:
            poly2 = poly

        A = poly2[:-1]
        B = poly2[1:]

        # segmentos mais próximos do ponto lookahead
        q_sel, _ = _nearest_segments(px, py, A, B, max_segments,
                                     seg_hint=seg_hint, key=key, seg_window=seg_window)

        for qx, qy in q_sel:
            dx = px - qx
            dy = py - qy
            dist = np.hypot(dx, dy)

            # direção para inflar pela elipse (robô)
            if dist < 1e-9:
                ux, uy = np.cos(th), np.sin(th)
            else:
                ux, uy = dx/dist, dy/dist

            r_robot = ellipse_radius_in_direction(a, b, ux, uy)
            d_safe = r_robot + margin

            # h = d^2 - d_safe^2
            h_val = (dx*dx + dy*dy) - (d_safe*d_safe)

            # p_dot = [ v cos(th) - l w sin(th),
            #           v sin(th) + l w cos(th) ]
            a_v = 2.0*(dx*np.cos(th) + dy*np.sin(th))
            a_w = 2.0*l*(-dx*np.sin(th) + dy*np.cos(th))

            # CBF: a_v v + a_w w + alpha*h >= 0
            # => -(a_v v + a_w w) <= alpha*h
            G_list.append([-a_v, -a_w])
            h_list.append(alpha*h_val)

        return G_list, h_list

    G_list_all, h_list_all = [], []

    for key, poly in enumerate((barrier_inner, barrier_outer)):
        Gi, hi = add_constraints_from_poly(poly, key)
        G_list_all.extend(Gi)
        h_list_all.extend(hi)

    if len(G_list_all) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    return np.array(G_list_all, dtype=float), np.array(h_list_all, dtype=float)
