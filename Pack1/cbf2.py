import numpy as np


def ellipse_radius_in_direction(a, b, ux, uy):
    denom = (ux / max(1e-9, a))**2 + (uy / max(1e-9, b))**2
    return 1.0 / np.sqrt(max(1e-12, denom))


def cbf_rows_for_circle_obstacles(
    x, y, th, obstacles,
    ellipse_ab=(0.30, 0.20),
    margin=0.05,
    lookahead_l=0.35,
    alpha=2.0,
):
    """
    Returns G, h for G u <= h, with u = [v, w].

    Obstacles may be static {"x", "y", "r"} or dynamic
    {"x", "y", "r", "vx", "vy"}. Missing velocities default to zero.
    """
    a, b = ellipse_ab
    l = lookahead_l

    px = x + l * np.cos(th)
    py = y + l * np.sin(th)

    G_list = []
    h_list = []

    for obs in obstacles:
        ox, oy, ro = obs["x"], obs["y"], obs["r"]
        ovx = obs.get("vx", 0.0)
        ovy = obs.get("vy", 0.0)

        dx_c = x - ox
        dy_c = y - oy
        dist_c = np.hypot(dx_c, dy_c)
        if dist_c < 1e-6:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx_c / dist_c, dy_c / dist_c

        r_robot = ellipse_radius_in_direction(a, b, ux, uy)
        r_safe = ro + r_robot + margin

        dx = px - ox
        dy = py - oy
        h_val = dx * dx + dy * dy - r_safe * r_safe

        # h_dot = 2(p-o)^T(p_dot-o_dot)
        #       = a_v v + a_w w - 2(p-o)^T o_dot
        a_v = 2.0 * (dx * np.cos(th) + dy * np.sin(th))
        a_w = 2.0 * l * (-dx * np.sin(th) + dy * np.cos(th))
        obstacle_motion = 2.0 * (dx * ovx + dy * ovy)

        # a_v v + a_w w - obstacle_motion + alpha*h >= 0
        # => -(a_v v + a_w w) <= alpha*h - obstacle_motion
        G_list.append([-a_v, -a_w])
        h_list.append(alpha * h_val - obstacle_motion)

    return np.array(G_list, dtype=float), np.array(h_list, dtype=float)


def _closest_point_on_segment(px, py, ax, ay, bx, by):
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
    x, y, th,
    barrier_inner, barrier_outer,
    ellipse_ab=(0.30, 0.20),
    margin=0.05,
    lookahead_l=0.35,
    alpha=2.0,
    max_segments=40,
):
    """Static CBF rows for the track barriers."""
    a, b = ellipse_ab
    l = lookahead_l

    px = x + l * np.cos(th)
    py = y + l * np.sin(th)

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

        A = poly2[:-1]
        B = poly2[1:]

        d2 = np.empty(len(A), dtype=float)
        q_cache = np.empty((len(A), 2), dtype=float)

        for i, ((ax, ay), (bx, by)) in enumerate(zip(A, B)):
            qx, qy, _ = _closest_point_on_segment(px, py, ax, ay, bx, by)
            q_cache[i, 0] = qx
            q_cache[i, 1] = qy
            dx = px - qx
            dy = py - qy
            d2[i] = dx * dx + dy * dy

        m = min(max_segments, len(A))
        idxs = np.argpartition(d2, m - 1)[:m]

        for i in idxs:
            qx, qy = q_cache[i, 0], q_cache[i, 1]
            dx = px - qx
            dy = py - qy
            dist = np.hypot(dx, dy)

            if dist < 1e-9:
                ux, uy = np.cos(th), np.sin(th)
            else:
                ux, uy = dx / dist, dy / dist

            r_robot = ellipse_radius_in_direction(a, b, ux, uy)
            d_safe = r_robot + margin
            h_val = dx * dx + dy * dy - d_safe * d_safe

            a_v = 2.0 * (dx * np.cos(th) + dy * np.sin(th))
            a_w = 2.0 * l * (-dx * np.sin(th) + dy * np.cos(th))

            G_list.append([-a_v, -a_w])
            h_list.append(alpha * h_val)

        return G_list, h_list

    G_list_all, h_list_all = [], []
    for poly in (barrier_inner, barrier_outer):
        Gi, hi = add_constraints_from_poly(poly)
        G_list_all.extend(Gi)
        h_list_all.extend(hi)

    if len(G_list_all) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    return np.array(G_list_all, dtype=float), np.array(h_list_all, dtype=float)
