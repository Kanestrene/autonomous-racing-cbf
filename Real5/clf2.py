import numpy as np


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def nearest_path_point_closed(px, py, x, y, last_idx=0, window=300, back=20):
    n = len(px)
    if n == 0:
        raise ValueError("Trajetoria vazia.")

    offsets = np.arange(-back, window)
    idxs = (last_idx + offsets) % n

    dx = px[idxs] - x
    dy = py[idxs] - y
    d2 = dx * dx + dy * dy

    j = np.argmin(d2)
    idx = idxs[j]
    return idx, np.sqrt(d2[j])


def advance_path_index_closed(s, start_idx, advance_s):
    n = len(s)
    if n == 0:
        raise ValueError("Trajetoria vazia.")

    if n == 1:
        return 0

    L = float(s[-1])
    if L <= 1e-9:
        return int(start_idx) % n

    start_idx = int(start_idx) % n
    target_s = (float(s[start_idx]) + float(advance_s)) % L

    idx_next = int(np.searchsorted(s, target_s, side="left")) % n
    idx_prev = (idx_next - 1) % n

    ds_next = (float(s[idx_next]) - target_s) % L
    ds_prev = (target_s - float(s[idx_prev])) % L

    if ds_prev <= ds_next:
        return idx_prev
    return idx_next


def path_curvature_closed(pyaw, s, idx):
    n = len(pyaw)
    im1 = (idx - 1) % n
    ip1 = (idx + 1) % n

    dyaw = wrap_to_pi(pyaw[ip1] - pyaw[im1])

    ds = s[ip1] - s[im1]
    L = s[-1]

    if ds < -0.5 * L:
        ds += L
    elif ds > 0.5 * L:
        ds -= L

    if abs(ds) < 1e-8:
        return 0.0

    return dyaw / ds


def clf_row_path_tracking(px, py, pyaw, s, robot_state,
                          last_idx=0,
                          v_ref=1.0,
                          qx=1.0, qy=1.0, qpsi=4.0,
                          eps=1.0,
                          lookahead_l=0.45):
    x, y, theta = robot_state
    l = lookahead_l

    # ponto lookahead
    xl = x + l * np.cos(theta)
    yl = y + l * np.sin(theta)

    # projeta a posicao real no caminho e avanca l metros ao longo dele
    idx_proj, _ = nearest_path_point_closed(px, py, x, y, last_idx=last_idx)
    idx = advance_path_index_closed(s, idx_proj, l - 0.3)

    xr = px[idx]
    yr = py[idx]
    psi_r = pyaw[idx]
    kappa_r = path_curvature_closed(pyaw, s, idx)

    w_ref = v_ref * kappa_r

    dx = xl - xr
    dy = yl - yr

    # erros no frame da trajetoria
    ex = np.cos(psi_r) * dx + np.sin(psi_r) * dy
    ey = -np.sin(psi_r) * dx + np.cos(psi_r) * dy
    epsi = wrap_to_pi(theta - psi_r)

    V = 0.5 * (qx * ex**2 + qy * ey**2 + qpsi * epsi**2)

    av = qx * ex * np.cos(epsi) + qy * ey * np.sin(epsi)

    aw = (
        - qx * ex * l * np.sin(epsi)
        + qy * ey * l * np.cos(epsi)
        + qpsi * epsi
    )

    c = (
        - qx * ex * v_ref
        + w_ref * ex * ey * (qx - qy)
        - qpsi * epsi * w_ref
    )

    G_clf = np.array([[av, aw, -1.0]], dtype=float)
    h_clf = np.array([-c - eps * V], dtype=float)

    info = {
        "idx": idx,
        "idx_proj": idx_proj,
        "V": V,
        "ex": ex,
        "ey": ey,
        "epsi": epsi,
        "xr": xr,
        "yr": yr,
        "psi_r": psi_r,
        "kappa_r": kappa_r,
        "w_ref": w_ref,
        "xl": xl,
        "yl": yl,
    }

    return G_clf, h_clf, info
