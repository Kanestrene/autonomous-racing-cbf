# controller.py
import numpy as np
from scipy.interpolate import CubicSpline

# -----------------------------
# Utilidades
# -----------------------------
def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def build_spline_path(waypoints_xy, ds=0.05):
    w = np.array(waypoints_xy, dtype=float)
    x = w[:, 0]
    y = w[:, 1]

    # garantir que fecha exatamente
    if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
        w = np.vstack([w, w[0]])
        x = w[:, 0]
        y = w[:, 1]

    # parâmetro t por comprimento acumulado
    d = np.hypot(np.diff(x), np.diff(y))
    t = np.concatenate(([0.0], np.cumsum(d)))
    if t[-1] < 1e-9:
        raise ValueError("Waypoints inválidos (todos iguais?).")

    # spline periódica => junta suave (posição + derivadas)
    sx = CubicSpline(t, x, bc_type="periodic")
    sy = CubicSpline(t, y, bc_type="periodic")

    s = np.arange(0.0, t[-1], ds)
    px = sx(s)
    py = sy(s)

    dx = sx(s, 1)
    dy = sy(s, 1)
    pyaw = np.arctan2(dy, dx)

    return px, py, pyaw, s

def nearest_point_index(px, py, x, y, last_idx=0, search_window=300):
    n = len(px)
    last_idx = int(last_idx) % n

    idxs = (last_idx + np.arange(search_window)) % n
    dx = px[idxs] - x
    dy = py[idxs] - y
    return int(idxs[np.argmin(dx*dx + dy*dy)])

# -----------------------------
# Pure Pursuit (para unicycle -> w)
# -----------------------------
def pure_pursuit_control(px, py, state, last_near_idx, Ld=0.9, v_ref=1.0, k_pp=2.0):
    x, y, yaw, v = state
    n = len(px)

    # índice do ponto mais próximo (mantém continuidade)
    near_idx = nearest_point_index(px, py, x, y, last_idx=last_near_idx) % n

    # procurar alvo à frente COM WRAP (pista fechada)
    target_idx = near_idx
    for k in range(min(2000, n)):
        j = (near_idx + k) % n
        if np.hypot(px[j] - x, py[j] - y) >= Ld:
            target_idx = j
            break

    tx, ty = px[target_idx], py[target_idx]
    angle_to_target = np.arctan2(ty - y, tx - x)
    alpha = wrap_to_pi(angle_to_target - yaw)

    # k_pp permite afinar a agressividade do pure pursuit sem mudar a geometria base.
    kappa = k_pp * np.sin(alpha) / max(1e-3, Ld)
    v_cmd = v_ref
    w_cmd = v_cmd * kappa

    cte_approx = np.sin(alpha) * np.hypot(tx - x, ty - y)
    return v_cmd, w_cmd, target_idx, near_idx, cte_approx

# -----------------------------
# Helpers para bicycle/servo
# -----------------------------
def omega_to_delta(omega, v, L, v_min=0.2):
    """Converte yaw-rate omega para ângulo de direção delta (rad)."""
    v_eff = max(abs(v), v_min)
    return np.arctan((L * omega) / v_eff)

def rate_limit(u, u_prev, du_max):
    """Limita a variação por passo: u in [u_prev-du_max, u_prev+du_max]."""
    return np.clip(u, u_prev - du_max, u_prev + du_max)
