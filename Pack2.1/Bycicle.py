import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from scipy.interpolate import CubicSpline
import qp  
from cbf import cbf_rows_for_circle_obstacles


# -----------------------------
# Utilidades
# -----------------------------
def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def build_spline_path(waypoints_xy, ds=0.05):
    w = np.array(waypoints_xy, dtype=float)
    x = w[:, 0]
    y = w[:, 1]
    d = np.hypot(np.diff(x), np.diff(y))
    t = np.concatenate(([0.0], np.cumsum(d)))
    if t[-1] < 1e-9:
        raise ValueError("Waypoints inválidos (todos iguais?).")

    sx = CubicSpline(t, x)
    sy = CubicSpline(t, y)

    s = np.arange(0.0, t[-1], ds)
    px = sx(s)
    py = sy(s)

    dx = sx(s, 1)
    dy = sy(s, 1)
    pyaw = np.arctan2(dy, dx)
    return px, py, pyaw, s

def nearest_point_index(px, py, x, y, last_idx=0, search_window=250):
    n = len(px)
    i0 = max(0, last_idx)
    i1 = min(n, last_idx + search_window)
    if i0 >= i1:
        i0, i1 = 0, n
    dx = px[i0:i1] - x
    dy = py[i0:i1] - y
    return i0 + int(np.argmin(dx * dx + dy * dy))

# -----------------------------
# Pure Pursuit (para unicycle -> w)
# -----------------------------
def pure_pursuit_control(px, py, state, last_near_idx, Ld=0.9, v_ref=1.0):
    x, y, yaw, v = state
    near_idx = nearest_point_index(px, py, x, y, last_idx=last_near_idx)

    n = len(px)
    target_idx = near_idx
    for j in range(near_idx, min(n, near_idx + 2000)):
        if np.hypot(px[j] - x, py[j] - y) >= Ld:
            target_idx = j
            break

    tx, ty = px[target_idx], py[target_idx]
    angle_to_target = np.arctan2(ty - y, tx - x)
    alpha = wrap_to_pi(angle_to_target - yaw)

    kappa = 2.0 * np.sin(alpha) / max(1e-3, Ld)
    v_cmd = v_ref
    w_cmd = v_cmd * kappa

    cte_approx = np.sin(alpha) * np.hypot(tx - x, ty - y)
    return v_cmd, w_cmd, target_idx, near_idx, cte_approx

def omega_to_delta(omega, v, L, v_min=0.2):
    """Converte yaw-rate omega para ângulo de direção delta (rad)."""
    v_eff = max(abs(v), v_min)
    return np.arctan((L * omega) / v_eff)

def rate_limit(u, u_prev, du_max):
    """Limita a variação por passo: u in [u_prev-du_max, u_prev+du_max]."""
    return np.clip(u, u_prev - du_max, u_prev + du_max)

# -----------------------------
# Simulação robô diferencial (unicycle)
# -----------------------------
def simulate():
    waypoints = [
    (0.0, 0.0),
    (2.0, 1.0),
    (4.0, 0.0),
    (6.0, 2.0),
    (8.0, 2.0),
    (10.0, 0.0),
    (12.0, -2.0),
    (14.0, -1.0),
    (16.0, 1.5),
    (18.0, 0.0),
    (20.0, -2.5),
    (22.0, -1.0),
    (24.0, 1.5),
    (26.0, 0.0),
    ]

    px, py, pyaw, s = build_spline_path(waypoints, ds=0.05)

    # Estado inicial
    x, y, yaw, v = -0.5, -0.5, np.deg2rad(20), 0.0

    dt = 0.02
    T = 25.0
    steps = int(T / dt)

    # Pure Pursuit
    v_ref = 1.5
    L0 = 0.1
    kv = 0.5

    # Limites
    w_max = 2.5
    a_max = 2.0

    # Obstáculos (define os teus aqui)
    obstacles = [
    # Primeira subida
    {"x": 2.5, "y": 1.4, "r": 0.45},

    # Primeira descida forte
    {"x": 4.5, "y": -1.4, "r": 0.50},

    # Segunda subida
    {"x": 6.5, "y": 1.5, "r": 0.45},

    # Segunda descida
    {"x": 8.5, "y": -1.6, "r": 0.50},

    # Terceira subida
    {"x": 10.5, "y": 1.6, "r": 0.45},

    # Terceira descida
    {"x": 12.5, "y": -1.5, "r": 0.50},

    # Zona final ondulada
    {"x": 15.0, "y": 0.8, "r": 0.55},
    {"x": 17.0, "y": -0.8, "r": 0.55},
    ]

    # Elipse do robô
    a_ell, b_ell = 0.30, 0.20   # semi-eixos [m]
    margin = 0.05              # margem CBF/desenho

    last_near = 0
    hx, hy, ctes = [], [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))

    delta = 0.0

    for k in range(steps):
        # lookahead adaptativo (usa velocidade atual)
        Ld = L0 + kv * abs(v)

        state = (x, y, yaw, v)
        v_cmd, w_cmd, target_idx, last_near, cte = pure_pursuit_control(
            px, py, state, last_near_idx=last_near, Ld=Ld, v_ref=v_ref
        )

        # saturação nominal (antes do QP)
        w_cmd = np.clip(w_cmd, -w_max, w_max)

        # dinâmica simples para v (para atualizar v antes de aplicar QP, opcional)
        dv = np.clip(v_cmd - v, -a_max * dt, a_max * dt)
        v = v + dv

        # -------- CBF-QP (filtra v_cmd,w_cmd) --------
        v_safe, w_safe = qp.cbf_qp_filter(
            u_nom=(v_cmd, w_cmd),
            robot_state=(x, y, yaw),
            obstacles=obstacles,
            ellipse_ab=(a_ell, b_ell),
            margin=margin,
            lookahead_l=0.2,
            alpha=2.0,
            W=(20.0, 1.0),
            v_bounds=(0.0, 1.5),
            w_bounds=(-w_max, w_max),
        )        

        # integrar com os comandos seguros
        L = 0.26               # <-- mete aqui o teu entre-eixos (m)
        delta_max = np.deg2rad(25)   # <-- limite do servo (exemplo)
        delta_rate_max = np.deg2rad(300)  # <-- rad/s (exemplo, ajusta)

        kappa_max = np.tan(delta_max) / L
        w_max_speed = abs(v_safe) * kappa_max
        w_safe = np.clip(w_safe, -w_max_speed, w_max_speed)

        # guarda delta anterior (define fora do loop: delta = 0.0)
        delta_cmd = omega_to_delta(w_safe, v_safe, L, v_min=0.2)
        delta_cmd = np.clip(delta_cmd, -delta_max, delta_max)

        # limita taxa do servo
        delta = rate_limit(delta_cmd, delta, du_max=delta_rate_max * dt)

        # integra bicycle cinemático
        x += v_safe * np.cos(yaw) * dt
        y += v_safe * np.sin(yaw) * dt
        yaw = wrap_to_pi(yaw + (v_safe / L) * np.tan(delta) * dt)

        hx.append(x)
        hy.append(y)
        ctes.append(cte)

        if target_idx >= len(px) - 5 and np.hypot(px[-1] - x, py[-1] - y) < 0.2:
            break

        # desenho
        if k % 5 == 0:
            ax.clear()

            # caminho e trajetória
            ax.plot(px, py, "--", label="Spline (referência)")
            ax.plot(hx, hy, "-", label="Trajetória robô (PP + CBF)")

            # obstáculos
            for obs in obstacles:
                ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

            # elipse do robô e elipse com margem (visual)
            ell = Ellipse((x, y), width=2*a_ell, height=2*b_ell,
                          angle=np.degrees(yaw), fill=False)
            ax.add_patch(ell)

            ell_safe = Ellipse((x, y), width=2*(a_ell+margin), height=2*(b_ell+margin),
                               angle=np.degrees(yaw), fill=False)
            ax.add_patch(ell_safe)

            # robô (ponto + heading)
            ax.plot(x, y, "o", label="Robô")
            ax.arrow(x, y, 0.4*np.cos(yaw), 0.4*np.sin(yaw), head_width=0.15)

            # alvo PP
            ax.plot(px[target_idx], py[target_idx], "x", markersize=10, label="Alvo (lookahead)")

            # círculo do lookahead
            ax.add_patch(Circle((x, y), Ld, fill=False))

            ax.set_aspect("equal", "box")
            ax.grid(True)
            ax.set_title(f"PP + CBF-QP | Ld={Ld:.2f} | v={v_safe:.2f} | w={w_safe:.2f} | cte~{cte:.3f}")
            ax.legend(loc="best")
            plt.pause(0.001)

    plt.ioff()

    fig2, ax2 = plt.subplots()
    ax2.plot(ctes)
    ax2.set_title("Erro lateral (aprox) - Pure Pursuit")
    ax2.set_xlabel("Passo")
    ax2.set_ylabel("cte~ [m]")
    ax2.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate()
