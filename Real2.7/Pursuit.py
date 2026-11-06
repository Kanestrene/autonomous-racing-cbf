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

import numpy as np

def build_rounded_path(waypoints_xy, radius=0.5, ds=0.05):
    w = np.array(waypoints_xy, dtype=float)
    n = len(w)

    if n < 2:
        raise ValueError("Precisa de pelo menos 2 pontos")

    px, py = [], []

    for i in range(n):
        if i == 0 or i == n - 1:
            # Primeiro e último ponto entram direto
            px.append(w[i, 0])
            py.append(w[i, 1])
            continue

        p0 = w[i - 1]
        p1 = w[i]
        p2 = w[i + 1]

        v1 = p1 - p0
        v2 = p2 - p1

        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        # Ângulo entre segmentos
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        theta = np.arccos(dot)

        if theta < 1e-3:
            # quase reto
            px.append(p1[0])
            py.append(p1[1])
            continue

        # Distância do ponto até tangência
        d = radius / np.tan(theta / 2)

        # Limita d ao tamanho do segmento
        d = min(d, np.linalg.norm(p1 - p0) * 0.5, np.linalg.norm(p2 - p1) * 0.5)

        t1 = p1 - v1 * d
        t2 = p1 + v2 * d

        # Adiciona ponto de tangência inicial
        px.append(t1[0])
        py.append(t1[1])

        # Centro do arco
        bisector = (v2 - v1)
        bisector /= np.linalg.norm(bisector)

        center = p1 + bisector * (radius / np.sin(theta / 2))

        # Ângulos do arco
        ang1 = np.arctan2(t1[1] - center[1], t1[0] - center[0])
        ang2 = np.arctan2(t2[1] - center[1], t2[0] - center[0])

        # Determina sentido
        if np.cross(v1, v2) > 0:
            if ang2 < ang1:
                ang2 += 2*np.pi
        else:
            if ang2 > ang1:
                ang2 -= 2*np.pi

        arc = np.arange(ang1, ang2, ds / radius)

        for a in arc:
            px.append(center[0] + radius * np.cos(a))
            py.append(center[1] + radius * np.sin(a))

        # ponto de tangência final
        px.append(t2[0])
        py.append(t2[1])

    px = np.array(px)
    py = np.array(py)

    # yaw
    dx = np.gradient(px)
    dy = np.gradient(py)
    pyaw = np.arctan2(dy, dx)

    return px, py, pyaw

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

    kappa = 0.5 * np.sin(alpha) / max(1e-3, Ld)
    v_cmd = v_ref
    w_cmd = v_cmd * kappa

    cte_approx = np.sin(alpha) * np.hypot(tx - x, ty - y)
    return v_cmd, w_cmd, target_idx, near_idx, cte_approx

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
    ]

    px, py, pyaw = build_rounded_path(waypoints, ds=0.01)

    # Estado inicial
    x, y, yaw, v = -0.5, -0.5, np.deg2rad(20), 0.0

    dt = 0.02
    T = 25.0
    steps = int(T / dt)

    # Pure Pursuit
    v_ref = 5.0
    L0 = 0.3
    kv = 0.1

    # Limites
    w_max = 2.5
    a_max = 2.0

    # Obstáculos (define os teus aqui)
    obstacles = [
        
    ]

    # Elipse do robô
    a_ell, b_ell = 0.30, 0.20   # semi-eixos [m]
    margin = 0.05              # margem CBF/desenho

    last_near = 0
    hx, hy, ctes = [], [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))

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
            lookahead_l=0.35,
            alpha=2.0,
            W=(20.0, 1.0),
            v_bounds=(0.0, 5),
            w_bounds=(-w_max, w_max),
        )

        # integrar com os comandos seguros
        x += v_safe * np.cos(yaw) * dt
        y += v_safe * np.sin(yaw) * dt
        yaw = wrap_to_pi(yaw + w_safe * dt)

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
