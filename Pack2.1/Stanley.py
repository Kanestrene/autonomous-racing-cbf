import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# -----------------------------
# Utilidades
# -----------------------------
def wrap_to_pi(a: float) -> float:
    """Normaliza ângulo para [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

def build_spline_path(waypoints_xy, ds=0.05):
    """
    Cria uma spline paramétrica x(s), y(s) por comprimento aproximado e amostra em passo ds.
    Retorna arrays: px, py, pyaw, s
    """
    w = np.array(waypoints_xy, dtype=float)
    x = w[:, 0]
    y = w[:, 1]

    # parâmetro t acumulado (distância entre waypoints)
    d = np.hypot(np.diff(x), np.diff(y))
    t = np.concatenate(([0.0], np.cumsum(d)))
    if t[-1] < 1e-9:
        raise ValueError("Waypoints inválidos (todos iguais?).")

    # spline cúbica paramétrica
    sx = CubicSpline(t, x)
    sy = CubicSpline(t, y)

    # amostragem uniforme em "t" ~ distância
    s = np.arange(0.0, t[-1], ds)
    px = sx(s)
    py = sy(s)

    # derivadas para yaw do caminho
    dx = sx(s, 1)
    dy = sy(s, 1)
    pyaw = np.arctan2(dy, dx)

    return px, py, pyaw, s

def nearest_point_index(px, py, x, y, last_idx=0, search_window=200):
    """
    Acha o índice do ponto do caminho mais próximo.
    Para eficiência, procura numa janela à frente do last_idx.
    """
    n = len(px)
    i0 = max(0, last_idx)
    i1 = min(n, last_idx + search_window)
    if i0 >= i1:
        i0, i1 = 0, n

    dx = px[i0:i1] - x
    dy = py[i0:i1] - y
    dist2 = dx * dx + dy * dy
    local = int(np.argmin(dist2))
    return i0 + local

# -----------------------------
# Controlador Stanley (para unicycle)
# -----------------------------
def stanley_control(px, py, pyaw, state, last_idx, k=1.0, ks=0.0, v_ref=0.8):
    """
    Stanley clássico: delta = heading_error + atan2(k * cte, v)
    Para robô diferencial (unicycle), geramos w diretamente:
        w = k_yaw * heading_error + atan2(k * cte, v)
    (Aqui usamos k_yaw=1 por simplicidade)
    ks: pequeno termo para estabilidade (opcional) na velocidade no denominador.
    """
    x, y, yaw, v = state

    idx = nearest_point_index(px, py, x, y, last_idx=last_idx)
    tx, ty, tyaw = px[idx], py[idx], pyaw[idx]

    # erro de heading
    heading_error = wrap_to_pi(tyaw - yaw)

    # erro lateral (cross-track) com sinal:
    # sinal via produto vetorial entre direção do caminho e vetor (robô->ponto)
    dx = tx - x
    dy = ty - y
    path_dir = np.array([np.cos(tyaw), np.sin(tyaw)])
    perp = np.array([-path_dir[1], path_dir[0]])  # normal à esquerda do caminho
    cte = np.dot(np.array([dx, dy]), perp)        # positivo = robô à esquerda do caminho

    # termo Stanley
    v_denom = max(1e-3, abs(v) + ks)
    stanley_term = np.arctan2(k * cte, v_denom)

    # "comando de curvatura" em forma de velocidade angular
    w_cmd = heading_error + stanley_term

    # velocidade desejada (pode ser constante, ou perfil)
    v_cmd = v_ref

    return v_cmd, w_cmd, idx, cte, heading_error

# -----------------------------
# Simulação robô diferencial (unicycle)
# -----------------------------
def simulate():
    # Waypoints (podes trocar pelos teus)
    waypoints = [
        (0.0, 0.0),
        (2.0, 1.0),
        (4.0, 0.0),
        (6.0, 2.0),
        (8.0, 2.0),
        (10.0, 0.0),
    ]

    # Construir caminho spline
    px, py, pyaw, s = build_spline_path(waypoints, ds=0.05)

    # Estado: x, y, yaw, v
    x, y, yaw, v = -0.5, -0.5, np.deg2rad(20), 0.0

    # Parâmetros de simulação
    dt = 0.02
    T = 25.0
    steps = int(T / dt)

    # Ganhos Stanley
    k = 10.0     # ganho do erro lateral
    ks = 0.2    # ajuda com v baixo
    v_ref = 1.0 # m/s

    # Limites (opcionais)
    w_max = 2.5        # rad/s
    a_max = 2.0        # m/s^2 (para dinâmica simples de v)

    last_idx = 0

    # Log
    hx, hy = [], []
    ctes = []

    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))

    for _ in range(steps):
        state = (x, y, yaw, v)
        v_cmd, w_cmd, last_idx, cte, e_yaw = stanley_control(
            px, py, pyaw, state, last_idx, k=k, ks=ks, v_ref=v_ref
        )

        # saturação de w
        w_cmd = np.clip(w_cmd, -w_max, w_max)

        # dinâmica simples para v (1ª ordem via aceleração limitada)
        dv = np.clip(v_cmd - v, -a_max * dt, a_max * dt)
        v += dv

        # cinemática unicycle (robô diferencial com comandos v,w)
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw = wrap_to_pi(yaw + w_cmd * dt)

        hx.append(x)
        hy.append(y)
        ctes.append(cte)

        # condição de término: perto do fim do caminho
        if last_idx >= len(px) - 5 and np.hypot(px[-1] - x, py[-1] - y) < 0.2:
            break

        # desenho
        if _ % 5 == 0:
            ax.clear()
            ax.plot(px, py, "--", label="Spline (referência)")
            ax.plot(hx, hy, "-", label="Trajetória robô")

            # robô
            ax.plot(x, y, "o", label="Robô")
            ax.arrow(x, y, 0.4*np.cos(yaw), 0.4*np.sin(yaw), head_width=0.15)

            # ponto alvo
            ax.plot(px[last_idx], py[last_idx], "x", markersize=10, label="Ponto mais próximo")

            ax.set_aspect("equal", "box")
            ax.grid(True)
            ax.set_title(f"Stanley | idx={last_idx} | cte={cte:.3f} m | e_yaw={np.rad2deg(e_yaw):.1f}° | v={v:.2f}")
            ax.legend(loc="best")
            plt.pause(0.001)

    plt.ioff()

    # Plot do erro lateral
    fig2, ax2 = plt.subplots()
    ax2.plot(ctes)
    ax2.set_title("Erro lateral (cross-track error)")
    ax2.set_xlabel("Passo")
    ax2.set_ylabel("cte [m]")
    ax2.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate()