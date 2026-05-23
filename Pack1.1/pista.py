import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

from controller import build_spline_path


def show_static():
    waypoints = [
        (3.0, 3.0),(2.6, 3.5),(2.2, 4.2),(2.0, 5.0),(2.0, 6.2),
        (2.0, 7.4),(2.0, 8.8),(2.0, 10.2),(2.0, 11.6),(2.0, 13.0),
        (2.2, 13.6),(2.6, 14.5),(3.1, 14.8),(3.7, 15.0),(4.2, 15.0),
        (4.8, 14.9),(5.3, 14.6),(5.6, 14.1),(5.7, 13.5),
        (5.6, 12.6),(5.5, 11.6),(5.5, 10.6),(5.5, 9.8),
        (5.7, 9.2),(6.0, 8.7),(6.6, 8.4),(7.4, 8.4),(8.2, 8.5),
        (8.8, 8.9),(9.1, 9.6),(9.3, 10.4),(9.5, 11.6),(9.7, 12.6),
        (9.9, 13.4),(10.2, 14.0),(10.8, 14.6),
        (11.6, 15.0),(12.6, 15.0),(13.6, 15.0),(14.6, 14.8),
        (15.4, 14.4),(16.0, 13.6),
        (16.0, 12.4),(16.0, 11.2),(16.0, 10.0),(16.0, 8.8),
        (16.0, 7.6),(16.0, 6.4),(16.0, 5.0),
        (15.5, 3.6),(14.2, 2.6),(12.0, 2.0),
        (10.8, 2.0),(9.6, 2.0),(8.4, 4.0),(7.2, 4.0),
        (6.0, 2.0),(4.0, 2.5),(3.0, 3.0),
    ]

    px, py, pyaw, s = build_spline_path(waypoints, ds=0.01)

    # -----------------------------
    # Obstáculos (igual ao original)
    # -----------------------------
    n_obs = 5
    idxs = np.linspace(0, len(px)-1, n_obs+2, dtype=int)[1:-1]

    obstacles = []
    for k, idx in enumerate(idxs):
        x_path = px[idx]
        y_path = py[idx]
        yaw_path = pyaw[idx]

        nx = -np.sin(yaw_path)
        ny =  np.cos(yaw_path)

        side = (-1)**k
        offset = 0.3

        ox = x_path + side * offset * nx
        oy = y_path + side * offset * ny

        obstacles.append({
            "x": ox,
            "y": oy,
            "r": 0.35
        })

    # -----------------------------
    # Estado FIXO (igual ao início)
    # -----------------------------
    x, y, yaw = 1, 6, np.deg2rad(90)

    # Elipse
    a_ell, b_ell = 0.30, 0.20
    margin = 0.01

    # -----------------------------
    # Barreiras
    # -----------------------------
    inner_bar = np.loadtxt("barreira_suavizada_interna.txt")
    outer_bar = np.loadtxt("barreira_suavizada_externa.txt")

    inner_x, inner_y = inner_bar[:, 0], inner_bar[:, 1]
    outer_x, outer_y = outer_bar[:, 0], outer_bar[:, 1]

    # -----------------------------
    # DESENHO (igual ao teu loop)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(px, py, "--", label="Spline (referência)")

    # NÃO há trajetória hx,hy porque não há simulação

    ax.plot(inner_x, inner_y, "-", color="red", linewidth=2, label="Barreira interna")
    ax.plot(outer_x, outer_y, "-", color="red", linewidth=2, label="Barreira externa")

    for obs in obstacles:
        ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

    ell = Ellipse((x, y), width=2*a_ell, height=2*b_ell,
                  angle=np.degrees(yaw), fill=False)
    #ax.add_patch(ell)

    ell_safe = Ellipse((x, y),
                       width=2*(a_ell+margin),
                       height=2*(b_ell+margin),
                       angle=np.degrees(yaw),
                       fill=False)
    #ax.add_patch(ell_safe)

    #ax.plot(x, y, "o", label="Robô")
    #ax.arrow(x, y, 0.4*np.cos(yaw), 0.4*np.sin(yaw), head_width=0.15)

    ax.set_aspect("equal", "box")
    ax.grid(True)
    ax.set_title("Mapa estático (sem simulação)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.show()


if __name__ == "__main__":
    show_static()