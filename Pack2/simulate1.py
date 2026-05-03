import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from pathlib import Path

import qp

from controller import (
    wrap_to_pi,
    build_spline_path,
    omega_to_delta,
    rate_limit,
)


def save_track_only():
    script_dir = Path(__file__).resolve().parent
    pdf_path = script_dir / "pista.pdf"

    waypoints = [
        (3.0, 3.0),
        (2.6, 3.5),
        (2.2, 4.2),
        (2.0, 5.0),
        (2.0, 6.2),
        (2.0, 7.4),
        (2.0, 8.8),
        (2.0, 10.2),
        (2.0, 11.6),
        (2.0, 13.0),
        (2.2, 13.6),
        (2.6, 14.5),
        (3.1, 14.8),
        (3.7, 15.0),
        (4.2, 15.0),
        (4.8, 14.9),
        (5.3, 14.6),
        (5.6, 14.1),
        (5.7, 13.5),
        (5.6, 12.6),
        (5.5, 11.6),
        (5.5, 10.6),
        (5.5, 9.8),
        (5.7, 9.2),
        (6.0, 8.7),
        (6.6, 8.4),
        (7.4, 8.4),
        (8.2, 8.5),
        (8.8, 8.9),
        (9.1, 9.6),
        (9.3, 10.4),
        (9.5, 11.6),
        (9.7, 12.6),
        (9.9, 13.4),
        (10.2, 14.0),
        (10.8, 14.6),
        (11.6, 15.0),
        (12.6, 15.0),
        (13.6, 15.0),
        (14.6, 14.8),
        (15.4, 14.4),
        (16.0, 13.6),
        (16.0, 12.4),
        (16.0, 11.2),
        (16.0, 10.0),
        (16.0, 8.8),
        (16.0, 7.6),
        (16.0, 6.4),
        (16.0, 5.0),
        (15.5, 3.6),
        (14.2, 2.6),
        (12.0, 2.0),
        (10.8, 2.0),
        (9.6, 2.0),
        (8.4, 4.0),
        (7.2, 4.0),
        (6.0, 2.0),
        (4.0, 2.5),
        (3.0, 3.0),
    ]

    px, py, pyaw, _ = build_spline_path(waypoints, ds=0.01)
    n_path = len(px)

    n_obs = 5
    idxs = np.linspace(0, n_path - 1, n_obs + 2, dtype=int)[1:-1]
    obstacles = []

    for k, idx in enumerate(idxs):
        x_path = px[idx]
        y_path = py[idx]
        yaw_path = pyaw[idx]

        nx = -np.sin(yaw_path)
        ny = np.cos(yaw_path)

        side = (-1) ** k
        offset = 0.3

        ox = x_path + side * offset * nx
        oy = y_path + side * offset * ny

        obstacles.append({
            "x": ox,
            "y": oy,
            "r": 0.35,
        })

    inner_bar = np.loadtxt(script_dir / "barreira_suavizada_interna.txt")
    outer_bar = np.loadtxt(script_dir / "barreira_suavizada_externa.txt")

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(px, py, "--")
    ax.plot(inner_bar[:, 0], inner_bar[:, 1], "-", linewidth=2, color="green")
    ax.plot(outer_bar[:, 0], outer_bar[:, 1], "-", linewidth=2, color="green")

    for obs in obstacles:
        ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

    ax.set_aspect("equal", "box")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Pista guardada em: {pdf_path}")
    return pdf_path


def simulate():
    script_dir = Path(__file__).resolve().parent
    pdf_path = script_dir / "simulate1_volta_completa.pdf"

    waypoints = [
        (3.0, 3.0),
        (2.6, 3.5),
        (2.2, 4.2),
        (2.0, 5.0),
        (2.0, 6.2),
        (2.0, 7.4),
        (2.0, 8.8),
        (2.0, 10.2),
        (2.0, 11.6),
        (2.0, 13.0),
        (2.2, 13.6),
        (2.6, 14.5),
        (3.1, 14.8),
        (3.7, 15.0),
        (4.2, 15.0),
        (4.8, 14.9),
        (5.3, 14.6),
        (5.6, 14.1),
        (5.7, 13.5),
        (5.6, 12.6),
        (5.5, 11.6),
        (5.5, 10.6),
        (5.5, 9.8),
        (5.7, 9.2),
        (6.0, 8.7),
        (6.6, 8.4),
        (7.4, 8.4),
        (8.2, 8.5),
        (8.8, 8.9),
        (9.1, 9.6),
        (9.3, 10.4),
        (9.5, 11.6),
        (9.7, 12.6),
        (9.9, 13.4),
        (10.2, 14.0),
        (10.8, 14.6),
        (11.6, 15.0),
        (12.6, 15.0),
        (13.6, 15.0),
        (14.6, 14.8),
        (15.4, 14.4),
        (16.0, 13.6),
        (16.0, 12.4),
        (16.0, 11.2),
        (16.0, 10.0),
        (16.0, 8.8),
        (16.0, 7.6),
        (16.0, 6.4),
        (16.0, 5.0),
        (15.5, 3.6),
        (14.2, 2.6),
        (12.0, 2.0),
        (10.8, 2.0),
        (9.6, 2.0),
        (8.4, 4.0),
        (7.2, 4.0),
        (6.0, 2.0),
        (4.0, 2.5),
        (3.0, 3.0),
    ]

    px, py, pyaw, s = build_spline_path(waypoints, ds=0.01)
    n_path = len(px)

    n_obs = 5
    idxs = np.linspace(0, n_path - 1, n_obs + 2, dtype=int)[1:-1]
    obstacles = []

    for k, idx in enumerate(idxs):
        x_path = px[idx]
        y_path = py[idx]
        yaw_path = pyaw[idx]

        nx = -np.sin(yaw_path)
        ny = np.cos(yaw_path)

        side = (-1) ** k
        offset = 0.3

        ox = x_path + side * offset * nx
        oy = y_path + side * offset * ny

        obstacles.append({
            "x": ox,
            "y": oy,
            "r": 0.35,
        })

    x, y, yaw, v = 2, 6, np.deg2rad(90), 0.0

    dt = 0.02
    T = 50.0
    steps = int(T / dt)

    v_ref = 2
    L0 = 0.1
    kv = 0.5

    w_max = 2.5

    a_ell, b_ell = 0.30, 0.20
    margin = 0.05

    last_near = 0
    hx, hy, ctes = [], [], []
    lap_progress_idx = 0.0
    prev_near_idx = None
    stop_requested = False

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    def on_key_press(event):
        nonlocal stop_requested
        if event.key in ("enter", "return"):
            stop_requested = True

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    delta = 0.0

    L = 0.26
    delta_max = np.deg2rad(25)
    delta_rate_max = np.deg2rad(300)

    inner_bar = np.loadtxt(script_dir / "barreira_suavizada_interna.txt")
    outer_bar = np.loadtxt(script_dir / "barreira_suavizada_externa.txt")
    inner_x, inner_y = inner_bar[:, 0], inner_bar[:, 1]
    outer_x, outer_y = outer_bar[:, 0], outer_bar[:, 1]

    def draw_frame(Ld, v_safe, w_safe, cte, show_labels=True):
        ax.clear()
        ax.set_facecolor("white")

        ax.plot(px, py, "--", label="Spline (referencia)")
        ax.plot(hx, hy, "-", label="Trajetoria robo (CLF + CBF)")

        ax.plot(inner_x, inner_y, "-", color = "green", linewidth=2, label="Barreira interna")
        ax.plot(outer_x, outer_y, "-", color = "green", linewidth=2, label="Barreira externa")

        for obs in obstacles:
            ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

        ell = Ellipse(
            (x, y),
            width=2 * a_ell,
            height=2 * b_ell,
            angle=np.degrees(yaw),
            fill=False,
        )
        ax.add_patch(ell)

        ell_safe = Ellipse(
            (x, y),
            width=2 * (a_ell + margin),
            height=2 * (b_ell + margin),
            angle=np.degrees(yaw),
            fill=False,
        )
        ax.add_patch(ell_safe)

        ax.plot(x, y, "o", label="Robo")
        ax.arrow(x, y, 0.4 * np.cos(yaw), 0.4 * np.sin(yaw), head_width=0.15)
        ax.add_patch(Circle((x, y), Ld, fill=False))

        ax.set_aspect("equal", "box")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if show_labels:
            ax.set_title(
                f"CLF + CBF-QP | Ld={Ld:.2f} | v={v_safe:.2f} | "
                f"w={w_safe:.2f} | cte~{cte:.3f}"
            )
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.pause(0.001)

    for k in range(steps):
        Ld = L0 + kv * abs(v)

        v_nom = v_ref
        w_nom = 0.0

        u_safe, clf_info = qp.cbf_clf_qp_filter(
            u_nom=(v_nom, w_nom),
            robot_state=(x, y, yaw),
            obstacles=obstacles,
            px=px,
            py=py,
            pyaw=pyaw,
            s=s,
            last_path_idx=last_near,
            ellipse_ab=(a_ell, b_ell),
            margin=margin,
            lookahead_l=0.1,
            alpha=5,
            eps_clf=1,
            q_clf=(1.0, 10.0, 0.01),
            W=(100000.0, 1.0),
            p_slack=50.0,
            v_ref=v_ref,
            v_bounds=(0.0, 2.0),
            w_bounds=(-w_max, w_max),
        )

        v_safe, w_safe = u_safe
        last_near = clf_info["idx"]
        cte = clf_info["ey"]

        if prev_near_idx is None:
            prev_near_idx = last_near
        else:
            delta_idx = last_near - prev_near_idx
            if delta_idx < -n_path / 2:
                delta_idx += n_path
            elif delta_idx > n_path / 2:
                delta_idx -= n_path

            lap_progress_idx += max(0.0, float(delta_idx))
            prev_near_idx = last_near

        kappa_max = np.tan(delta_max) / L
        w_max_speed = abs(v_safe) * kappa_max
        w_safe = np.clip(w_safe, -w_max_speed, w_max_speed)

        delta_cmd = omega_to_delta(w_safe, v_safe, L, v_min=0.2)
        delta_cmd = np.clip(delta_cmd, -delta_max, delta_max)

        delta = rate_limit(delta_cmd, delta, du_max=delta_rate_max * dt)

        x += v_safe * np.cos(yaw) * dt
        y += v_safe * np.sin(yaw) * dt
        yaw = wrap_to_pi(yaw + (v_safe / L) * np.tan(delta) * dt)

        hx.append(x)
        hy.append(y)
        ctes.append(cte)

        lap_completed = lap_progress_idx >= (n_path - 1)

        if k % 5 == 0 or lap_completed:
            draw_frame(Ld, v_safe, w_safe, cte)

        if stop_requested:
            draw_frame(Ld, v_safe, w_safe, cte, show_labels=False)
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            print(f"Simulacao encerrada por Enter. Figura guardada em: {pdf_path}")
            plt.ioff()
            plt.close(fig)
            return pdf_path

        if lap_completed:
            draw_frame(Ld, v_safe, w_safe, cte, show_labels=False)
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            print(f"Volta completa. Figura guardada em: {pdf_path}")
            plt.ioff()
            plt.close(fig)
            return pdf_path

    plt.ioff()
    print("A simulacao terminou por tempo maximo sem completar uma volta.")

    fig2, ax2 = plt.subplots()
    ax2.plot(ctes)
    ax2.set_title("Erro lateral (aprox) - CLF")
    ax2.set_xlabel("Passo")
    ax2.set_ylabel("cte~ [m]")
    ax2.grid(False)
    plt.show()


if __name__ == "__main__":
    simulate()
