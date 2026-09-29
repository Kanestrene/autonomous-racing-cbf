import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from pathlib import Path

import paper_qp

from controller import (
    wrap_to_pi,
    build_spline_path,
    omega_to_delta,
    rate_limit,
)
from simulate1 import WAYPOINTS


def simulate():
    script_dir = Path(__file__).resolve().parent
    pdf_path = script_dir / "simulate_paper_volta_completa.pdf"
    linear_speed_pdf_path = script_dir / "simulate_paper_velocidade_linear.pdf"
    angular_speed_pdf_path = script_dir / "simulate_paper_velocidade_angular.pdf"
    delta_pdf_path = script_dir / "simulate_paper_delta.pdf"
    cpu_pdf_path = script_dir / "simulate_paper_tempo_cpu.pdf"
    qp_pdf_path = script_dir / "simulate_paper_tempo_qp.pdf"
    lat_pdf_path = script_dir / "simulate_paper_erro_lateral.pdf"

    waypoints = WAYPOINTS

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

    # (x, y) = eixo traseiro (paper, eq. 1); centro = (x + l cos, y + l sin) (eq. 2)
    x, y, yaw = 2, 6, np.deg2rad(90)

    dt = 0.02
    T = 50.0
    steps = int(T / dt)

    w_max = 2.5

    # parâmetros do paper (Tab. I): l, alpha, p
    # gamma, d_goal, P, H, Q: não vêm no paper / ajustados para seguir a pista
    l = 0.15
    r_robot = 0.30
    margin = 0.05
    d_goal = 1.5

    # só para o desenho (igual ao simulate1)
    a_ell, b_ell = 0.30, 0.20
    Ld = 0.0

    last_near = 0
    hx, hy, ctes = [], [], []
    v_qp_hist, v_applied_hist = [], []
    omega_qp_hist, omega_applied_hist = [], []
    delta_qp_hist, delta_applied_hist = [], []
    qp_time_hist = []
    step_time_hist = []
    seg_hint = {}  # janela de segmentos das paredes (atualizada pelo QP)
    lap_progress_idx = 0.0
    prev_near_idx = None
    stop_requested = False
    u_pre = np.array([0.0, 0.0])
    goal = (px[0], py[0], pyaw[0])

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

    def draw_frame(v_safe, w_safe, cte, show_labels=True):
        ax.clear()
        ax.set_facecolor("white")

        ax.plot(px, py, "--", label="Spline (referencia)")
        ax.plot(hx, hy, "-", label="Trajetoria robo (paper)")

        ax.plot(inner_x, inner_y, "-", color="green", linewidth=2, label="Barreira interna")
        ax.plot(outer_x, outer_y, "-", color="green", linewidth=2, label="Barreira externa")

        for obs in obstacles:
            ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

        # mesmo desenho do simulate1, centrado no centro do robô
        xc = x + l * np.cos(yaw)
        yc = y + l * np.sin(yaw)

        ell = Ellipse(
            (xc, yc),
            width=2 * a_ell,
            height=2 * b_ell,
            angle=np.degrees(yaw),
            fill=False,
        )
        ax.add_patch(ell)

        ell_safe = Ellipse(
            (xc, yc),
            width=2 * (a_ell + margin),
            height=2 * (b_ell + margin),
            angle=np.degrees(yaw),
            fill=False,
        )
        ax.add_patch(ell_safe)

        ax.plot(xc, yc, "o", label="Robo")
        ax.arrow(xc, yc, 0.4 * np.cos(yaw), 0.4 * np.sin(yaw), head_width=0.15)
        ax.add_patch(Circle((xc, yc), Ld, fill=False))

        ax.set_aspect("equal", "box")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if show_labels:
            ax.set_title(
                f"CLF-CBF-QP (paper) | v={v_safe:.2f} | "
                f"w={w_safe:.2f} | cte~{cte:.3f} | "
                f"QP={qp_time_hist[-1] * 1e3:.2f} ms"
            )
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.pause(0.001)

    def show_velocity_plots():
        if not v_qp_hist:
            return

        step_ms = np.array(step_time_hist) * 1e3
        qp_ms = np.array(qp_time_hist) * 1e3
        lat = np.array(ctes)  # erro lateral do centro do robô à spline
        print(
            f"Tempo CPU controlador [ms]: media={step_ms.mean():.3f} | "
            f"mediana={np.median(step_ms):.3f} | max={step_ms.max():.3f} | N={len(step_ms)}"
        )
        print(
            f"Tempo CPU so solve_qp [ms]: media={qp_ms.mean():.3f} | "
            f"mediana={np.median(qp_ms):.3f} | max={qp_ms.max():.3f}"
        )
        print(
            f"Erro lateral [m]: medio |e|={np.mean(np.abs(lat)):.4f} | "
            f"rms={np.sqrt(np.mean(lat ** 2)):.4f} | max={np.max(np.abs(lat)):.4f}"
        )

        t = np.arange(len(v_qp_hist)) * dt
        line_width = 2.2
        title_fontsize = 14
        label_fontsize = 13
        tick_fontsize = 11
        legend_fontsize = 12

        fig_v, ax_v = plt.subplots()
        ax_v.plot(t, v_qp_hist, label="QP", linewidth=line_width)
        ax_v.plot(t, v_applied_hist, label="Applied", linewidth=line_width)
        ax_v.set_title("Linear velocity: QP vs Applied", fontsize=title_fontsize)
        ax_v.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax_v.set_ylabel("v [m/s]", fontsize=label_fontsize)
        ax_v.grid(False)
        ax_v.tick_params(axis="both", labelsize=tick_fontsize)
        ax_v.legend(fontsize=legend_fontsize)
        fig_v.tight_layout()
        fig_v.savefig(linear_speed_pdf_path, format="pdf", bbox_inches="tight")

        fig_omega, ax_omega = plt.subplots()
        ax_omega.plot(t, omega_qp_hist, label="QP", linewidth=line_width)
        ax_omega.plot(t, omega_applied_hist, label="Applied", linewidth=line_width)
        ax_omega.set_title("Angular velocity: QP vs Applied", fontsize=title_fontsize)
        ax_omega.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax_omega.set_ylabel("w [rad/s]", fontsize=label_fontsize)
        ax_omega.grid(False)
        ax_omega.tick_params(axis="both", labelsize=tick_fontsize)
        ax_omega.legend(fontsize=legend_fontsize)
        fig_omega.tight_layout()
        fig_omega.savefig(angular_speed_pdf_path, format="pdf", bbox_inches="tight")

        fig_delta, ax_delta = plt.subplots()
        ax_delta.plot(t, delta_qp_hist, label="QP", linewidth=line_width)
        ax_delta.plot(t, delta_applied_hist, label="Applied", linewidth=line_width)
        ax_delta.set_title("Steering angle: QP vs Applied", fontsize=title_fontsize)
        ax_delta.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax_delta.set_ylabel("delta [rad]", fontsize=label_fontsize)
        ax_delta.grid(False)
        ax_delta.tick_params(axis="both", labelsize=tick_fontsize)
        ax_delta.legend(fontsize=legend_fontsize)
        fig_delta.tight_layout()
        fig_delta.savefig(delta_pdf_path, format="pdf", bbox_inches="tight")

        fig_cpu, ax_cpu = plt.subplots()
        ax_cpu.plot(t, step_ms, label="Total time", linewidth=line_width)
        ax_cpu.axhline(step_ms.mean(), linestyle="--", color="C1",
                       label=f"Mean = {step_ms.mean():.2f} ms")
        ax_cpu.set_title("Total time per step", fontsize=title_fontsize)
        ax_cpu.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax_cpu.set_ylabel("Time [ms]", fontsize=label_fontsize)
        ax_cpu.grid(False)
        ax_cpu.tick_params(axis="both", labelsize=tick_fontsize)
        ax_cpu.legend(fontsize=legend_fontsize)
        fig_cpu.tight_layout()
        fig_cpu.savefig(cpu_pdf_path, format="pdf", bbox_inches="tight")

        fig_qp, ax_qp = plt.subplots()
        ax_qp.plot(t, qp_ms, label="solve_qp", linewidth=line_width)
        ax_qp.axhline(qp_ms.mean(), linestyle="--", color="C1",
                      label=f"Mean = {qp_ms.mean():.3f} ms")
        ax_qp.set_title("QP solve time per step", fontsize=title_fontsize)
        ax_qp.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax_qp.set_ylabel("QP time [ms]", fontsize=label_fontsize)
        ax_qp.grid(False)
        ax_qp.tick_params(axis="both", labelsize=tick_fontsize)
        ax_qp.legend(fontsize=legend_fontsize)
        fig_qp.tight_layout()
        fig_qp.savefig(qp_pdf_path, format="pdf", bbox_inches="tight")

        fig_lat, ax_lat = plt.subplots()
        ax_lat.plot(t, lat, label="Lateral error", linewidth=line_width)
        ax_lat.axhline(np.mean(np.abs(lat)), linestyle="--", color="C1",
                       label=f"Mean |e| = {np.mean(np.abs(lat)):.3f} m")
        ax_lat.axhline(0.0, color="black", linewidth=0.8)
        ax_lat.set_title("Lateral error to path", fontsize=title_fontsize)
        ax_lat.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax_lat.set_ylabel("e_lat [m]", fontsize=label_fontsize)
        ax_lat.grid(False)
        ax_lat.tick_params(axis="both", labelsize=tick_fontsize)
        ax_lat.legend(fontsize=legend_fontsize)
        fig_lat.tight_layout()
        fig_lat.savefig(lat_pdf_path, format="pdf", bbox_inches="tight")

        print(
            "Graficos guardados em: "
            f"{linear_speed_pdf_path}, {angular_speed_pdf_path}, {delta_pdf_path}, "
            f"{cpu_pdf_path}, {qp_pdf_path} e {lat_pdf_path}"
        )
        plt.show()

    for k in range(steps):
        t_ctrl0 = time.perf_counter()
        goal, last_near = paper_qp.paper_goal_on_path(
            px, py, pyaw, s, (x, y, yaw), l, last_near, d_goal
        )

        u_safe, info = paper_qp.paper_clf_cbf_qp(
            u_pre=u_pre,
            robot_state=(x, y, yaw),
            goal=goal,
            obstacles=obstacles,
            barrier_inner=inner_bar,
            barrier_outer=outer_bar,
            l=l,
            r_robot=r_robot,
            margin=margin,
            alpha=5,     # paper
            gamma=3.0,     # paper: 0.5 (com objetivo movel dava v ~ 0.4 m/s)
            p_slack=100,  # paper
            H=(1.0, 1.0),
            Q=(10.0, 10.0),
            P_clf=((1.0, 0.0, 0.0),
                   (0.0, 4.0, 0.5),
                   (0.0, 0.5, 0.5)),
            v_bounds=(0.0, 2.0),
            w_bounds=(-w_max, w_max),
            kappa_max=np.tan(delta_max) / L,
            seg_hint=seg_hint,
        )

        step_time_hist.append(time.perf_counter() - t_ctrl0)
        v_safe, w_safe = u_safe
        w_qp = w_safe
        qp_time_hist.append(info["qp_time"])

        # erro de seguimento: centro do robô vs spline
        xc = x + l * np.cos(yaw)
        yc = y + l * np.sin(yaw)
        cte = (-np.sin(pyaw[last_near]) * (xc - px[last_near])
               + np.cos(pyaw[last_near]) * (yc - py[last_near]))

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
        w_applied = (v_safe / L) * np.tan(delta)

        u_pre = np.array([v_safe, w_applied])

        v_qp_hist.append(v_safe)
        v_applied_hist.append(v_safe)
        omega_qp_hist.append(w_qp)
        omega_applied_hist.append(w_applied)
        delta_qp_hist.append(np.clip(omega_to_delta(w_qp, v_safe, L, v_min=0.2), -delta_max, delta_max))
        delta_applied_hist.append(delta)

        x += v_safe * np.cos(yaw) * dt
        y += v_safe * np.sin(yaw) * dt
        yaw = wrap_to_pi(yaw + w_applied * dt)

        hx.append(x + l * np.cos(yaw))
        hy.append(y + l * np.sin(yaw))
        ctes.append(cte)

        lap_completed = lap_progress_idx >= (n_path - 1)

        if k % 5 == 0 or lap_completed:
            draw_frame(v_safe, w_applied, cte)

        if stop_requested:
            draw_frame(v_safe, w_applied, cte, show_labels=False)
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            print(f"Simulacao encerrada por Enter. Figura guardada em: {pdf_path}")
            plt.ioff()
            plt.close(fig)
            show_velocity_plots()
            return pdf_path

        if lap_completed:
            draw_frame(v_safe, w_applied, cte, show_labels=False)
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            print(f"Volta completa. Figura guardada em: {pdf_path}")
            plt.ioff()
            plt.close(fig)
            show_velocity_plots()
            return pdf_path

    plt.ioff()
    plt.close(fig)
    print("A simulacao terminou por tempo maximo sem completar uma volta.")
    show_velocity_plots()


if __name__ == "__main__":
    simulate()
