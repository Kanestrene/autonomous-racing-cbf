import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from pathlib import Path

import qp
from clf import nearest_path_point_closed

from controller import (
    wrap_to_pi,
    build_spline_path,
    omega_to_delta,
    rate_limit,
)


WAYPOINTS = [
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
    (8.4, 2.0),
    (7.2, 2.0),
    (6.0, 2.0),
    (4.0, 2.5),
    (3.0, 3.0),
]


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
    pdf_path = script_dir / "simulate1_custo_paper_volta_completa.pdf"
    linear_speed_pdf_path = script_dir / "simulate1_custo_paper_velocidade_linear.pdf"
    angular_speed_pdf_path = script_dir / "simulate1_custo_paper_velocidade_angular.pdf"
    delta_pdf_path = script_dir / "simulate1_custo_paper_delta.pdf"
    cpu_pdf_path = script_dir / "simulate1_custo_paper_tempo_cpu.pdf"
    qp_pdf_path = script_dir / "simulate1_custo_paper_tempo_qp.pdf"
    lat_pdf_path = script_dir / "simulate1_custo_paper_erro_lateral.pdf"

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
    v_controller_hist, v_qp_hist = [], []
    omega_controller_hist, omega_qp_hist = [], []
    delta_controller_hist, delta_qp_hist = [], []
    qp_time_hist = []
    step_time_hist = []
    seg_hint = {}  # janela de segmentos das barreiras (atualizada pelo QP)
    lat_err_hist = []
    lat_idx = 0
    u_pre = np.array([0.0, 0.0])
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
                f"CLF + CBF-QP (custo paper) | Ld={Ld:.2f} | v={v_safe:.2f} | "
                f"w={w_safe:.2f} | cte~{cte:.3f} | "
                f"QP={qp_time_hist[-1] * 1e3:.2f} ms"
            )
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.pause(0.001)

    def show_velocity_plots():
        if not v_controller_hist:
            return

        step_ms = np.array(step_time_hist) * 1e3
        qp_ms = np.array(qp_time_hist) * 1e3
        lat = np.array(lat_err_hist)
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

        t = np.arange(len(v_controller_hist)) * dt
        line_width = 2.2
        title_fontsize = 14
        label_fontsize = 13
        tick_fontsize = 11
        legend_fontsize = 12

        fig_v, ax_v = plt.subplots()
        ax_v.plot(t, v_controller_hist, label="Controller", linewidth=line_width)
        ax_v.plot(t, v_qp_hist, label="QP", linewidth=line_width)
        ax_v.set_title("Linear velocity: Controller vs QP", fontsize=title_fontsize)
        ax_v.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax_v.set_ylabel("v [m/s]", fontsize=label_fontsize)
        ax_v.grid(False)
        ax_v.tick_params(axis="both", labelsize=tick_fontsize)
        ax_v.legend(fontsize=legend_fontsize)
        fig_v.tight_layout()
        fig_v.savefig(linear_speed_pdf_path, format="pdf", bbox_inches="tight")

        fig_omega, ax_omega = plt.subplots()
        ax_omega.plot(t, omega_controller_hist, label="Controller", linewidth=line_width)
        ax_omega.plot(t, omega_qp_hist, label="QP", linewidth=line_width)
        ax_omega.set_title("Angular velocity: Controller vs QP", fontsize=title_fontsize)
        ax_omega.set_xlabel("Time [s]", fontsize=label_fontsize)
        ax_omega.set_ylabel("w [rad/s]", fontsize=label_fontsize)
        ax_omega.grid(False)
        ax_omega.tick_params(axis="both", labelsize=tick_fontsize)
        ax_omega.legend(fontsize=legend_fontsize)
        fig_omega.tight_layout()
        fig_omega.savefig(angular_speed_pdf_path, format="pdf", bbox_inches="tight")

        fig_delta, ax_delta = plt.subplots()
        ax_delta.plot(t, delta_controller_hist, label="Controller", linewidth=line_width)
        ax_delta.plot(t, delta_qp_hist, label="QP", linewidth=line_width)
        ax_delta.set_title("Steering angle: Controller vs QP", fontsize=title_fontsize)
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
        Ld = L0 + kv * abs(v)

        v_nom = v_ref
        w_nom = 0.0
        delta_controller = omega_to_delta(w_nom, v_nom, L, v_min=0.2)
        delta_controller = np.clip(delta_controller, -delta_max, delta_max)

        t_ctrl0 = time.perf_counter()
        u_safe, clf_info = qp.cbf_clf_qp_paper_cost(
            u_pre=u_pre,
            robot_state=(x, y, yaw),
            obstacles=obstacles,
            px=px,
            py=py,
            pyaw=pyaw,
            s=s,
            last_path_idx=last_near,
            ellipse_ab=(a_ell, b_ell),
            margin=margin,
            lookahead_l=0.1, #0.1
            alpha=5, #5
            eps_clf=6, #3
            q_clf=(1.0, 10.0, 1), #1 10 0.01
            dt=dt,
            H=(1.0, 1.0),      # paper: 1/2 u^T H u
            Q=(10.0, 10.0),    # paper: (u-u_pre)^T Q (u-u_pre)
            gamma_v=5.0,       # CLF velocidade
            p_v=50000.0,        # peso slack CLF velocidade
            p_slack=10.0, #50
            v_ref=v_ref,
            v_bounds=(0.0, 2.0),
            w_bounds=(-w_max, w_max),
            barrier_inner=inner_bar,   # carregadas uma vez (sem np.loadtxt por passo)
            barrier_outer=outer_bar,
            seg_hint=seg_hint,
        )

        step_time_hist.append(time.perf_counter() - t_ctrl0)
        v_safe, w_safe = u_safe
        last_near = clf_info["idx"]
        cte = clf_info["ey"]
        lat_idx, _ = nearest_path_point_closed(px, py, x, y, last_idx=lat_idx)
        lat_err_hist.append(
            -np.sin(pyaw[lat_idx]) * (x - px[lat_idx])
            + np.cos(pyaw[lat_idx]) * (y - py[lat_idx])
        )
        qp_time_hist.append(clf_info["qp_time"])

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

        v_controller_hist.append(v_nom)
        v_qp_hist.append(v_safe)
        omega_controller_hist.append(w_nom)
        omega_qp_hist.append(w_applied)
        delta_controller_hist.append(delta_controller)
        delta_qp_hist.append(delta)

        x += v_safe * np.cos(yaw) * dt
        y += v_safe * np.sin(yaw) * dt
        yaw = wrap_to_pi(yaw + w_applied * dt)

        hx.append(x)
        hy.append(y)
        ctes.append(cte)

        lap_completed = lap_progress_idx >= (n_path - 1)

        if k % 5 == 0 or lap_completed:
            draw_frame(Ld, v_safe, w_applied, cte)

        if stop_requested:
            draw_frame(Ld, v_safe, w_applied, cte, show_labels=False)
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            print(f"Simulacao encerrada por Enter. Figura guardada em: {pdf_path}")
            plt.ioff()
            plt.close(fig)
            show_velocity_plots()
            return pdf_path

        if lap_completed:
            draw_frame(Ld, v_safe, w_applied, cte, show_labels=False)
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
