import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from pathlib import Path

import qp
from ml_cbf_experiment import (
    DEFAULT_OBSTACLE_PATH_OFFSET_M,
    build_problem,
    min_barrier_clearance,
    min_obstacle_clearance,
    score_episode,
    shift_path_indices,
    start_state_for_variation,
)

from controller import (
    wrap_to_pi,
    build_spline_path,
    omega_to_delta,
    rate_limit,
)


def save_track_only(obstacle_path_offset_m=DEFAULT_OBSTACLE_PATH_OFFSET_M):
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
        (8.4, 2.0),
        (7.2, 2.0),
        (6.0, 2.0),
        (4.0, 2.5),
        (3.0, 3.0),
    ]

    px, py, pyaw, _ = build_spline_path(waypoints, ds=0.01)
    n_path = len(px)

    n_obs = 5
    idxs = np.linspace(0, n_path - 1, n_obs + 2, dtype=int)[1:-1]
    idxs = shift_path_indices(px, py, idxs, obstacle_path_offset_m)
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
    variation_id = 15
    obstacle_path_offset_m = DEFAULT_OBSTACLE_PATH_OFFSET_M

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
        (8.4, 2.0),
        (7.2, 2.0),
        (6.0, 2.0),
        (4.0, 2.5),
        (3.0, 3.0),
    ]

    px, py, pyaw, s, obstacles, inner_bar, outer_bar = build_problem(
        variation_id=variation_id,
        obstacle_path_offset_m=obstacle_path_offset_m,
    )
    n_path = len(px)

    x, y, yaw, v = start_state_for_variation(variation_id)

    dt = 0.02
    T = 50.0
    steps = int(T / dt)

    v_ref = 2
    L0 = 0.1
    kv = 0.5

    w_max = 2.5

    a_ell, b_ell = 0.30, 0.20
    margin = 0.05
    class_k_p = 7 #3 #6.76 #5.98
    class_k_q = 1 #1 #1.03 #1.13
    alpha = 3.0
    params_suffix = f"p{class_k_p:g}_q{class_k_q:g}"
    pdf_path = script_dir / f"simulate1_volta_completa_{params_suffix}.pdf"
    linear_speed_pdf_path = script_dir / f"simulate1_velocidade_linear_clf_qp_{params_suffix}.pdf"
    cte_pdf_path = script_dir / f"simulate1_erro_lateral_{params_suffix}.pdf"

    last_near = 0
    hx, hy, ctes = [], [], []
    v_safe_values = []
    lap_progress_idx = 0.0
    prev_near_idx = None
    qp_failures = 0
    min_obs_clear = np.inf
    min_bar_clear = np.inf
    collided = False
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

        ax.plot(px, py, "--", label="Reference spline")
        ax.plot(hx, hy, "-", label="Robot trajectory (CLF + CBF)")

        ax.plot(inner_x, inner_y, "-", color = "green", linewidth=2, label="Inner barrier")
        ax.plot(outer_x, outer_y, "-", color = "green", linewidth=2, label="Outer barrier")

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

        ax.plot(x, y, "o", label="Robot")
        ax.arrow(x, y, 0.4 * np.cos(yaw), 0.4 * np.sin(yaw), head_width=0.15)

        ax.set_aspect("equal", "box")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if show_labels:
            ax.set_title(
                f"CLF + CBF-QP | Ld={Ld:.2f} | v={v_safe:.2f} | "
                f"w={w_safe:.2f} | cte~{cte:.3f} | "
                f"p={class_k_p:.2f}, q={class_k_q:.2f}"
            )
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.pause(0.001)

    def print_speed_metrics():
        if v_safe_values:
            mean_v = float(np.mean(np.asarray(v_safe_values, dtype=float)))
        else:
            mean_v = 0.0

        if len(v_safe_values) > 1:
            abs_dv = np.abs(np.diff(np.asarray(v_safe_values, dtype=float)))
            mean_abs_dv = float(np.mean(abs_dv))
            total_abs_dv = float(np.sum(abs_dv))
        else:
            mean_abs_dv = 0.0
            total_abs_dv = 0.0

        print(f"Velocidade media: {mean_v:.4f} m/s")
        print(f"mean_abs_dv: {mean_abs_dv:.6f}")
        print(f"total_abs_dv: {total_abs_dv:.6f}")

    def build_score_metrics(completed):
        if len(v_safe_values) > 1:
            abs_dv = np.abs(np.diff(np.asarray(v_safe_values, dtype=float)))
            mean_abs_dv = float(np.mean(abs_dv))
            total_abs_dv = float(np.sum(abs_dv))
        else:
            mean_abs_dv = 0.0
            total_abs_dv = 0.0

        progress_ratio = min(1.0, lap_progress_idx / max(1.0, float(n_path - 1)))
        metrics = {
            "completed": bool(completed),
            "collided": bool(collided),
            "progress_ratio": float(progress_ratio),
            "qp_failures": int(qp_failures),
            "mean_abs_cte": float(np.mean(ctes)) if ctes else np.inf,
            "max_abs_cte": float(np.max(ctes)) if ctes else np.inf,
            "mean_abs_dv": mean_abs_dv,
            "total_abs_dv": total_abs_dv,
            "min_obstacle_clearance": float(min_obs_clear),
            "min_barrier_clearance": float(min_bar_clear),
        }
        metrics["D_min"] = float(min(min_obs_clear, min_bar_clear))
        metrics["score"] = score_episode(metrics)
        return metrics

    def print_score_summary(completed):
        metrics = build_score_metrics(completed)
        print(
            "Score final: "
            f"{metrics['score']:.6f} | "
            f"D_min={metrics['D_min']:.6f} | "
            f"progress={metrics['progress_ratio']:.4f} | "
            f"mean_abs_dv={metrics['mean_abs_dv']:.6f} | "
            f"total_abs_dv={metrics['total_abs_dv']:.6f} | "
            f"qp_failures={metrics['qp_failures']} | "
            f"completed={metrics['completed']} | "
            f"collided={metrics['collided']}"
        )

    def save_final_plots():
        saved_paths = []
        line_width = 2.2
        title_fontsize = 14
        label_fontsize = 13
        tick_fontsize = 11
        legend_fontsize = 12

        if v_safe_values:
            t = np.arange(len(v_safe_values)) * dt
            fig_v, ax_v = plt.subplots()
            ax_v.plot(t, np.full(len(v_safe_values), v_ref), label="Controller", linewidth=line_width)
            ax_v.plot(t, v_safe_values, label="QP", linewidth=line_width)
            ax_v.set_title("Linear velocity: Controller vs QP", fontsize=title_fontsize)
            ax_v.set_xlabel("Time [s]", fontsize=label_fontsize)
            ax_v.set_ylabel("v [m/s]", fontsize=label_fontsize)
            ax_v.grid(False)
            ax_v.tick_params(axis="both", labelsize=tick_fontsize)
            ax_v.legend(fontsize=legend_fontsize)
            fig_v.tight_layout()
            fig_v.savefig(linear_speed_pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig_v)
            saved_paths.append(linear_speed_pdf_path)

        if ctes:
            t = np.arange(len(ctes)) * dt
            fig_cte, ax_cte = plt.subplots()
            ax_cte.plot(t, ctes, linewidth=line_width)
            ax_cte.set_title("Lateral track error", fontsize=title_fontsize)
            ax_cte.set_xlabel("Time [s]", fontsize=label_fontsize)
            ax_cte.set_ylabel("Lateral error [m]", fontsize=label_fontsize)
            ax_cte.grid(False)
            ax_cte.tick_params(axis="both", labelsize=tick_fontsize)
            fig_cte.tight_layout()
            fig_cte.savefig(cte_pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig_cte)
            saved_paths.append(cte_pdf_path)

        if saved_paths:
            print("Graficos guardados em: " + ", ".join(str(path) for path in saved_paths))

    def finalize_simulation(message, completed, Ld, v_safe, w_safe, cte):
        draw_frame(Ld, v_safe, w_safe, cte, show_labels=False)
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        print(f"{message} Figura guardada em: {pdf_path}")
        print_speed_metrics()
        print_score_summary(completed=completed)
        save_final_plots()
        plt.ioff()
        plt.close(fig)
        return pdf_path
    
    w_safe = 0

    for k in range(steps):
        Ld = L0 + kv * abs(v)

        v_nom = v_ref
        w_nom = 0.0

        u_safe, clf_info = qp.cbf_clf_qp_filter(
            u_nom=(v_nom, np.sign(w_safe)*2),
            robot_state=(x, y, yaw),
            obstacles=obstacles,
            px=px,
            py=py,
            pyaw=pyaw,
            s=s,
            last_path_idx=last_near,
            ellipse_ab=(a_ell, b_ell),
            margin=margin,
            lookahead_l=0.2,
            barrier_lookahead_l=0.1,
            alpha=alpha,
            class_k_p=class_k_p,
            class_k_q=class_k_q,
            eps_clf=2,
            q_clf=(1.0, 10.0, 0.01),
            W=(100000.0, 1.0),
            p_slack=50.0,
            v_ref=v_ref,
            v_bounds=(0.0, 2.0),
            w_bounds=(-w_max, w_max),
        )

        if clf_info.get("qp_failed", False):
            qp_failures += 1

        v_safe, w_safe = u_safe
        v_safe_values.append(float(v_safe))
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
        v = float(v_safe)

        obs_clear = min_obstacle_clearance(x, y, obstacles, (a_ell, b_ell), margin)
        bar_clear = min_barrier_clearance(x, y, inner_bar, outer_bar, (a_ell, b_ell), margin)
        min_obs_clear = min(min_obs_clear, obs_clear)
        min_bar_clear = min(min_bar_clear, bar_clear)
        if min(obs_clear, bar_clear) < -0.35:
            collided = True

        hx.append(x)
        hy.append(y)
        ctes.append(abs(float(cte)))

        lap_completed = lap_progress_idx >= (n_path - 1)

        if k % 5 == 0 or lap_completed:
            draw_frame(Ld, v_safe, w_safe, cte)

        if stop_requested:
            return finalize_simulation(
                "Simulacao encerrada por Enter.",
                completed=False,
                Ld=Ld,
                v_safe=v_safe,
                w_safe=w_safe,
                cte=cte,
            )

        if lap_completed:
            return finalize_simulation(
                "Volta completa.",
                completed=True,
                Ld=Ld,
                v_safe=v_safe,
                w_safe=w_safe,
                cte=cte,
            )

        if collided:
            return finalize_simulation(
                "Simulacao terminou por colisao.",
                completed=False,
                Ld=Ld,
                v_safe=v_safe,
                w_safe=w_safe,
                cte=cte,
            )

    return finalize_simulation(
        "A simulacao terminou por tempo maximo sem completar uma volta.",
        completed=False,
        Ld=Ld,
        v_safe=v_safe,
        w_safe=w_safe,
        cte=cte,
    )


if __name__ == "__main__":
    simulate()
