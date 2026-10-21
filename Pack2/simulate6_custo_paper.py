import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from pathlib import Path
from scipy.spatial import cKDTree
from contourpy import contour_generator

import qp
from clf import nearest_path_point_closed

from controller import (
    wrap_to_pi,
    build_spline_path,
    omega_to_delta,
    rate_limit,
)


OUT_DIR = Path(__file__).resolve().parent / "figuras" / "simulate6_custo_paper"

# Pista em 8 (simulate6), no mesmo retangulo da pista do simulate1:
# dois lobulos (arcos R=2.6 m + retas verticais) ligados por diagonais que se cruzam
# no centro (9, 8.5) com ~45 graus. Lobulo esquerdo no sentido horario, direito no anti-horario.
# Raio de curvatura minimo da spline ~2.3 m (> TRACK_HALF_WIDTH).
# Comeca no lobulo esquerdo, logo antes da posicao inicial do carro (2, 6): a volta conta
# a partir do indice 0 do caminho.
WAYPOINTS = [
    (2.000, 5.423),
    (2.000, 6.196),
    (2.000, 6.969),
    (2.000, 7.742),
    (2.000, 8.515),
    (2.000, 9.288),
    (2.000, 10.061),
    (2.000, 10.834),
    (2.000, 11.607),
    (2.000, 12.380),
    (2.121, 13.163),
    (2.471, 13.873),
    (3.019, 14.444),
    (3.714, 14.824),
    (4.490, 14.978),
    (5.277, 14.890),
    (6.002, 14.570),
    (6.596, 14.047),
    (7.005, 13.369),
    (7.312, 12.622),
    (7.619, 11.875),
    (7.926, 11.128),
    (8.233, 10.382),
    (8.540, 9.635),
    (8.847, 8.888),
    (9.153, 8.142),
    (9.460, 7.395),
    (9.767, 6.648),
    (10.074, 5.902),
    (10.381, 5.155),
    (10.688, 4.408),
    (10.995, 3.661),
    (11.404, 2.983),
    (11.998, 2.460),
    (12.723, 2.140),
    (13.510, 2.052),
    (14.286, 2.206),
    (14.981, 2.586),
    (15.529, 3.157),
    (15.879, 3.867),
    (16.000, 4.650),
    (16.000, 5.423),
    (16.000, 6.196),
    (16.000, 6.969),
    (16.000, 7.742),
    (16.000, 8.515),
    (16.000, 9.288),
    (16.000, 10.061),
    (16.000, 10.834),
    (16.000, 11.607),
    (16.000, 12.380),
    (15.879, 13.163),
    (15.529, 13.873),
    (14.981, 14.444),
    (14.286, 14.824),
    (13.510, 14.978),
    (12.723, 14.890),
    (11.998, 14.570),
    (11.404, 14.047),
    (10.995, 13.369),
    (10.710, 12.675),
    (10.425, 11.982),
    (10.140, 11.288),
    (9.855, 10.595),
    (9.570, 9.902),
    (9.285, 9.208),
    (9.000, 8.515),
    (8.715, 7.822),
    (8.430, 7.128),
    (8.145, 6.435),
    (7.860, 5.742),
    (7.575, 5.048),
    (7.290, 4.355),
    (7.005, 3.661),
    (6.596, 2.983),
    (6.002, 2.460),
    (5.277, 2.140),
    (4.490, 2.052),
    (3.714, 2.206),
    (3.019, 2.586),
    (2.471, 3.157),
    (2.121, 3.867),
    (2.000, 4.650),
    (2.000, 5.423),
]

# meia-largura da pista (distancia do caminho a cada barreira), igual a pista do simulate1
TRACK_HALF_WIDTH = 1.5


def build_track_barriers(px, py, pyaw, half_width=TRACK_HALF_WIDTH, grid=0.02):
    """
    Limites da pista = curva de nivel a half_width do caminho (funciona com cruzamentos).
    Como o caminho se cruza, os offsets +-normal atravessariam o cruzamento; aqui
    calcula-se a distancia ao caminho numa grelha e extrai-se o contorno a half_width.
    Retorna (ilhas, exterior): lista de polignos interiores (2 no 8) e o contorno exterior.
    """
    pad = half_width + 0.5
    gx = np.arange(px.min() - pad, px.max() + pad, grid)
    gy = np.arange(py.min() - pad, py.max() + pad, grid)
    GX, GY = np.meshgrid(gx, gy)
    dist, _ = cKDTree(np.column_stack([px, py])).query(np.column_stack([GX.ravel(), GY.ravel()]))
    lines = contour_generator(gx, gy, dist.reshape(GX.shape)).lines(half_width)

    def area(poly):
        return 0.5 * abs(np.dot(poly[:-1, 0], poly[1:, 1]) - np.dot(poly[1:, 0], poly[:-1, 1]))

    # o exterior e o contorno que envolve maior area; os restantes sao ilhas
    lines = sorted(lines, key=area, reverse=True)
    return lines[1:], lines[0]


def save_track_only():
    pdf_path = OUT_DIR / "simulate6_custo_paper_pista.pdf"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    px, py, pyaw, _ = build_spline_path(WAYPOINTS, ds=0.01)
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

    inner_bar, outer_bar = build_track_barriers(px, py, pyaw)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(px, py, "--")
    for island in inner_bar:
        ax.plot(island[:, 0], island[:, 1], "-", linewidth=2, color="green")
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


def simulate(animate=True):
    pdf_path = OUT_DIR / "simulate6_custo_paper_volta_completa.pdf"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    linear_speed_pdf_path = OUT_DIR / "simulate6_custo_paper_velocidade_linear.pdf"
    angular_speed_pdf_path = OUT_DIR / "simulate6_custo_paper_velocidade_angular.pdf"
    delta_pdf_path = OUT_DIR / "simulate6_custo_paper_delta.pdf"
    cpu_pdf_path = OUT_DIR / "simulate6_custo_paper_tempo_cpu.pdf"
    qp_pdf_path = OUT_DIR / "simulate6_custo_paper_tempo_qp.pdf"
    lat_pdf_path = OUT_DIR / "simulate6_custo_paper_erro_lateral.pdf"

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

    def on_key_press(event):
        nonlocal stop_requested
        if event.key in ("enter", "return"):
            stop_requested = True

    if animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("white")
        fig.canvas.mpl_connect("key_press_event", on_key_press)

    delta = 0.0

    L = 0.26
    delta_max = np.deg2rad(25)
    delta_rate_max = np.deg2rad(300)

    inner_bar, outer_bar = build_track_barriers(px, py, pyaw)
    outer_x, outer_y = outer_bar[:, 0], outer_bar[:, 1]

    def draw_frame(Ld, v_safe, w_safe, cte, show_labels=True):
        ax.clear()
        ax.set_facecolor("white")

        ax.plot(px, py, "--", label="Spline (referencia)")
        ax.plot(hx, hy, "-", label="Trajetoria robo (CLF + CBF)")

        for i, island in enumerate(inner_bar):
            ax.plot(island[:, 0], island[:, 1], "-", color = "green", linewidth=2,
                    label="Barreira interna" if i == 0 else None)
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

        box_kw = dict(
            widths=0.5, showmeans=False, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor="lightsteelblue", edgecolor="C0", linewidth=1.5),
            whiskerprops=dict(color="C0", linewidth=1.5),
            capprops=dict(color="C0", linewidth=1.5),
            medianprops=dict(color="C0", linewidth=line_width),
        )

        fig_cpu, ax_cpu = plt.subplots()
        bp_cpu = ax_cpu.boxplot(step_ms, **box_kw)
        ax_cpu.set_xticks([1])
        ax_cpu.set_xticklabels(["Total time"])
        ax_cpu.set_title("Total time per step", fontsize=title_fontsize)
        ax_cpu.set_ylabel("Time [ms]", fontsize=label_fontsize)
        ax_cpu.grid(False)
        ax_cpu.tick_params(axis="both", labelsize=tick_fontsize)
        ax_cpu.legend(
            [bp_cpu["medians"][0]],
            [f"Median = {np.median(step_ms):.2f} ms"],
            fontsize=legend_fontsize,
        )
        fig_cpu.tight_layout()
        fig_cpu.savefig(cpu_pdf_path, format="pdf", bbox_inches="tight")

        fig_qp, ax_qp = plt.subplots()
        bp_qp = ax_qp.boxplot(qp_ms, **box_kw)
        ax_qp.set_xticks([1])
        ax_qp.set_xticklabels(["solve_qp"])
        ax_qp.set_title("QP solve time per step", fontsize=title_fontsize)
        ax_qp.set_ylabel("QP time [ms]", fontsize=label_fontsize)
        ax_qp.grid(False)
        ax_qp.tick_params(axis="both", labelsize=tick_fontsize)
        ax_qp.legend(
            [bp_qp["medians"][0]],
            [f"Median = {np.median(qp_ms):.3f} ms"],
            fontsize=legend_fontsize,
        )
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

    def collect_results():
        return {
            "hx": np.array(hx),
            "hy": np.array(hy),
            "v_nom": np.array(v_controller_hist),
            "v_qp": np.array(v_qp_hist),
            "w_nom": np.array(omega_controller_hist),
            "w_qp": np.array(omega_qp_hist),
            "delta_nom": np.array(delta_controller_hist),
            "delta_qp": np.array(delta_qp_hist),
            "step_ms": np.array(step_time_hist) * 1e3,
            "qp_ms": np.array(qp_time_hist) * 1e3,
            "lat": np.array(lat_err_hist),
            "dt": dt,
            "lap_completed": lap_progress_idx >= (n_path - 1),
            "px": px,
            "py": py,
            "obstacles": obstacles,
            "inner_bar": inner_bar,
            "outer_bar": outer_bar,
        }

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
            barrier_inner=inner_bar,   # lista com as 2 ilhas do 8 (calculadas uma vez)
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

        if animate and (k % 5 == 0 or lap_completed):
            draw_frame(Ld, v_safe, w_applied, cte)

        if stop_requested:
            draw_frame(Ld, v_safe, w_applied, cte, show_labels=False)
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            print(f"Simulacao encerrada por Enter. Figura guardada em: {pdf_path}")
            plt.ioff()
            plt.close(fig)
            show_velocity_plots()
            return collect_results()

        if lap_completed:
            if animate:
                draw_frame(Ld, v_safe, w_applied, cte, show_labels=False)
                fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
                print(f"Volta completa. Figura guardada em: {pdf_path}")
                plt.ioff()
                plt.close(fig)
                show_velocity_plots()
            return collect_results()

    print("A simulacao terminou por tempo maximo sem completar uma volta.")
    if animate:
        plt.ioff()
        plt.close(fig)
        show_velocity_plots()
    return collect_results()


if __name__ == "__main__":
    simulate()
