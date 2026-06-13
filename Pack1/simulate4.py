# simulate.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import qp2 as qp
from shapely.geometry import LineString, LinearRing
from shapely.ops import unary_union
from controller import (
    wrap_to_pi,
    build_spline_path,
    pure_pursuit_control,
    omega_to_delta,
    rate_limit,
)
import os
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(BASE_DIR, "paths.yaml")

with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

cars_config = config["cars"]

SIMULATE1_WAYPOINTS = [
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

cars = []

for name, data in cars_config.items():

    waypoints = SIMULATE1_WAYPOINTS
    v_ref_car = data.get("v_ref", 2.0)

    px, py, pyaw, s = build_spline_path(waypoints, ds=0.01)

    start = data.get("start", [waypoints[0][0], waypoints[0][1], 0.0])
    start_idx = int(np.argmin((px - start[0]) ** 2 + (py - start[1]) ** 2))
    start_yaw = float(wrap_to_pi(pyaw[start_idx]))

    cars.append({
        "name": name,
        "x": start[0],
        "y": start[1],
        "yaw": start_yaw,
        "v": 0.0,
        "actual_v": 0.0,
        "actual_w": 0.0,
        "delta": 0.0,
        "last_near": start_idx,
        "px": px,
        "py": py,
        "pyaw": pyaw,
        "v_ref": v_ref_car,
        "color": data.get("color", "black")
    })

def rel_in_car_frame(car, other):
    dx = other["x"] - car["x"]
    dy = other["y"] - car["y"]
    forward = dx*np.cos(car["yaw"]) + dy*np.sin(car["yaw"])
    lateral = -dx*np.sin(car["yaw"]) + dy*np.cos(car["yaw"])
    dist = np.hypot(dx, dy)
    return dist, forward, lateral

def should_consider_other(car, other, d_act=3.0):
    dist, forward, lateral = rel_in_car_frame(car, other)
    if dist > d_act:
        return False
    # só interessa se estiver mais ou menos no "corredor" à frente (evita travar por cruzamentos laterais)
    if forward < 0.0:
        return False
    if abs(lateral) > 1.2:   # ajusta à largura da pista
        return False
    return True

# parâmetros "leader" vs "follower"
QP_PARAMS = {
    "leader":   {"W": (2000.0, 1.0), "alpha": 3.0, "lookahead_l": 0.10},
    "follower": {"W": (2000.0, 1.0), "alpha": 3.0, "lookahead_l": 0.10},
}

def simulate():
    obstacles = []
    if cars:
        px_ref = cars[0]["px"]
        py_ref = cars[0]["py"]
        pyaw_ref = cars[0]["pyaw"]
        n_path = len(px_ref)
        n_obs = 5
        idxs = np.linspace(0, n_path - 1, n_obs + 2, dtype=int)[1:-1]

        for k, idx in enumerate(idxs):
            x_path = px_ref[idx]
            y_path = py_ref[idx]
            yaw_path = pyaw_ref[idx]

            nx = -np.sin(yaw_path)
            ny = np.cos(yaw_path)

            side = (-1) ** k
            offset = 0.3

            ox = x_path + side * offset * nx
            oy = y_path + side * offset * ny

            '''
            obstacles.append({
                "x": ox,
                "y": oy,
                "r": 0.35,
            })
            '''

    dt = 0.02
    T = 120.0
    steps = int(T / dt)

    # Pure Pursuit
    L0 = 1
    kv = 0.0

    # Limites
    w_max = 2.5
    a_max = 2.0 #2

    # Elipse do robô
    a_ell, b_ell = 0.30, 0.20 #0.60, 0.50
    margin = 0.05

    # Parâmetros bicycle/servo
    L = 0.26
    delta_max = np.deg2rad(25)
    delta_rate_max = np.deg2rad(300)

    # carregar barreiras UMA VEZ
    inner_bar = np.loadtxt(os.path.join(BASE_DIR, "barreira_suavizada_interna.txt"))
    outer_bar = np.loadtxt(os.path.join(BASE_DIR, "barreira_suavizada_externa.txt"))
    inner_x, inner_y = inner_bar[:, 0], inner_bar[:, 1]
    outer_x, outer_y = outer_bar[:, 0], outer_bar[:, 1]

    # históricos por carro
    for car in cars:
        car["hx"], car["hy"], car["ctes"] = [], [], []
        car["lap_progress_idx"] = 0.0
        car["prev_near_idx"] = None
        car["finished"] = False
        car["finish_step"] = None

    def car_as_obstacle(car, r=0.45):
        v_obs = car.get("actual_v", car.get("v", 0.0))
        return {
            "x": car["x"],
            "y": car["y"],
            "r": r,
            "vx": v_obs * np.cos(car["yaw"]),
            "vy": v_obs * np.sin(car["yaw"]),
        }

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))
    pdf_save_count = 0

    def next_pdf_path():
        nonlocal pdf_save_count
        while True:
            pdf_save_count += 1
            pdf_path = os.path.join(BASE_DIR, f"simulate4_pista_{pdf_save_count:03d}.pdf")
            if not os.path.exists(pdf_path):
                return pdf_path

    def save_pdf_without_title_legend(pdf_path):
        title = ax.get_title()
        legend = ax.get_legend()
        legend_visible = legend.get_visible() if legend is not None else None
        axis_was_on = ax.axison

        ax.set_title("")
        if legend is not None:
            legend.set_visible(False)
        ax.set_axis_off()

        try:
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        finally:
            ax.set_title(title)
            if legend is not None:
                legend.set_visible(legend_visible)
            if axis_was_on:
                ax.set_axis_on()
            else:
                ax.set_axis_off()
            fig.canvas.draw_idle()

    def on_key_press(event):
        if event.key in ("enter", "return"):
            pdf_path = next_pdf_path()
            save_pdf_without_title_legend(pdf_path)
            print(f"Pista guardada em PDF: {pdf_path}")

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    for k in range(steps):
        cars_snapshot = [
            {
                "x": car["x"],
                "y": car["y"],
                "yaw": car["yaw"],
                "v": car["v"],
                "actual_v": car.get("actual_v", car["v"]),
                "actual_w": car.get("actual_w", 0.0),
                "finished": car["finished"],
            }
            for car in cars
        ]

        # atualiza todos os carros
        for i, car in enumerate(cars):
            px, py = car["px"], car["py"]

            if car["finished"]:
                Ld = L0 + kv * abs(car["v"])
                cte = car["ctes"][-1] if car["ctes"] else 0.0
                target_idx = int(car["last_near"]) % len(px)
                car["v"] = 0.0
                car["delta"] = 0.0
                car["actual_v"] = 0.0
                car["actual_w"] = 0.0
                car["hx"].append(car["x"])
                car["hy"].append(car["y"])
                car["ctes"].append(cte)
                car["_plot"] = {
                    "Ld": Ld, "target_idx": target_idx,
                    "v_safe": 0.0, "w_safe": 0.0, "cte": cte, "role": "finished"
                }
                continue

            Ld = L0 + kv * abs(car["v"])
            state = (car["x"], car["y"], car["yaw"], car["v"])

            v_cmd, w_cmd, target_idx, car["last_near"], cte = pure_pursuit_control(
                px, py, state,
                last_near_idx=car["last_near"],
                Ld=Ld,
                v_ref=car["v_ref"],   # <- usa v_ref do YAML
            )

            n_path = len(px)
            if car["prev_near_idx"] is None:
                car["prev_near_idx"] = car["last_near"]
            else:
                delta_idx = car["last_near"] - car["prev_near_idx"]
                if delta_idx < -n_path / 2:
                    delta_idx += n_path
                elif delta_idx > n_path / 2:
                    delta_idx -= n_path

                car["lap_progress_idx"] += max(0.0, float(delta_idx))
                car["prev_near_idx"] = car["last_near"]

            w_cmd = np.clip(w_cmd, -w_max, w_max)

            dv = np.clip(v_cmd - car["v"], -a_max * dt, a_max * dt)
            car["v"] += dv

            # obstáculos fixos + outros carros
            # --- escolhe o "role" (leader/follower) conforme o vizinho mais relevante ---
            role = "leader"  # default: não reagir muito
            nearest_front_dist = 1e9

            obs_all = list(obstacles)

            for j, other in enumerate(cars):
                if j == i:
                    continue
                other_state = cars_snapshot[j]

                # se quiseres, usa gating para só considerar quando faz sentido
                if not should_consider_other(car, other_state, d_act=3.0):
                    continue

                dist, forward, lateral = rel_in_car_frame(car, other_state)

                # obstáculo dinâmico (podes aumentar r, mas com gating não precisa exagerar)
                obs_all.append(car_as_obstacle(other_state, r=0.45))

                # se há alguém à frente, e é o mais perto, eu viro follower
                if forward > 0.0 and dist < nearest_front_dist:
                    nearest_front_dist = dist
                    role = "follower"

            p = QP_PARAMS[role]

            v_safe, w_safe = qp.cbf_qp_filter(
                u_nom=(v_cmd, w_cmd),
                robot_state=(car["x"], car["y"], car["yaw"]),
                obstacles=obs_all,
                ellipse_ab=(a_ell, b_ell),
                margin=margin,
                lookahead_l=p["lookahead_l"],
                alpha=p["alpha"],
                W=p["W"],
                v_bounds=(0.0, 10.0),
                w_bounds=(-w_max, w_max),
            )

            # limita w pela física do steering
            kappa_max = np.tan(delta_max) / L
            w_max_speed = abs(v_safe) * kappa_max
            w_safe = np.clip(w_safe, -w_max_speed, w_max_speed)

            delta_cmd = omega_to_delta(w_safe, v_safe, L, v_min=0.2)
            delta_cmd = np.clip(delta_cmd, -delta_max, delta_max)
            car["delta"] = rate_limit(delta_cmd, car["delta"], du_max=delta_rate_max * dt)
            w_applied = (v_safe / L) * np.tan(car["delta"])

            car["v"] = v_safe
            car["actual_v"] = v_safe
            car["actual_w"] = w_applied

            # integra bicycle
            car["x"] += v_safe * np.cos(car["yaw"]) * dt
            car["y"] += v_safe * np.sin(car["yaw"]) * dt
            car["yaw"] = wrap_to_pi(car["yaw"] + w_applied * dt)

            if car["lap_progress_idx"] >= (n_path - 1):
                car["finished"] = True
                car["finish_step"] = k
                car["v"] = 0.0
                car["delta"] = 0.0
                car["actual_v"] = 0.0
                car["actual_w"] = 0.0
                v_safe = 0.0
                w_safe = 0.0
                role = "finished"
                print(f"{car['name']} chegou ao fim da pista e ficou parado.")

            # guarda histórico do carro
            car["hx"].append(car["x"])
            car["hy"].append(car["y"])
            car["ctes"].append(cte)

            # guarda info para desenho (por carro)
            car["_plot"] = {
                "Ld": Ld, "target_idx": target_idx,
                "v_safe": v_safe, "w_safe": w_safe, "cte": cte, "role": role
            }

        all_finished = bool(cars) and all(car["finished"] for car in cars)

        # DESENHO (a cada 5 passos)
        if k % 5 == 0 or all_finished:
            ax.clear()

            # barreiras
            ax.plot(inner_x, inner_y, "-", color="green", linewidth=2, label="Barreira interna")
            ax.plot(outer_x, outer_y, "-", color="green", linewidth=2, label="Barreira externa")

            for obs in obstacles:
                ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

            # caminhos + trajetórias + carros
            for car in cars:
                px, py = car["px"], car["py"]
                ax.plot(px, py, "--", linewidth=1, label=f"Spline {car['name']}")

                role = car["_plot"].get("role", "?")
                ax.plot(
                    car["hx"],
                    car["hy"],
                    "-",
                    linewidth=2,
                    label=f"Traj {car['name']} ({role})"
)

                # corpo do carro (elipse)
                ell = Ellipse((car["x"], car["y"]),
                              width=2*a_ell, height=2*b_ell,
                              angle=np.degrees(car["yaw"]),
                              fill=False)
                ax.add_patch(ell)

                ell_safe = Ellipse((car["x"], car["y"]),
                                   width=2*(a_ell+margin), height=2*(b_ell+margin),
                                   angle=np.degrees(car["yaw"]),
                                   fill=False)
                ax.add_patch(ell_safe)

                # heading
                ax.plot(car["x"], car["y"], "o")
                ax.arrow(car["x"], car["y"],
                         0.4*np.cos(car["yaw"]), 0.4*np.sin(car["yaw"]),
                         head_width=0.15)

            ax.set_aspect("equal", "box")
            ax.grid(False)
            ax.set_title("Multi-carro: Pure Pursuit + CBF-QP (caminho do simulate1)")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            plt.pause(0.001)

        if all_finished:
            print("Todos os carros chegaram ao fim da pista. Simulação encerrada.")
            break

    plt.ioff()

    # cte por carro
    fig2, ax2 = plt.subplots()
    for car in cars:
        ax2.plot(car["ctes"], label=car["name"])
    ax2.set_title("Erro lateral (aprox) - Pure Pursuit")
    ax2.set_xlabel("Passo")
    ax2.set_ylabel("cte~ [m]")
    ax2.grid(True)
    ax2.legend()
    plt.show()

if __name__ == "__main__":
    simulate()
