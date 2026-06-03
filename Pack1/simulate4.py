# simulate.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import qp
from controller import (
    wrap_to_pi,
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

STRAIGHT_X_MIN = 0.0
STRAIGHT_X_MAX = 120.0
VIEW_HALF_HEIGHT = 2.5
STRAIGHT_DS = 0.01
LEADER_START_X = 8.0
CAR_GAP = 4.0
FOLLOWER_W_BIAS = 0.4

straight_x = np.arange(STRAIGHT_X_MIN, STRAIGHT_X_MAX + STRAIGHT_DS, STRAIGHT_DS)
straight_y = np.zeros_like(straight_x)
straight_yaw = np.zeros_like(straight_x)

cars = []

for index, (name, data) in enumerate(cars_config.items()):
    v_ref_car = data.get("v_ref", 2.0)
    start_x = LEADER_START_X - index * CAR_GAP
    start_idx = int(round((start_x - STRAIGHT_X_MIN) / STRAIGHT_DS))

    cars.append({
        "name": name,
        "x": start_x,
        "y": 0.0,
        "yaw": 0.0,
        "v": 0.0,
        "delta": 0.0,
        "last_near": start_idx,
        "px": straight_x,
        "py": straight_y,
        "pyaw": straight_yaw,
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
    "leader":   {"W": (200.0, 1.0), "alpha": 2.0, "lookahead_l": 0.10},
    "follower": {"W": (200.0, 1.0), "alpha": 2.0, "lookahead_l": 0.20},
}

def simulate():
    obstacles = []

    dt = 0.02
    T = 50.0
    steps = int(T / dt)

    # Pure Pursuit
    L0 = 0.2
    kv = 0.2

    # Limites
    w_max = 2.5
    a_max = 2.0

    # Elipse do robô
    a_ell, b_ell = 0.60, 0.50
    margin = 0.05

    # Parâmetros bicycle/servo
    L = 0.26
    delta_max = np.deg2rad(25)
    delta_rate_max = np.deg2rad(300)

    # históricos por carro
    for car in cars:
        car["hx"], car["hy"], car["ctes"] = [], [], []

    def car_as_obstacle(car, r=0.45):
        return {"x": car["x"], "y": car["y"], "r": r}

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 4))

    for k in range(steps):

        # atualiza todos os carros
        for i, car in enumerate(cars):
            px, py = car["px"], car["py"]

            Ld = L0 + kv * abs(car["v"])
            state = (car["x"], car["y"], car["yaw"], car["v"])

            v_cmd, w_cmd, target_idx, car["last_near"], cte = pure_pursuit_control(
                px, py, state,
                last_near_idx=car["last_near"],
                Ld=Ld,
                v_ref=car["v_ref"],   # <- usa v_ref do YAML
            )

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

                # se quiseres, usa gating para só considerar quando faz sentido
                if not should_consider_other(car, other, d_act=3.0):
                    continue

                dist, forward, lateral = rel_in_car_frame(car, other)

                # obstáculo dinâmico (podes aumentar r, mas com gating não precisa exagerar)
                obs_all.append(car_as_obstacle(other, r=0.25))

                # se há alguém à frente, e é o mais perto, eu viro follower
                if forward > 0.0 and dist < nearest_front_dist:
                    nearest_front_dist = dist
                    role = "follower"

            if role == "follower":
                w_cmd = np.clip(w_cmd + FOLLOWER_W_BIAS, -w_max, w_max)

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
                v_bounds=(0.0, 2.0),
                w_bounds=(-w_max, w_max),
                use_barriers=False,
            )

            # limita w pela física do steering
            kappa_max = np.tan(delta_max) / L
            w_max_speed = abs(v_safe) * kappa_max
            w_safe = np.clip(w_safe, -w_max_speed, w_max_speed)

            delta_cmd = omega_to_delta(w_safe, v_safe, L, v_min=0.2)
            delta_cmd = np.clip(delta_cmd, -delta_max, delta_max)
            car["delta"] = rate_limit(delta_cmd, car["delta"], du_max=delta_rate_max * dt)

            # integra bicycle
            car["x"] += v_safe * np.cos(car["yaw"]) * dt
            car["y"] += v_safe * np.sin(car["yaw"]) * dt
            car["yaw"] = wrap_to_pi(car["yaw"] + (v_safe / L) * np.tan(car["delta"]) * dt)

            # guarda histórico do carro
            car["hx"].append(car["x"])
            car["hy"].append(car["y"])
            car["ctes"].append(cte)

            # guarda info para desenho (por carro)
            car["_plot"] = {
                "Ld": Ld, "target_idx": target_idx,
                "v_safe": v_safe, "w_safe": w_safe, "cte": cte, "role": role
            }

        # DESENHO (a cada 5 passos)
        if k % 5 == 0:
            ax.clear()

            # caminhos + trajetórias + carros
            for car in cars:
                px, py = car["px"], car["py"]
                ax.plot(px, py, "--", linewidth=1, label=f"Reta {car['name']}")

                role = car["_plot"].get("role", "?")
                ax.plot(
                    car["hx"],
                    car["hy"],
                    "-",
                    linewidth=2,
                    label=f"Traj {car['name']} ({role})"
)

                # obstáculos fixos
                for obs in obstacles:
                    ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

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

                # lookahead target
                ti = car["_plot"]["target_idx"]
                ax.plot(px[ti], py[ti], "x", markersize=8)
                ax.add_patch(Circle((car["x"], car["y"]), car["_plot"]["Ld"], fill=False))

                
            ax.set_aspect("equal", "box")
            car_x_values = [car["x"] for car in cars]
            view_x_min = max(STRAIGHT_X_MIN, min(car_x_values) - 2.0)
            view_x_max = min(STRAIGHT_X_MAX, max(car_x_values) + 8.0)
            ax.set_xlim(view_x_min, view_x_max)
            ax.set_ylim(-VIEW_HALF_HEIGHT, VIEW_HALF_HEIGHT)
            ax.grid(True)
            ax.set_title("Multi-carro: Pure Pursuit + CBF-QP em pista reta")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            plt.pause(0.001)

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
