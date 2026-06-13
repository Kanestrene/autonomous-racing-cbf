# simulate.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import qp
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

cars = []

for name, data in cars_config.items():

    waypoints = data["waypoints"]
    v_ref_car = data.get("v_ref", 2.0)

    px, py, pyaw, s = build_spline_path(waypoints, ds=0.01)

    start = data.get("start", [waypoints[0][0], waypoints[0][1], 0.0])

    cars.append({
        "name": name,
        "x": start[0],
        "y": start[1],
        "yaw": start[2],
        "v": 0.0,
        "delta": 0.0,
        "last_near": 0,
        "px": px,
        "py": py,
        "pyaw": pyaw,
        "s": s,
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
    "leader":   {"W": (10000.0, 1.0), "alpha": 3.0, "lookahead_l": 0.10},
    "follower": {"W": ( 100.0, 1.0), "alpha": 2.0, "lookahead_l": 0.30},
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

    # carregar barreiras UMA VEZ
    inner_bar = np.loadtxt("barreira_suavizada_interna.txt")
    outer_bar = np.loadtxt("barreira_suavizada_externa.txt")
    inner_x, inner_y = inner_bar[:, 0], inner_bar[:, 1]
    outer_x, outer_y = outer_bar[:, 0], outer_bar[:, 1]

    # históricos por carro
    for car in cars:
        car["hx"], car["hy"], car["ctes"] = [], [], []

    def car_as_obstacle(car, r=0.45):
        return {"x": car["x"], "y": car["y"], "r": r}

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))

    for k in range(steps):

        # atualiza todos os carros
        for i, car in enumerate(cars):
            px, py, pyaw, s = car["px"], car["py"], car["pyaw"], car["s"]

            Ld = L0 + kv * abs(car["v"])
                
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

            p = QP_PARAMS[role]
            
            # nominal simples
            v_nom = car["v_ref"]
            w_nom = 1.0

            #v_nom = v_ref
            #w_nom = v_ref * clf_info["kappa_r"]

            (u_safe, clf_info) = qp.cbf_clf_qp_filter(
                u_nom=(v_nom, w_nom),
                robot_state=(car["x"], car["y"], car["yaw"]),
                obstacles=obs_all,
                px=px, py=py, pyaw=pyaw, s=s,
                last_path_idx=car["last_near"],
                ellipse_ab=(a_ell, b_ell),
                margin=margin,
                lookahead_l=p["lookahead_l"],
                alpha=p["alpha"],
                eps_clf=1.5,
                W=p["W"],
                p_slack=500.0,
                v_ref=car["v_ref"],
                v_bounds=(0.0, 2.0),
                w_bounds=(-w_max, w_max),
            )

            v_safe, w_safe = u_safe
            car["last_near"] = clf_info["idx"]
            cte = clf_info["ey"]
            car["v"] = v_safe

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
                "Ld": Ld, "v_safe": v_safe, "w_safe": w_safe, "cte": cte, "role": role
            }

        # DESENHO (a cada 5 passos)
        if k % 5 == 0:
            ax.clear()

            # barreiras
            ax.plot(inner_x, inner_y, "-", linewidth=2, label="Barreira interna")
            ax.plot(outer_x, outer_y, "-", linewidth=2, label="Barreira externa")

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

                '''
                # lookahead target
                ti = car["_plot"]["target_idx"]
                ax.plot(px[ti], py[ti], "x", markersize=8)
                ax.add_patch(Circle((car["x"], car["y"]), car["_plot"]["Ld"], fill=False))
                '''
                
            ax.set_aspect("equal", "box")
            ax.grid(True)
            ax.set_title("Multi-carro: CLF + CBF-QP (cada um com caminho do YAML)")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            plt.pause(0.001)

    plt.ioff()

    # cte por carro
    fig2, ax2 = plt.subplots()
    for car in cars:
        ax2.plot(car["ctes"], label=car["name"])
    ax2.set_title("Erro lateral (aprox) - CLF")
    ax2.set_xlabel("Passo")
    ax2.set_ylabel("cte~ [m]")
    ax2.grid(True)
    ax2.legend()
    plt.show()

if __name__ == "__main__":
    simulate()