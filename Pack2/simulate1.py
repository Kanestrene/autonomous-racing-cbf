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

# --- Barreiras (carregar uma vez) ---
inner_bar = np.loadtxt("barreira_suavizada_interna.txt")        # ou o nome que usaste
outer_bar = np.loadtxt("barreira_suavizada_externa.txt")       # ou o nome que usaste

def simulate():
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

    # topo esquerdo mais progressivo
    (2.2, 13.6),
    (2.6, 14.5),
    (3.1, 14.8),
    (3.7, 15.0),
    (4.2, 15.0),

    # curva do topo (em vez de mudanças curtas em x)
    (4.8, 14.9),
    (5.3, 14.6),
    (5.6, 14.1),
    (5.7, 13.5),

    # descida suavizada (mantém “coluna” mas com entrada/saída mais suave)
    (5.6, 12.6),
    (5.5, 11.6),
    (5.5, 10.6),
    (5.5, 9.8),

    # curva embaixo (antes tinha um “cotovelo”)
    (5.7, 9.2),
    (6.0, 8.7),
    (6.6, 8.4),
    (7.4, 8.4),
    (8.2, 8.5),

    # subida para o miolo direito, mais gradual
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
    
    # -----------------------------
    # Gerar 5 obstáculos ao longo da trajetória
    # -----------------------------
    n_obs = 5
    idxs = np.linspace(0, len(px)-1, n_obs+2, dtype=int)[1:-1]

    obstacles = []

    
    
    for k, idx in enumerate(idxs):
        x_path = px[idx]
        y_path = py[idx]
        yaw_path = pyaw[idx]

        # vetor normal à trajetória
        nx = -np.sin(yaw_path)
        ny =  np.cos(yaw_path)

        # alterna lado esquerdo/direito
        side = (-1)**k
        offset = 0.3   # distância lateral ao centro da pista

        ox = x_path + side * offset * nx
        oy = y_path + side * offset * ny

        obstacles.append({
            "x": ox,
            "y": oy,
            "r": 0.35
        })

    
    
    # Estado inicial
    x, y, yaw, v = 3, 3, np.deg2rad(90), 0.0

    dt = 0.02
    T = 50.0
    steps = int(T / dt)

    # Pure Pursuit
    v_ref = 2
    L0 = 0.1
    kv = 0.5

    # Limites
    w_max = 2.5
    a_max = 2.0

    # Obstáculos
    
    # Elipse do robô
    a_ell, b_ell = 0.30, 0.20   # semi-eixos [m]
    margin = 0.05              # margem CBF/desenho

    last_near = 0
    hx, hy, ctes = [], [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))

    # Steering state
    delta = 0.0

    # Parâmetros bicycle/servo
    L = 0.26                         # entre-eixos (m)
    delta_max = np.deg2rad(25)       # limite do servo
    delta_rate_max = np.deg2rad(300) # rad/s

    for k in range(steps):
        # lookahead adaptativo
        Ld = L0 + kv * abs(v)

        state = (x, y, yaw, v)

        # nominal simples
        v_nom = v_ref
        w_nom = 0.0

        #v_nom = v_ref
        #w_nom = v_ref * clf_info["kappa_r"]
        '''
        d_min = min(
        np.hypot(x - obs["x"], y - obs["y"]) - obs["r"]
        for obs in obstacles
        )

        if d_min < 1.5:
            Wv = 10000.0
        else:
            Wv = 200.0
        '''
        (u_safe, clf_info) = qp.cbf_clf_qp_filter(
            u_nom=(v_nom, w_nom),
            robot_state=(x, y, yaw),
            obstacles=obstacles,
            px=px, py=py, pyaw=pyaw, s=s,
            last_path_idx=last_near,
            ellipse_ab=(a_ell, b_ell),
            margin=margin,
            lookahead_l=0.01,
            alpha=3,
            eps_clf=0.5,
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

        # Converte w -> delta com limites físicos dependentes da velocidade
        kappa_max = np.tan(delta_max) / L
        w_max_speed = abs(v_safe) * kappa_max
        w_safe = np.clip(w_safe, -w_max_speed, w_max_speed)

        delta_cmd = omega_to_delta(w_safe, v_safe, L, v_min=0.2)
        delta_cmd = np.clip(delta_cmd, -delta_max, delta_max)

        # limita taxa do servo
        delta = rate_limit(delta_cmd, delta, du_max=delta_rate_max * dt)

        # integra bicycle cinemático
        x += v_safe * np.cos(yaw) * dt
        y += v_safe * np.sin(yaw) * dt
        yaw = wrap_to_pi(yaw + (v_safe / L) * np.tan(delta) * dt)

        hx.append(x)
        hy.append(y)
        ctes.append(cte)
        
        '''
        if target_idx >= len(px) - 5 and np.hypot(px[-1] - x, py[-1] - y) < 0.2:
            break
        '''
        
        inner_x, inner_y = inner_bar[:, 0], inner_bar[:, 1]
        outer_x, outer_y = outer_bar[:, 0], outer_bar[:, 1]

        # desenho
        if k % 5 == 0:
            ax.clear()

            ax.plot(px, py, "--", label="Spline (referência)")
            ax.plot(hx, hy, "-", label="Trajetória robô (PP + CBF)")

            # Barreiras
            ax.plot(inner_x, inner_y, "-", linewidth=2, label="Barreira interna")
            ax.plot(outer_x, outer_y, "-", linewidth=2, label="Barreira externa")

            for obs in obstacles:
                ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

            ell = Ellipse((x, y), width=2*a_ell, height=2*b_ell,
                          angle=np.degrees(yaw), fill=False)
            ax.add_patch(ell)

            ell_safe = Ellipse((x, y), width=2*(a_ell+margin), height=2*(b_ell+margin),
                               angle=np.degrees(yaw), fill=False)
            ax.add_patch(ell_safe)

            ax.plot(x, y, "o", label="Robô")
            ax.arrow(x, y, 0.4*np.cos(yaw), 0.4*np.sin(yaw), head_width=0.15)

            #ax.plot(px[target_idx], py[target_idx], "x", markersize=10, label="Alvo (lookahead)")
            ax.add_patch(Circle((x, y), Ld, fill=False))
           
            ax.set_aspect("equal", "box")
            ax.grid(True)
            ax.set_title(f"CLF + CBF-QP | Ld={Ld:.2f} | v={v_safe:.2f} | w={w_safe:.2f} | cte~{cte:.3f}")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            plt.pause(0.001)

    plt.ioff()

    fig2, ax2 = plt.subplots()
    ax2.plot(ctes)
    ax2.set_title("Erro lateral (aprox) - CLF")
    ax2.set_xlabel("Passo")
    ax2.set_ylabel("cte~ [m]")
    ax2.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate()