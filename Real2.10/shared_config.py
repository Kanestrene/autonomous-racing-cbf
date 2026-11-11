'''
TRACK_POINTS_M = [
    (0.7, 0.1),
    (-0.7, 0.1),
    (-0.98, 0.4),
    (-0.92, 1.35),
    (-0.65, 1.5),
    (-0.4, 1.35),
    (-0.35, 1.0),
    (-0.0, 0.87),
    (0.25, 1.02),
    (0.3, 1.45),
    (0.6, 1.6),
    (0.95, 1.45),
    (1.0, 0.4),
]
'''
'''
TRACK_POINTS_M = [
    (0.7, 0.1),
    (-0.7, 0.1),
    (-0.98, 0.4),
    (-0.98, 1.35),
    (-0.65, 1.5),
    (-0.4, 1.3),
    (-0.3, 1.0),
    (-0.1, 0.95),
    (0.0, 1.02),
    (0.2, 1.4),
    (0.5, 1.55),
    (0.95, 1.4),
    (1.0, 0.4),
]
'''

# Pista ativa: sim6 (pista em 8 do Pack2/simulate6.py escalada para a pista real).
# A pista real esta comentada no fim do ficheiro.
TRACK_POINTS_M = [
    (0.40, -0.02),
    (0.73, -0.01),
    (0.89, 0.29),
    (0.93, 1.18),
    (0.83, 1.38),
    (0.59, 1.48),
    (0.34, 1.41),
    (0.10, 1.13),
    (-0.27, 0.29),
    (-0.38, 0.12),
    (-0.58, 0.06),
    (-0.81, 0.20),
    (-0.94, 0.47),
    (-0.88, 1.26),
    (-0.75, 1.40),
    (-0.54, 1.45),
    (-0.38, 1.40),
    (-0.16, 1.10),
    (0.08, 0.33),
    (0.25, 0.07),
]


BARRIER_ROUND_CORNER_RADIUS_M = 0.20
BARRIER_ROUND_CORNER_SAMPLES = 24


BARRIER_INNER_M = [
    (0.461, 0.288),
    (0.54, 0.27),
    (0.60, 0.32),
    (0.62, 0.40),
    (0.63, 0.98),
    (0.64, 1.11),
    (0.59, 1.19),
    (0.48, 1.22),
    (0.42, 1.16),
    (0.31, 0.99),
    (0.25, 0.81),
    (0.25, 0.69),
    (0.27, 0.61),
    (0.32, 0.49),
    (0.35, 0.40),
    (0.395, 0.324),
]

# Ilhas interiores alem da BARRIER_INNER_M (pista em 8 -> 2 ilhas); vazio na pista real
BARRIER_INNER_EXTRA_M = [
    [
        (-0.52, 0.28),
        (-0.44, 0.32),
        (-0.39, 0.44),
        (-0.32, 0.66),
        (-0.31, 0.75),
        (-0.38, 1.01),
        (-0.48, 1.19),
        (-0.53, 1.22),
        (-0.61, 1.22),
        (-0.66, 1.17),
        (-0.69, 1.08),
        (-0.67, 0.39),
        (-0.63, 0.31),
    ],
]

BARRIER_OUTER_M = [
    (-1.12, -0.13),
    (-1.19, 0.02),
    (-1.21, 1.60),
    (-1.10, 1.71),
    (-0.99, 1.77),
    (-0.46, 1.78),
    (-0.12, 1.78),
    (0.08, 1.78),
    (0.43, 1.78),
    (0.96, 1.78),
    (1.16, 1.67),
    (1.19, 1.36),
    (1.16, 0.26),
    (1.15, -0.08),
    (1.00, -0.20),
    (0.64, -0.22),
    (0.18, -0.23),
    (-0.20, -0.22),
    (-0.93, -0.20),
]

ROBOT_ELLIPSE_A_M = 0.05
ROBOT_ELLIPSE_B_M = 0.02
# Marcadores do carro: bolinha verde no centro (posicao) e azul na frente (angulo)
GREEN_HSV_LOWER = (41, 44, 89)
GREEN_HSV_UPPER = (74, 255, 255)
BLUE_HSV_LOWER = (95, 80, 60)
BLUE_HSV_UPPER = (130, 255, 255)
# Desvio do angulo das bolinhas (theta = theta_bolinhas + desvio); valor inicial do
# filtro do track1.py, que o estima em andamento. Tecla O na janela Tracking guarda-o aqui.
MARKER_THETA_OFFSET_DEG = 0.0

# Parametros do controlador CBF-CLF-QP (lidos pelo run_real_esp_wifi.py e guardados
# pelo track1.py no JSON de cada video em recordings/). Muda-os AQUI, nao no runner.
CONTROLLER_PARAMS = {
    "dt": 0.02,                       # periodo do loop de controlo (s)
    "v_ref": 0.35,                    # velocidade de referencia (m/s)
    "v_max": 0.47,                    # saturacao final de v (m/s)
    "a_max": 2.0,
    "v_bounds": (0.0, 2.0),           # limites de v dentro do QP
    "ellipse_ab": (0.04, 0.03),       # semi-eixos da elipse do carro nas CBF (m)
    "margin": 0.01,                   # margem de seguranca das CBF (m)
    "lookahead_l": 0.1, #0.05 #0.1               # ponto a frente usado nas CBF dos obstaculos (m)
    "lookahead_barrier_l": 0.1,       # ponto a frente usado nas CBF das barreiras (m)
    "alpha": 3.0,                     # ganho da CBF
    "eps_clf": 8.0,  #8                 # taxa de decaimento da CLF
    "q_clf": (0.01, 10.0, 0.01),      # pesos da CLF (qx, qy, qpsi)
    "W": (2500000.0, 1.0),            # pesos do custo em (v, w)
    "p_slack": 50.0,    #50              # peso da folga da CLF
    "L": 0.06,                        # distancia entre eixos (m)
    "delta_max_deg": 13.0,
    "delta_rate_max_deg": 300.0,
    "omega_v_min": 0.2,               # v minimo na conversao w -> delta
    "path_corner_radius_m": 0.08,     # arredondamento dos cantos do caminho
    "path_corner_samples": 10,
    "path_ds": 0.01,                  # passo da spline do caminho (m)
    "delay_comp": True,               # compensacao do atraso (delay_comp.py)
    "loop_delay_s": 0.10,
    "steer_gain": 1.0,
    "pred_speed_source": "measured",  # "measured" (visao) ou "command"
}

'''
TRACK_POINTS_M = [
    (0.7, 0.1),

    (0.3, 0.1),
    (-0.2, 0.1),
    (-0.7, 0.1),

    (-0.82, 0.18),
    (-0.92, 0.28),
    (-0.98, 0.4),

    (-0.98, 0.7),
    (-0.98, 1.0),
    (-0.98, 1.35),

    (-0.9, 1.42),
    (-0.78, 1.48),
    (-0.65, 1.5),

    (-0.52, 1.46),
    (-0.44, 1.38),
    (-0.4, 1.3),

    (-0.37, 1.15),
    (-0.35, 1.0),
    (-0.35, 0.9),

    (-0.25, 0.84),
    (-0.12, 0.81),
    (0.0, 0.8),

    (0.12, 0.82),
    (0.18, 0.86),
    (0.2, 0.9),

    (0.2, 1.05),
    (0.2, 1.2),
    (0.2, 1.3),

    (0.22, 1.4),
    (0.25, 1.5),

    (0.35, 1.58),
    (0.42, 1.62),
    (0.5, 1.65),

    (0.65, 1.62),
    (0.78, 1.56),
    (0.85, 1.5),

    (0.9, 1.42),
    (0.93, 1.36),
    (0.95, 1.3),

    (0.97, 1.0),
    (0.99, 0.7),
    (1.0, 0.4),
]
'''
OBSTACLES_M = [
    {"x": -0.59, "y": 1.42, "r": 0.07},
    {"x": 0.92, "y": 0.81, "r": 0.07},
    {"x": -0.93, "y": 0.74, "r": 0.07},
    {"x": -0.01, "y": 0.88, "r": 0.07},
    {"x": 0.13, "y": 0.25, "r": 0.07},
    {"x": -0.53, "y": 0.10, "r": 0.07},
]


'''
# ============================================================
# Pista real (para voltar a ela, troca estes blocos pelos ativos acima e deixa
# BARRIER_INNER_EXTRA_M sem ilhas)
# ============================================================
TRACK_POINTS_M = [
    (0.73, -0.01),
    (-0.65, 0.05),
    (-0.96, 0.30),
    (-0.93, 1.10),
    (-0.66, 1.23),
    (-0.41, 1.20),
    (-0.25, 1.10),
    (-0.14, 1.03),
    (-0.01, 1.03),
    (0.23, 1.10),
    (0.49, 1.27),
    (0.74, 1.25),
    (0.89, 1.18),
    (0.95, 1.06),
    (0.96, 0.44),
]

BARRIER_INNER_M = [
    (0.52, 0.26),
    (-0.63, 0.30),
    (-0.68, 0.45),
    (-0.66, 1.00),
    (-0.59, 1.03),
    (-0.50, 1.01),
    (-0.46, 1.00),
    (-0.33, 0.68),
    (-0.02, 0.57),
    (0.24, 0.70),
    (0.44, 0.99),
    (0.54, 1.02),
    (0.61, 0.97),
    (0.65, 0.89),
    (0.60, 0.35),
]

BARRIER_OUTER_M = [
    (0.86, -0.23),
    (-0.95, -0.20),
    (-1.13, -0.11),
    (-1.22, 0.14),
    (-1.23, 1.55),
    (-1.14, 1.69),
    (-1.02, 1.76),
    (-0.63, 1.78),
    (-0.43, 1.71),
    (-0.22, 1.41),
    (-0.12, 1.29),
    (0.10, 1.27),
    (0.43, 1.65),
    (0.52, 1.73),
    (0.67, 1.77),
    (1.02, 1.76),
    (1.22, 1.58),
    (1.15, 0.01),
    (1.07, -0.17),
]

OBSTACLES_M = [
    {"x": -0.34, "y": 0.02, "r": 0.08},
    {"x": 0.92, "y": 0.46, "r": 0.08},
    {"x": -0.95, "y": 0.74, "r": 0.08},
    {"x": -0.05, "y": 1.07, "r": 0.08},
]
'''
