"""
Pistas do Pack2 (simulate4/5/6) escaladas para a area da pista real.

Os WAYPOINTS sao lidos diretamente do Pack2/simulateN.py (sem importar o modulo),
por isso qualquer alteracao a pista na simulacao aparece aqui automaticamente.
Barreiras e obstaculos sao gerados com a mesma regra da simulacao:
  - barreiras = caminho +- SIM_HALF_WIDTH na normal
  - 5 obstaculos equidistantes ao longo do caminho, alternando +-SIM_OBS_OFFSET
e depois tudo e escalado por REAL_SCALE e centrado em REAL_CENTER_M.

Uso: em shared_config.py, TRACK_NAME = "sim4" / "sim5" / "sim6".
"""
import ast
from pathlib import Path

import numpy as np

from controller import build_spline_path

PACK2_DIR = Path(__file__).resolve().parents[1] / "Pack2"
SIM_TRACK_FILES = {
    "sim4": "simulate4.py",
    "sim5": "simulate5.py",
    "sim6": "simulate6.py",
}

# Retangulo onde estao as pistas do Pack2 [m simulacao] e onde fica na pista real [m]
SIM_BOX_X = (2.0, 16.0)
SIM_BOX_Y = (2.0, 15.0)
REAL_CENTER_M = (0.0, 0.83)
REAL_SCALE = 0.11  # 13 m de altura no Pack2 -> ~1.43 m (altura da pista real)

# Parametros da simulacao [m simulacao], iguais aos do Pack2/simulateN.py
SIM_HALF_WIDTH = 1.5
SIM_N_OBS = 5
SIM_OBS_OFFSET = 0.3
SIM_OBS_R = 0.35
SIM_DS = 0.01

# Espacamento dos pontos das barreiras [m simulacao] (pontos editaveis no track1.py)
BARRIER_POINT_SPACING = 0.7


def _read_waypoints(track_name):
    if track_name not in SIM_TRACK_FILES:
        raise ValueError(
            f"Pista '{track_name}' desconhecida. Opcoes: 'real', {', '.join(map(repr, SIM_TRACK_FILES))}."
        )

    path = PACK2_DIR / SIM_TRACK_FILES[track_name]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "WAYPOINTS" for t in node.targets)
        ):
            return [tuple(map(float, p)) for p in ast.literal_eval(node.value)]

    raise RuntimeError(f"Nao encontrei WAYPOINTS em {path}.")


def _to_real(points):
    pts = np.asarray(points, dtype=float)
    cx = 0.5 * (SIM_BOX_X[0] + SIM_BOX_X[1])
    cy = 0.5 * (SIM_BOX_Y[0] + SIM_BOX_Y[1])
    x = REAL_CENTER_M[0] + REAL_SCALE * (pts[:, 0] - cx)
    y = REAL_CENTER_M[1] + REAL_SCALE * (pts[:, 1] - cy)
    return [(round(float(a), 3), round(float(b), 3)) for a, b in zip(x, y)]


def load_sim_track(track_name):
    """
    Devolve dict com 'track', 'inner', 'outer' (listas de (x, y) em metros reais),
    'obstacles' (lista de {"x", "y", "r"}) e 'start' (x, y, yaw) do inicio do caminho.
    """
    waypoints = _read_waypoints(track_name)
    px, py, pyaw, _ = build_spline_path(waypoints, ds=SIM_DS)
    n_path = len(px)
    nx = -np.sin(pyaw)
    ny = np.cos(pyaw)

    # obstaculos: mesma regra do Pack2/simulateN.py
    idxs = np.linspace(0, n_path - 1, SIM_N_OBS + 2, dtype=int)[1:-1]
    obs_xy = []
    for k, idx in enumerate(idxs):
        side = (-1) ** k
        obs_xy.append((px[idx] + side * SIM_OBS_OFFSET * nx[idx],
                       py[idx] + side * SIM_OBS_OFFSET * ny[idx]))
    obstacles = [
        {"x": x, "y": y, "r": round(SIM_OBS_R * REAL_SCALE, 3)}
        for x, y in _to_real(obs_xy)
    ]

    # barreiras: offsets +-SIM_HALF_WIDTH (no simulate6 cruzam-se no centro do 8)
    step = max(1, int(round(BARRIER_POINT_SPACING / SIM_DS)))
    left = np.column_stack([px + SIM_HALF_WIDTH * nx, py + SIM_HALF_WIDTH * ny])[::step]
    right = np.column_stack([px - SIM_HALF_WIDTH * nx, py - SIM_HALF_WIDTH * ny])[::step]

    def area(poly):
        return 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                         - np.dot(np.roll(poly[:, 0], -1), poly[:, 1]))

    inner, outer = (left, right) if area(left) < area(right) else (right, left)

    # o caminho real fecha sozinho (build_rounded_polyline closed=True): tirar o ponto repetido
    if np.allclose(waypoints[0], waypoints[-1]):
        waypoints = waypoints[:-1]

    return {
        "track": _to_real(waypoints),
        "inner": _to_real(inner),
        "outer": _to_real(outer),
        "obstacles": obstacles,
        "start": (*_to_real([(px[0], py[0])])[0], float(pyaw[0])),
    }


if __name__ == "__main__":
    for name in SIM_TRACK_FILES:
        t = load_sim_track(name)
        xy = np.array(t["track"])
        print(
            f"{name}: {len(t['track'])} pontos | x=[{xy[:, 0].min():.2f}, {xy[:, 0].max():.2f}] "
            f"y=[{xy[:, 1].min():.2f}, {xy[:, 1].max():.2f}] m | "
            f"inicio=({t['start'][0]:.2f}, {t['start'][1]:.2f}) m, "
            f"yaw={np.degrees(t['start'][2]):.0f} graus | obstaculos={len(t['obstacles'])}"
        )
