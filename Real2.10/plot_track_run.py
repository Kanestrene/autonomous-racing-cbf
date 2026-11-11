"""
Desenha o caminho de referencia (do track1_meta_<ts>.json) e o caminho que o carro
fez (do track1_log_<ts>.csv), com barreiras e obstaculos.

Uso:
    py -3.13 plot_track_run.py                       -> ultima gravacao em recordings/
    py -3.13 plot_track_run.py recordings/track1_log_20261107_175832.csv
    py -3.13 plot_track_run.py 20261107_175832
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RECORDINGS_DIR = Path(__file__).with_name("recordings")


def _xy(points):
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.empty(0), np.empty(0)
    return arr[:, 0], arr[:, 1]


def _closed(x, y):
    if len(x) and (x[0] != x[-1] or y[0] != y[-1]):
        return np.append(x, x[0]), np.append(y, y[0])
    return x, y


def _reference_path(geometry):
    path = geometry["path"]
    if "spline" in path:
        return np.asarray(path["spline"]["x"], float), np.asarray(path["spline"]["y"], float)
    return _closed(*_xy(path["rounded"]))


def load_run_csv(csv_path):
    """Devolve t_rel, x, y, speed; frames sem detecao ficam a NaN (quebram a linha)."""
    t, x, y, v = [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t.append(float(row["t_rel"]))
            except (TypeError, ValueError):
                continue
            if row.get("detected") == "1" and row.get("x_m") and row.get("y_m"):
                x.append(float(row["x_m"]))
                y.append(float(row["y_m"]))
                v.append(float(row["speed_m_s"]) if row.get("speed_m_s") else np.nan)
            else:
                x.append(np.nan)
                y.append(np.nan)
                v.append(np.nan)
    return np.array(t), np.array(x), np.array(y), np.array(v)


def _dist_to_polyline(px, py, rx, ry):
    """Distancia minima de cada ponto (px, py) a polyline (rx, ry)."""
    ax, ay = rx[:-1], ry[:-1]
    dx, dy = rx[1:] - ax, ry[1:] - ay
    seg_len2 = np.maximum(dx * dx + dy * dy, 1e-12)
    out = np.empty(len(px))
    for i, (qx, qy) in enumerate(zip(px, py)):
        u = np.clip(((qx - ax) * dx + (qy - ay) * dy) / seg_len2, 0.0, 1.0)
        out[i] = np.sqrt(np.min((ax + u * dx - qx) ** 2 + (ay + u * dy - qy) ** 2))
    return out


def plot_run(csv_path, meta_path=None, out_path=None, show_pdf=True):
    csv_path = Path(csv_path)
    if meta_path is None:
        meta_path = csv_path.with_name(csv_path.name.replace("track1_log_", "track1_meta_")).with_suffix(".json")
    meta_path = Path(meta_path)
    if out_path is None:
        out_path = csv_path.with_name(csv_path.name.replace("track1_log_", "track1_path_")).with_suffix(".png")
    out_path = Path(out_path)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    geometry = meta["geometry"]
    t, x, y, v = load_run_csv(csv_path)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        print(f"Sem posicoes detetadas em {csv_path.name}; imagem nao gerada.")
        return None

    fig, ax = plt.subplots(figsize=(8, 7))

    # barreiras
    barriers = geometry.get("barriers", {})
    barrier_label = "Barreiras"
    for b in [barriers.get("outer"), barriers.get("inner"), *barriers.get("inner_extra", [])]:
        if not b:
            continue
        bx, by = _closed(*_xy(b["rounded"]))
        ax.plot(bx, by, color="0.25", lw=1.6, label=barrier_label)
        barrier_label = None

    # obstaculos
    obstacle_label = "Obstaculos"
    for o in geometry.get("obstacles", []):
        ax.add_patch(plt.Circle((o["x"], o["y"]), o["r"], color="tab:red", alpha=0.35,
                                label=obstacle_label))
        obstacle_label = None

    # referencia
    rx, ry = _reference_path(geometry)
    ax.plot(rx, ry, "--", color="tab:blue", lw=1.8, label="Caminho de referencia")
    if "geometry_end" in meta:
        ex, ey = _reference_path(meta["geometry_end"])
        ax.plot(ex, ey, ":", color="tab:purple", lw=1.5, label="Referencia (editada no fim)")

    # caminho real
    ax.plot(x, y, "-", color="tab:orange", lw=1.6, label="Caminho do carro")
    i0 = np.flatnonzero(valid)[0]
    i1 = np.flatnonzero(valid)[-1]
    ax.plot(x[i0], y[i0], "o", color="tab:green", ms=8, label="Inicio")
    ax.plot(x[i1], y[i1], "s", color="black", ms=7, label="Fim")

    # erro ao caminho
    err = _dist_to_polyline(x[valid], y[valid], rx, ry)
    rms = float(np.sqrt(np.mean(err ** 2)))
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    stats = (f"Erro ao caminho: RMS {rms * 100:.1f} cm | max {err.max() * 100:.1f} cm\n"
             f"Duracao {duration:.1f} s | v_media {np.nanmean(v):.2f} m/s | "
             f"detecao {100.0 * valid.mean():.0f}%")

    params = meta.get("controller_params", {})
    title = f"Track1 {meta.get('timestamp', '')}"
    if "v_ref" in params:
        title += f"  (v_ref={params['v_ref']} m/s)"
    ax.set_title(title, fontsize=11)
    ax.text(0.01, 0.01, stats, transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Imagem do caminho guardada em {out_path}")
    return out_path


def _resolve_csv(arg):
    if arg is None:
        logs = sorted(RECORDINGS_DIR.glob("track1_log_*.csv"))
        if not logs:
            raise SystemExit(f"Nenhum track1_log_*.csv em {RECORDINGS_DIR}")
        return logs[-1]
    p = Path(arg)
    if p.exists():
        return p
    return RECORDINGS_DIR / f"track1_log_{arg}.csv"


if __name__ == "__main__":
    plot_run(_resolve_csv(sys.argv[1] if len(sys.argv) > 1 else None))
