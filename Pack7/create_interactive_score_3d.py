from pathlib import Path

import argparse
import json

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "ml_cbf_pq_model.npz"
DEFAULT_OUTPUT = BASE_DIR / "ml_cbf_interactive_score_3d.html"


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Interactive ML-CBF Score Surface</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
    }
    body {
      margin: 0;
      overflow: hidden;
      background: #f7f8fb;
      color: #17202a;
    }
    #viewer {
      display: block;
      width: 100vw;
      height: 100vh;
      cursor: grab;
    }
    #viewer:active {
      cursor: grabbing;
    }
    .toolbar {
      position: fixed;
      top: 14px;
      left: 14px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px;
      border: 1px solid rgba(20, 30, 40, 0.16);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.88);
      box-shadow: 0 8px 24px rgba(20, 30, 40, 0.12);
      backdrop-filter: blur(8px);
      user-select: none;
    }
    button, label {
      height: 32px;
      border: 1px solid rgba(20, 30, 40, 0.18);
      border-radius: 6px;
      background: white;
      color: #17202a;
      font-size: 13px;
      line-height: 30px;
      padding: 0 10px;
    }
    button {
      cursor: pointer;
    }
    label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    input {
      margin: 0;
    }
    .readout {
      position: fixed;
      right: 14px;
      bottom: 14px;
      max-width: min(420px, calc(100vw - 28px));
      padding: 10px 12px;
      border: 1px solid rgba(20, 30, 40, 0.16);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.88);
      box-shadow: 0 8px 24px rgba(20, 30, 40, 0.10);
      font-size: 13px;
      line-height: 1.45;
      backdrop-filter: blur(8px);
    }
    .legend {
      position: fixed;
      top: 14px;
      right: 14px;
      padding: 10px 12px;
      border: 1px solid rgba(20, 30, 40, 0.16);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.88);
      box-shadow: 0 8px 24px rgba(20, 30, 40, 0.10);
      font-size: 13px;
      line-height: 1.55;
      backdrop-filter: blur(8px);
    }
    .swatch {
      display: inline-block;
      width: 11px;
      height: 11px;
      margin-right: 6px;
      border-radius: 999px;
      vertical-align: -1px;
      background: #1f77b4;
    }
    .star {
      color: #d89a00;
      margin-right: 4px;
    }
  </style>
</head>
<body>
  <canvas id="viewer"></canvas>
  <div class="toolbar">
    <button id="reset">Reset</button>
    <button id="zoomIn">+</button>
    <button id="zoomOut">-</button>
    <label><input id="surfaceToggle" type="checkbox" checked /> Surface</label>
    <label><input id="pointsToggle" type="checkbox" checked /> Points</label>
  </div>
  <div class="legend">
    <div><span class="swatch"></span>Feasible samples</div>
    <div><span class="star">★</span>Max feasible score</div>
  </div>
  <div class="readout" id="readout"></div>
  <script>
const DATA = __DATA__;

const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");
const readout = document.getElementById("readout");
const surfaceToggle = document.getElementById("surfaceToggle");
const pointsToggle = document.getElementById("pointsToggle");

let width = 0;
let height = 0;
let dpr = window.devicePixelRatio || 1;
let rotX = -0.78;
let rotY = 0.76;
let zoom = 1.0;
let panX = 0;
let panY = 0;
let dragging = false;
let lastX = 0;
let lastY = 0;
let panMode = false;

function resize() {
  dpr = window.devicePixelRatio || 1;
  width = Math.floor(window.innerWidth);
  height = Math.floor(window.innerHeight);
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = width + "px";
  canvas.style.height = height + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function norm(value, lo, hi) {
  if (hi === lo) return 0;
  return 2 * (value - lo) / (hi - lo) - 1;
}

function colorMap(value) {
  const t = clamp((value - DATA.zMin) / (DATA.zMax - DATA.zMin || 1), 0, 1);
  const stops = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ];
  const scaled = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const a = stops[i];
  const b = stops[i + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}

function world(q, pLog, score) {
  return {
    x: norm(q, DATA.qMin, DATA.qMax),
    y: norm(pLog, DATA.pLogMin, DATA.pLogMax),
    z: norm(score, DATA.zMin, DATA.zMax),
    score,
  };
}

function rotatePoint(point) {
  const cx = Math.cos(rotX);
  const sx = Math.sin(rotX);
  const cy = Math.cos(rotY);
  const sy = Math.sin(rotY);

  const x1 = point.x * cy + point.z * sy;
  const z1 = -point.x * sy + point.z * cy;
  const y1 = point.y * cx - z1 * sx;
  const z2 = point.y * sx + z1 * cx;
  return { x: x1, y: y1, z: z2, score: point.score };
}

function project(point) {
  const r = rotatePoint(point);
  const scale = Math.min(width, height) * 0.34 * zoom;
  const perspective = 1 / (1 + 0.12 * r.z);
  return {
    x: width * 0.5 + panX + r.x * scale * perspective,
    y: height * 0.52 + panY - r.y * scale * perspective,
    z: r.z,
    score: point.score,
  };
}

function makeGridPoint(i, j) {
  return world(DATA.qGrid[j], DATA.pLogGrid[i], DATA.zGrid[i][j]);
}

function drawPolygon(points, fillStyle, alpha) {
  if (!points.length) return;
  ctx.globalAlpha = alpha;
  ctx.fillStyle = fillStyle;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
}

function drawSurface() {
  const rows = DATA.pLogGrid.length;
  const cols = DATA.qGrid.length;
  const faces = [];
  for (let i = 0; i < rows - 1; i += 1) {
    for (let j = 0; j < cols - 1; j += 1) {
      const pts = [
        makeGridPoint(i, j),
        makeGridPoint(i, j + 1),
        makeGridPoint(i + 1, j + 1),
        makeGridPoint(i + 1, j),
      ];
      const projected = pts.map(project);
      const depth = projected.reduce((sum, p) => sum + p.z, 0) / projected.length;
      const avgScore = pts.reduce((sum, p) => sum + p.score, 0) / pts.length;
      faces.push({ projected, depth, avgScore });
    }
  }
  faces.sort((a, b) => a.depth - b.depth);
  for (const face of faces) {
    drawPolygon(face.projected, colorMap(face.avgScore), 0.72);
  }
}

function drawLine(a, b, color = "#1f2933", widthPx = 1) {
  const pa = project(a);
  const pb = project(b);
  ctx.strokeStyle = color;
  ctx.lineWidth = widthPx;
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
}

function drawText(text, point, dx = 0, dy = 0, color = "#17202a") {
  const p = project(point);
  ctx.fillStyle = color;
  ctx.font = "13px Arial, Helvetica, sans-serif";
  ctx.fillText(text, p.x + dx, p.y + dy);
}

function drawAxes() {
  const x0 = world(DATA.qMin, DATA.pLogMin, DATA.zMin);
  const x1 = world(DATA.qMax, DATA.pLogMin, DATA.zMin);
  const y1 = world(DATA.qMin, DATA.pLogMax, DATA.zMin);
  const z1 = world(DATA.qMin, DATA.pLogMin, DATA.zMax);
  drawLine(x0, x1, "#283747", 1.4);
  drawLine(x0, y1, "#283747", 1.4);
  drawLine(x0, z1, "#283747", 1.4);
  drawText("q", x1, 8, 0);
  drawText("p", y1, 8, 0);
  drawText("score", z1, 8, 0);
}

function drawPoints() {
  const points = DATA.points.map((point) => {
    const projected = project(world(point.q, point.pLog, point.score));
    return { ...point, projected };
  });
  points.sort((a, b) => a.projected.z - b.projected.z);
  for (const point of points) {
    const p = point.projected;
    ctx.beginPath();
    ctx.fillStyle = "#1f77b4";
    ctx.strokeStyle = "white";
    ctx.lineWidth = 1;
    ctx.arc(p.x, p.y, 4.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  const best = DATA.best;
  const bp = project(world(best.q, best.pLog, best.score));
  drawStar(bp.x, bp.y, 14, 6, "#ffbf00", "#111111");
  ctx.fillStyle = "#111111";
  ctx.font = "13px Arial, Helvetica, sans-serif";
  ctx.fillText(`p=${best.p.toFixed(3)}, q=${best.q.toFixed(3)}`, bp.x + 12, bp.y - 10);
}

function drawStar(cx, cy, outer, inner, fill, stroke) {
  ctx.beginPath();
  for (let i = 0; i < 10; i += 1) {
    const r = i % 2 === 0 ? outer : inner;
    const a = -Math.PI / 2 + i * Math.PI / 5;
    const x = cx + Math.cos(a) * r;
    const y = cy + Math.sin(a) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.5;
  ctx.fill();
  ctx.stroke();
}

function drawColorBar() {
  const barX = width - 62;
  const barY = 92;
  const barW = 18;
  const barH = Math.min(360, height - 190);
  const grad = ctx.createLinearGradient(0, barY + barH, 0, barY);
  for (let i = 0; i <= 20; i += 1) {
    const t = i / 20;
    grad.addColorStop(t, colorMap(DATA.zMin + t * (DATA.zMax - DATA.zMin)));
  }
  ctx.fillStyle = grad;
  ctx.fillRect(barX, barY, barW, barH);
  ctx.strokeStyle = "rgba(20,30,40,0.45)";
  ctx.strokeRect(barX, barY, barW, barH);
  ctx.fillStyle = "#17202a";
  ctx.font = "12px Arial, Helvetica, sans-serif";
  ctx.fillText(DATA.zMax.toFixed(1), barX - 12, barY - 8);
  ctx.fillText(DATA.zMin.toFixed(1), barX - 12, barY + barH + 18);
}

function drawTitle() {
  ctx.fillStyle = "#17202a";
  ctx.font = "18px Arial, Helvetica, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(DATA.title, width / 2, 42);
  ctx.textAlign = "left";
}

function draw() {
  ctx.clearRect(0, 0, width, height);
  drawTitle();
  if (surfaceToggle.checked) drawSurface();
  drawAxes();
  if (pointsToggle.checked) drawPoints();
  drawColorBar();
  readout.innerHTML = `
    <b>Best feasible sample</b><br>
    p = ${DATA.best.p.toFixed(6)} &nbsp; q = ${DATA.best.q.toFixed(6)}<br>
    score = ${DATA.best.score.toFixed(6)}<br>
    points = ${DATA.points.length}, grid = ${DATA.qGrid.length} x ${DATA.pLogGrid.length}
  `;
}

canvas.addEventListener("pointerdown", (event) => {
  dragging = true;
  panMode = event.shiftKey || event.button === 1;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  const dx = event.clientX - lastX;
  const dy = event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;
  if (panMode) {
    panX += dx;
    panY += dy;
  } else {
    rotY += dx * 0.008;
    rotX += dy * 0.008;
    rotX = clamp(rotX, -Math.PI * 0.49, Math.PI * 0.49);
  }
  draw();
});

canvas.addEventListener("pointerup", (event) => {
  dragging = false;
  canvas.releasePointerCapture(event.pointerId);
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const factor = event.deltaY < 0 ? 1.08 : 0.92;
  zoom = clamp(zoom * factor, 0.35, 5.0);
  draw();
}, { passive: false });

document.getElementById("reset").addEventListener("click", () => {
  rotX = -0.78;
  rotY = 0.76;
  zoom = 1.0;
  panX = 0;
  panY = 0;
  draw();
});
document.getElementById("zoomIn").addEventListener("click", () => {
  zoom = clamp(zoom * 1.15, 0.35, 5.0);
  draw();
});
document.getElementById("zoomOut").addEventListener("click", () => {
  zoom = clamp(zoom / 1.15, 0.35, 5.0);
  draw();
});
surfaceToggle.addEventListener("change", draw);
pointsToggle.addEventListener("change", draw);
window.addEventListener("resize", resize);
resize();
  </script>
</body>
</html>
"""


def normalize_params(params, bounds, log_scale):
    params = np.asarray(params, dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    out = np.empty_like(params, dtype=float)

    linear = ~log_scale
    out[..., linear] = (params[..., linear] - low[linear]) / (high[linear] - low[linear])

    log_low = np.log(low[log_scale])
    log_high = np.log(high[log_scale])
    out[..., log_scale] = (np.log(params[..., log_scale]) - log_low) / (
        log_high - log_low
    )
    return np.clip(out, 0.0, 1.0)


def rbf_surface(train_normal, train_scores, bounds, log_scale, grid_size, bandwidth):
    sigma = float(bandwidth)
    ridge = 1e-8
    dist2 = np.sum(
        (train_normal[:, None, :] - train_normal[None, :, :]) ** 2,
        axis=2,
    )
    kernel = np.exp(-0.5 * dist2 / sigma**2)
    coef = np.linalg.solve(kernel + ridge * np.eye(len(train_normal)), train_scores)

    p_min, p_max = bounds[0]
    q_min, q_max = bounds[1]
    q_grid = np.linspace(q_min, q_max, grid_size)
    p_grid = np.geomspace(p_min, p_max, grid_size)
    q_mesh, p_mesh = np.meshgrid(q_grid, p_grid)
    raw_grid = np.c_[p_mesh.ravel(), q_mesh.ravel()]
    normal_grid = normalize_params(raw_grid, bounds, log_scale)

    grid_dist2 = np.sum(
        (normal_grid[:, None, :] - train_normal[None, :, :]) ** 2,
        axis=2,
    )
    grid_kernel = np.exp(-0.5 * grid_dist2 / sigma**2)
    z_grid = (grid_kernel @ coef).reshape(q_mesh.shape)
    return q_grid, p_grid, z_grid


def smooth_surface(train_normal, train_scores, bounds, log_scale, grid_size, bandwidth):
    p_min, p_max = bounds[0]
    q_min, q_max = bounds[1]
    q_grid = np.linspace(q_min, q_max, grid_size)
    p_grid = np.geomspace(p_min, p_max, grid_size)
    q_mesh, p_mesh = np.meshgrid(q_grid, p_grid)
    raw_grid = np.c_[p_mesh.ravel(), q_mesh.ravel()]
    normal_grid = normalize_params(raw_grid, bounds, log_scale)

    diff = normal_grid[:, None, :] - train_normal[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    weights = np.exp(-0.5 * dist2 / max(float(bandwidth), 1e-6) ** 2)
    denom = np.sum(weights, axis=1)
    z = np.full(normal_grid.shape[0], float(np.mean(train_scores)), dtype=float)
    good = denom > 1e-12
    z[good] = weights[good] @ train_scores / denom[good]
    return q_grid, p_grid, z.reshape(q_mesh.shape)


def round_nested(array, decimals=5):
    return np.round(np.asarray(array, dtype=float), decimals=decimals).tolist()


def build_html(model_path, output_path, grid_size, method, include_all_points, clip_percentiles):
    model = np.load(model_path, allow_pickle=True)
    train_params = model["train_params"].astype(float)
    train_normal = model["train_normal_params"].astype(float)
    scores = model["scores"].astype(float)
    labels = model["labels"].astype(int)
    bounds = model["param_bounds"].astype(float)
    log_scale = model["param_log_scale"].astype(bool)
    score_bandwidth = float(model["score_bandwidth"][0])

    mask = np.ones_like(labels, dtype=bool) if include_all_points else labels > 0
    selected_params = train_params[mask]
    selected_normal = train_normal[mask]
    selected_scores = scores[mask]

    if len(selected_scores) == 0:
        raise ValueError("no samples selected for plotting")

    if method == "smooth":
        q_grid, p_grid, z_grid = smooth_surface(
            selected_normal,
            selected_scores,
            bounds,
            log_scale,
            grid_size,
            score_bandwidth,
        )
        method_label = "smoothed"
    else:
        q_grid, p_grid, z_grid = rbf_surface(
            selected_normal,
            selected_scores,
            bounds,
            log_scale,
            grid_size,
            score_bandwidth,
        )
        method_label = "RBF-interpolated"

    lo_pct, hi_pct = clip_percentiles
    z_min = float(np.percentile(z_grid, lo_pct))
    z_max = float(np.percentile(z_grid, hi_pct))
    z_grid = np.clip(z_grid, z_min, z_max)

    best_idx = int(np.argmax(selected_scores))
    best_p = float(selected_params[best_idx, 0])
    best_q = float(selected_params[best_idx, 1])
    best_score = float(selected_scores[best_idx])

    p_logs = np.log10(selected_params[:, 0])
    points = [
        {
            "p": float(p_val),
            "pLog": float(p_log),
            "q": float(q_val),
            "score": float(score),
        }
        for p_val, p_log, q_val, score in zip(
            selected_params[:, 0],
            p_logs,
            selected_params[:, 1],
            selected_scores,
        )
    ]

    data = {
        "title": f"Interactive {method_label} score surface",
        "qGrid": round_nested(q_grid, 5),
        "pLogGrid": round_nested(np.log10(p_grid), 5),
        "zGrid": round_nested(z_grid, 5),
        "points": points,
        "best": {
            "p": best_p,
            "pLog": float(np.log10(best_p)),
            "q": best_q,
            "score": best_score,
        },
        "qMin": float(bounds[1, 0]),
        "qMax": float(bounds[1, 1]),
        "pLogMin": float(np.log10(bounds[0, 0])),
        "pLogMax": float(np.log10(bounds[0, 1])),
        "zMin": z_min,
        "zMax": z_max,
    }

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path, len(points), best_p, best_q, best_score


def main():
    parser = argparse.ArgumentParser(
        description="Create a dependency-free interactive HTML 3D score graph."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--method", choices=("rbf", "smooth"), default="rbf")
    parser.add_argument("--all-points", action="store_true")
    parser.add_argument(
        "--clip-percentiles",
        type=float,
        nargs=2,
        default=(2.0, 98.0),
        metavar=("LOW", "HIGH"),
    )
    args = parser.parse_args()

    output, n_points, best_p, best_q, best_score = build_html(
        model_path=args.model,
        output_path=args.output,
        grid_size=args.grid_size,
        method=args.method,
        include_all_points=args.all_points,
        clip_percentiles=args.clip_percentiles,
    )
    print(f"Saved: {output}")
    print(f"Points: {n_points}")
    print(f"Best point: p={best_p:.6f}, q={best_q:.6f}, score={best_score:.6f}")


if __name__ == "__main__":
    main()
