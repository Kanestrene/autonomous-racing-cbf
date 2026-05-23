import time
import json
import socket
import os
import numpy as np
import cv2
from pupil_apriltags import Detector
from ultralytics import YOLO
from shared_config import OBSTACLES_M, TRACK_POINTS_M

# =========================
# CONFIG
# =========================
PC_IP = "127.0.0.1"
PC_PORT = 5005

SHOW_WINDOW = True
SHOW_MASK = True
SHOW_HEATMAPS = True
SHOW_REFERENCE_PATH = True
SHOW_TRACK_TRAIL = True
SEND_UDP = True

WIDTH, HEIGHT = 1280, 720
TARGET_FPS = 30
OPENCV_CAMERA_INDEX = 0


# Referencial em metros
# (616, 600) px -> (0.0, 0.0) m
# (892, 600) px -> (0.9, 0.0) m
REF_PX_ORIGIN = (616.0, 600.0)
REF_PX_X_AXIS = (892.0, 600.0)
REF_X_METERS = 0.9

# Pontos em metros para desenhar no frame
MARK_POINTS_M = list(TRACK_POINTS_M)
DRAW_OBSTACLES_M = [dict(obstacle) for obstacle in OBSTACLES_M]
ROUND_CORNER_RADIUS_M = 0.08
ROUND_CORNER_SAMPLES = 10
TRACK_TRAIL_MAX_POINTS = 120
TRACK_TRAIL_MIN_STEP_PX = 6.0

if REF_PX_X_AXIS[0] == REF_PX_ORIGIN[0]:
    raise ValueError("Referencia invalida: pontos de calibracao com o mesmo x em pixels.")

PIXEL_TO_METER = REF_X_METERS / (REF_PX_X_AXIS[0] - REF_PX_ORIGIN[0])


def px_to_m(x_px, y_px):
    x_m = (x_px - REF_PX_ORIGIN[0]) * PIXEL_TO_METER
    y_m = (REF_PX_ORIGIN[1] - y_px) * PIXEL_TO_METER
    return float(x_m), float(y_m)


def m_to_px(x_m, y_m):
    x_px = REF_PX_ORIGIN[0] + (x_m / PIXEL_TO_METER)
    y_px = REF_PX_ORIGIN[1] - (y_m / PIXEL_TO_METER)
    return float(x_px), float(y_px)


def build_rounded_polyline(points, corner_radius=0.08, corner_samples=10, closed=True):
    n = len(points)
    if n < 2:
        return points.copy()
    if n == 2:
        return points.copy()

    pts = [np.array(p, dtype=float) for p in points]

    if not closed:
        out_open = [tuple(pts[0])]
        for i in range(1, n - 1):
            p_prev = pts[i - 1]
            p_curr = pts[i]
            p_next = pts[i + 1]
            v_in = p_curr - p_prev
            v_out = p_next - p_curr
            len_in = float(np.linalg.norm(v_in))
            len_out = float(np.linalg.norm(v_out))
            if len_in < 1e-9 or len_out < 1e-9:
                out_open.append(tuple(p_curr))
                continue
            u_in = v_in / len_in
            u_out = v_out / len_out
            d = min(corner_radius, 0.45 * len_in, 0.45 * len_out)
            p_start = p_curr - u_in * d
            p_end = p_curr + u_out * d
            out_open.append(tuple(p_start))
            for t in np.linspace(0.0, 1.0, corner_samples + 2)[1:-1]:
                pt = ((1.0 - t) ** 2) * p_start + 2.0 * (1.0 - t) * t * p_curr + (t ** 2) * p_end
                out_open.append((float(pt[0]), float(pt[1])))
            out_open.append(tuple(p_end))
        out_open.append(tuple(pts[-1]))
        return out_open

    out = []
    first_start = None

    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]

        v_in = p_curr - p_prev
        v_out = p_next - p_curr
        len_in = float(np.linalg.norm(v_in))
        len_out = float(np.linalg.norm(v_out))

        if len_in < 1e-9 or len_out < 1e-9:
            continue

        u_in = v_in / len_in
        u_out = v_out / len_out
        d = min(corner_radius, 0.45 * len_in, 0.45 * len_out)

        p_start = p_curr - u_in * d
        p_end = p_curr + u_out * d

        if i == 0:
            first_start = (float(p_start[0]), float(p_start[1]))
            out.append(first_start)
        else:
            out.append((float(p_start[0]), float(p_start[1])))

        for t in np.linspace(0.0, 1.0, corner_samples + 2)[1:-1]:
            pt = ((1.0 - t) ** 2) * p_start + 2.0 * (1.0 - t) * t * p_curr + (t ** 2) * p_end
            out.append((float(pt[0]), float(pt[1])))

        out.append((float(p_end[0]), float(p_end[1])))

    return out


def vel_px_to_m(vx_px_s, vy_px_s):
    vx_m_s = vx_px_s * PIXEL_TO_METER
    vy_m_s = -vy_px_s * PIXEL_TO_METER
    return float(vx_m_s), float(vy_m_s)


def acc_px_to_m(ax_px_s2, ay_px_s2):
    ax_m_s2 = ax_px_s2 * PIXEL_TO_METER
    ay_m_s2 = -ay_px_s2 * PIXEL_TO_METER
    return float(ax_m_s2), float(ay_m_s2)


def wrap_angle_pi(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def angle_diff_abs(a, b):
    return abs(wrap_angle_pi(a - b))


def update_raw_theta(track, x_m, y_m, t_now):
    track["raw_x_m"] = float(x_m)
    track["raw_y_m"] = float(y_m)

    speed_anchor_pos_m = track.get("raw_speed_anchor_pos_m")
    speed_anchor_t = track.get("raw_speed_anchor_t")
    speed_cycle_count = int(track.get("raw_speed_cycle_count", 0))
    if speed_anchor_pos_m is None or speed_anchor_t is None:
        track["raw_speed_anchor_pos_m"] = (x_m, y_m)
        track["raw_speed_anchor_t"] = float(t_now)
        track["raw_speed_cycle_count"] = 0
    else:
        speed_cycle_count += 1
        if speed_cycle_count >= RAW_SPEED_UPDATE_CYCLES:
            dx_speed = x_m - speed_anchor_pos_m[0]
            dy_speed = y_m - speed_anchor_pos_m[1]
            dt_speed = max(1e-6, float(t_now - speed_anchor_t))
            track["raw_speed_m_s"] = float(np.hypot(dx_speed, dy_speed)) / dt_speed
            track["raw_speed_anchor_pos_m"] = (x_m, y_m)
            track["raw_speed_anchor_t"] = float(t_now)
            track["raw_speed_cycle_count"] = 0
        else:
            track["raw_speed_cycle_count"] = speed_cycle_count

    raw_theta_anchor_pos_m = track.get("raw_theta_anchor_pos_m")
    if raw_theta_anchor_pos_m is None:
        track["raw_theta_anchor_pos_m"] = (x_m, y_m)
    else:
        dx = x_m - raw_theta_anchor_pos_m[0]
        dy = y_m - raw_theta_anchor_pos_m[1]
        step_m = float(np.hypot(dx, dy))

        if step_m >= RAW_THETA_MIN_UPDATE_M:
            theta_raw = float(np.arctan2(dy, dx))
            theta_alt = wrap_angle_pi(theta_raw + np.pi)
            theta_prev = float(track.get("raw_theta_rad", 0.0))

            if angle_diff_abs(theta_raw, theta_prev) <= angle_diff_abs(theta_alt, theta_prev):
                track["raw_theta_rad"] = theta_raw
            else:
                track["raw_theta_rad"] = theta_alt

            track["raw_theta_anchor_pos_m"] = (x_m, y_m)

    track["prev_raw_pos_m"] = (x_m, y_m)
    track["prev_raw_t"] = float(t_now)
    return float(track.get("raw_theta_rad", 0.0))


def update_track_theta(track):
    x_px = float(track.get("x_px", 0.0))
    y_px = float(track.get("y_px", 0.0))
    vx_px_s = float(track.get("vx_px_s", 0.0))
    vy_px_s = float(track.get("vy_px_s", 0.0))
    x_m, y_m = px_to_m(x_px, y_px)
    vx_m_s, vy_m_s = vel_px_to_m(vx_px_s, vy_px_s)
    speed_m_s = float(np.hypot(vx_m_s, vy_m_s))

    prev_pos_m = track.get("prev_pos_m")
    if prev_pos_m is not None:
        dx = x_m - prev_pos_m[0]
        dy = y_m - prev_pos_m[1]
        step_m = float(np.hypot(dx, dy))

        if step_m >= THETA_MIN_STEP_M and speed_m_s >= THETA_MIN_SPEED_M_S:
            theta_raw = float(np.arctan2(dy, dx))
            theta_alt = wrap_angle_pi(theta_raw + np.pi)
            theta_prev = float(track["theta_rad"])

            if angle_diff_abs(theta_raw, theta_prev) <= angle_diff_abs(theta_alt, theta_prev):
                track["theta_rad"] = theta_raw
            else:
                track["theta_rad"] = theta_alt

    track["prev_pos_m"] = (x_m, y_m)
    track["speed_m_s"] = speed_m_s
    return track["theta_rad"], np.degrees(track["theta_rad"]), speed_m_s

DT_MIN = 1 / 240
DT_MAX = 1 / 20
RAW_SPEED_UPDATE_CYCLES = 1
THETA_MIN_STEP_M = 0.1
THETA_MIN_SPEED_M_S = 0.03
RAW_THETA_MIN_UPDATE_M = 0.12

# Persistência dos tracks
MAX_MISSED_CONFIRMED = 15
MAX_MISSED_VISION = 12

# Gating
RESET_DIST_PX = 100.0
VISION_GATE_BASE_PX = 60.0
VISION_GATE_GAIN_PX = 8.0
TAG_ATTACH_DIST_PX = 80.0

# Vision bootstrap
WARMUP_FRAMES = 40
VISION_CONFIRM_HITS = 4
VISION_MIN_HITS_TO_DRAW = 1

# Blob filters
BLOB_MIN_AREA = 20
BLOB_MAX_AREA = 2000

BLOB_MIN_W = 4
BLOB_MIN_H = 4
BLOB_MAX_W = 100
BLOB_MAX_H = 100

BLOB_MIN_AR = 0.2
BLOB_MAX_AR = 5.0

BORDER_MARGIN = 4

USE_BLOB = False   # <── False = apenas YOLO; True = YOLO + blobs


# YOLO
YOLO_MODEL_PATH = r"C:\Users\Pedru\OneDrive\Área de Trabalho\ControllerRC\best.pt"
YOLO_CONF_THRESH = 0.5
YOLO_DRAW_RADIUS_M = 0.3

# ROI da pista
TRACK_POLYGON = np.array([
    [250, 10],
    [1000, 10],
    [1000, 650],
    [250, 650]
], dtype=np.int32)

# =========================
# TAG DETECTION STATS
# =========================
TARGET_TAG_ID = 2
GRID_W = 20
GRID_H = 15
SAVE_STATS_FILE = "tag_detection_stats.npz"

# =========================
# CÂMARA (GoPro robusta)
# =========================
def _decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


def _is_valid_frame(frame, black_threshold=5, min_std=10):
    if frame is None or frame.size == 0:
        return False
    mean_val = np.mean(frame)
    std_val = np.std(frame)
    return mean_val >= black_threshold and std_val >= min_std


class Camera:
    def __init__(self):
        cfgs = [
            (cv2.CAP_MSMF, "MJPG", "MSMF+MJPG"),
            (cv2.CAP_MSMF, "YUY2", "MSMF+YUY2"),
            (cv2.CAP_DSHOW, "MJPG", "DSHOW+MJPG"),
        ]
        self._cam = None

        for backend, fmt, desc in cfgs:
            print(f"[Camera] A tentar {desc} ...")
            cam = cv2.VideoCapture(OPENCV_CAMERA_INDEX, backend)
            if not cam.isOpened():
                print(f"[Camera]   Falhou.")
                continue

            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            cam.set(cv2.CAP_PROP_FPS, TARGET_FPS)

            for _ in range(10):
                cam.read()

            ret, frame = cam.read()
            if ret and _is_valid_frame(frame):
                actual_fourcc = _decode_fourcc(cam.get(cv2.CAP_PROP_FOURCC))
                print(f"[Camera]   OK → {desc} ({actual_fourcc})")
                self._cam = cam
                break
            else:
                cam.release()
                print("[Camera]   Frame preto/inválido")

        if self._cam is None:
            raise RuntimeError("Erro ao abrir GoPro com qualquer configuração")

        print(f"[INFO] width = {self._cam.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}")
        print(f"[INFO] height = {self._cam.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
        print(f"[INFO] fps = {self._cam.get(cv2.CAP_PROP_FPS):.1f}")

    def capture_bgr(self):
        ret, frame = self._cam.read()
        if not ret or frame is None or not _is_valid_frame(frame):
            raise RuntimeError("Erro ao capturar frame válido")
        return frame

    def stop(self):
        self._cam.release()


# =========================
# UTIL
# =========================
def clamp_dt(dt):
    return max(DT_MIN, min(DT_MAX, dt))


def dist2d(x1, y1, x2, y2):
    return float(np.hypot(x1 - x2, y1 - y2))


def build_track_mask():
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.fillPoly(mask, [TRACK_POLYGON], 255)
    return mask


def point_inside_track(x, y):
    return cv2.pointPolygonTest(TRACK_POLYGON, (float(x), float(y)), False) >= 0


def get_dynamic_vision_gate(track):
    return VISION_GATE_BASE_PX + VISION_GATE_GAIN_PX * track["missed"]


def get_cell(x, y):
    gx = int((x / WIDTH) * GRID_W)
    gy = int((y / HEIGHT) * GRID_H)
    gx = max(0, min(GRID_W - 1, gx))
    gy = max(0, min(GRID_H - 1, gy))
    return gx, gy


def find_best_blob_for_track(track, detections, max_dist=None):
    last_meas = track.get("last_meas")
    if last_meas is not None:
        x_ref, y_ref = float(last_meas[0]), float(last_meas[1])
    else:
        x_ref = float(track.get("x_px", 0.0))
        y_ref = float(track.get("y_px", 0.0))

    if max_dist is None:
        max_dist = get_dynamic_vision_gate(track)

    best_idx = -1
    best_dist = float("inf")

    for i, det in enumerate(detections):
        if det["used"]:
            continue

        cx, cy = det["center"]
        d = dist2d(cx, cy, x_ref, y_ref)

        if d < best_dist and d < max_dist:
            best_dist = d
            best_idx = i

    if best_idx >= 0:
        detections[best_idx]["used"] = True
        detections[best_idx]["match_dist"] = best_dist
        return detections[best_idx]

    return None


def find_best_track_for_tag(x_meas, y_meas, tracks_dict, max_dist=TAG_ATTACH_DIST_PX):
    best_key = None
    best_dist = float("inf")

    for key, tr in tracks_dict.items():
        last_meas = tr.get("last_meas")
        if last_meas is not None:
            x_ref, y_ref = float(last_meas[0]), float(last_meas[1])
        else:
            x_ref = float(tr.get("x_px", 0.0))
            y_ref = float(tr.get("y_px", 0.0))
        d = dist2d(x_meas, y_meas, x_ref, y_ref)

        if d < best_dist and d < max_dist:
            best_dist = d
            best_key = key

    return best_key, best_dist


def build_heatmap_image(raw_map, normalize=True):
    img = raw_map.copy().astype(np.float32)

    if normalize:
        m = img.max()
        if m > 0:
            img = img / m
    else:
        img = np.clip(img, 0.0, 1.0)

    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def build_rate_map_image(detect_map, visit_map):
    rate_map = detect_map / (visit_map + 1e-6)
    return build_heatmap_image(rate_map, normalize=False), rate_map


def resolve_yolo_model_path(path):
    if not path:
        return None

    supported_suffixes = (".pt", ".onnx", ".engine", ".torchscript")

    if os.path.isfile(path) and path.lower().endswith(supported_suffixes):
        return path

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for name in files:
                if name.lower().endswith(supported_suffixes):
                    return os.path.join(root, name)

    return None


def describe_yolo_model_path_issue(path):
    if not path:
        return "caminho YOLO vazio"

    if not os.path.exists(path):
        return f"caminho inexistente: {path}"

    if os.path.isdir(path):
        return (
            "a pasta existe, mas nao contem nenhum ficheiro de modelo suportado "
            "(.pt, .onnx, .engine, .torchscript)"
        )

    return f"ficheiro nao suportado ou inacessivel: {path}"


def reset_detection_stats():
    return {
        "total_frames_count": 0,
        "tag_detect_count_any": 0,
        "tag_detect_count_target": 0,
        "current_gap_target": 0,
        "max_gap_target": 0,
        "visit_map": np.zeros((GRID_H, GRID_W), dtype=np.float64),
        "detect_map": np.zeros((GRID_H, GRID_W), dtype=np.float64),
    }


def save_detection_stats(path, stats):
    np.savez(
        path,
        total_frames_count=stats["total_frames_count"],
        tag_detect_count_any=stats["tag_detect_count_any"],
        tag_detect_count_target=stats["tag_detect_count_target"],
        current_gap_target=stats["current_gap_target"],
        max_gap_target=stats["max_gap_target"],
        visit_map=stats["visit_map"],
        detect_map=stats["detect_map"],
        grid_w=GRID_W,
        grid_h=GRID_H,
        width=WIDTH,
        height=HEIGHT,
        target_tag_id=TARGET_TAG_ID,
    )
    print(f"[INFO] Estatísticas guardadas em {path}")


# =========================
# DETEÇÃO VISUAL (BLOBS)
# =========================
class VehicleBlobDetector:
    def __init__(self, track_mask):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=10,
            detectShadows=False
        )
        self.kernel_open = np.ones((2, 2), np.uint8)
        self.kernel_close = np.ones((3, 3), np.uint8)
        self.track_mask = track_mask

    def detect(self, frame):
        fg = self.bg.apply(frame)

        _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, self.track_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < BLOB_MIN_AREA or area > BLOB_MAX_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if w < BLOB_MIN_W or h < BLOB_MIN_H:
                continue
            if w > BLOB_MAX_W or h > BLOB_MAX_H:
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < BLOB_MIN_AR or aspect_ratio > BLOB_MAX_AR:
                continue

            if x <= BORDER_MARGIN or y <= BORDER_MARGIN:
                continue
            if (x + w) >= (WIDTH - BORDER_MARGIN) or (y + h) >= (HEIGHT - BORDER_MARGIN):
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0

            if not point_inside_track(cx, cy):
                continue

            detections.append({
                "bbox": (x, y, w, h),
                "center": (cx, cy),
                "area": area,
                "used": False,
                "source": "blob",
            })

        return detections, mask


def make_track(x0, y0, t_now, state_label, source, temp_id=None, tag_id=None):
    return {
        "x_px": float(x0),
        "y_px": float(y0),
        "vx_px_s": 0.0,
        "vy_px_s": 0.0,
        "ax_px_s2": 0.0,
        "ay_px_s2": 0.0,
        "t_last": t_now,
        "missed": 0,
        "seen_this_frame": True,
        "state_label": state_label,
        "last_meas": (x0, y0),
        "last_source": source,
        "hits": 1,
        "temp_id": temp_id,
        "tag_id": tag_id,
        "theta_rad": 0.0,
        "raw_theta_rad": 0.0,
        "speed_m_s": 0.0,
        "raw_speed_m_s": 0.0,
        "prev_pos_m": None,
        "prev_raw_pos_m": None,
        "prev_raw_t": None,
        "raw_speed_anchor_pos_m": None,
        "raw_speed_anchor_t": None,
        "raw_speed_cycle_count": 0,
        "raw_theta_anchor_pos_m": None,
        "raw_x_m": None,
        "raw_y_m": None,
        "trail_px": [(float(x0), float(y0))],
    }


def append_track_trail(track, x_px, y_px):
    trail = track.setdefault("trail_px", [])
    point = (float(x_px), float(y_px))

    if trail:
        last_x, last_y = trail[-1]
        if dist2d(last_x, last_y, point[0], point[1]) < TRACK_TRAIL_MIN_STEP_PX:
            trail[-1] = point
        else:
            trail.append(point)
    else:
        trail.append(point)

    if len(trail) > TRACK_TRAIL_MAX_POINTS:
        del trail[:-TRACK_TRAIL_MAX_POINTS]


def prepare_track_for_frame(track):
    track["seen_this_frame"] = False


def update_track_motion(track, x_meas, y_meas, t_now):
    prev_meas = track.get("last_meas")
    prev_t = track.get("t_last")

    if prev_meas is not None and prev_t is not None:
        dt = max(1e-6, float(t_now - prev_t))
        prev_vx = float(track.get("vx_px_s", 0.0))
        prev_vy = float(track.get("vy_px_s", 0.0))
        vx_px_s = (float(x_meas) - float(prev_meas[0])) / dt
        vy_px_s = (float(y_meas) - float(prev_meas[1])) / dt
        ax_px_s2 = (vx_px_s - prev_vx) / dt
        ay_px_s2 = (vy_px_s - prev_vy) / dt
    else:
        vx_px_s = 0.0
        vy_px_s = 0.0
        ax_px_s2 = 0.0
        ay_px_s2 = 0.0

    track["x_px"] = float(x_meas)
    track["y_px"] = float(y_meas)
    track["vx_px_s"] = float(vx_px_s)
    track["vy_px_s"] = float(vy_px_s)
    track["ax_px_s2"] = float(ax_px_s2)
    track["ay_px_s2"] = float(ay_px_s2)


def update_track_with_tag(track, x_meas, y_meas, t_now):
    update_track_motion(track, x_meas, y_meas, t_now)
    track["t_last"] = t_now
    track["missed"] = 0
    track["seen_this_frame"] = True
    track["state_label"] = "MEAS_TAG"
    track["last_meas"] = (x_meas, y_meas)
    track["last_source"] = "tag"
    track["hits"] += 1
    append_track_trail(track, x_meas, y_meas)
    x_m, y_m = px_to_m(x_meas, y_meas)
    update_raw_theta(track, x_m, y_m, t_now)


def update_track_with_vision(track, cx, cy, t_now):
    update_track_motion(track, cx, cy, t_now)
    track["t_last"] = t_now
    track["missed"] = 0
    track["seen_this_frame"] = True
    track["last_meas"] = (cx, cy)
    track["last_source"] = "vision"
    track["hits"] += 1
    append_track_trail(track, cx, cy)

    x_m, y_m = px_to_m(cx, cy)
    update_raw_theta(track, x_m, y_m, t_now)

    if track["tag_id"] is None:
        if track["hits"] >= VISION_CONFIRM_HITS:
            track["state_label"] = "CONFIRMED_VISION"
        else:
            track["state_label"] = "TENTATIVE_VISION"
    else:
        track["state_label"] = "MEAS_VISION"


def get_track_display_pose(track):
    label = str(track.get("state_label", ""))
    last_meas = track.get("last_meas")

    if label == "MEAS_TAG" and last_meas is not None:
        x_draw_px = float(last_meas[0])
        y_draw_px = float(last_meas[1])
        x_draw_m = track["raw_x_m"] if track.get("raw_x_m") is not None else px_to_m(x_draw_px, y_draw_px)[0]
        y_draw_m = track["raw_y_m"] if track.get("raw_y_m") is not None else px_to_m(x_draw_px, y_draw_px)[1]
        return x_draw_px, y_draw_px, float(x_draw_m), float(y_draw_m)

    if label in ("MEAS_VISION", "TENTATIVE_VISION", "CONFIRMED_VISION") and last_meas is not None:
        x_draw_px = float(last_meas[0])
        y_draw_px = float(last_meas[1])
        x_draw_m, y_draw_m = px_to_m(x_draw_px, y_draw_px)
        return x_draw_px, y_draw_px, x_draw_m, y_draw_m

    x_px = float(track.get("x_px", 0.0))
    y_px = float(track.get("y_px", 0.0))
    x_draw_m, y_draw_m = px_to_m(x_px, y_px)
    return x_px, y_px, x_draw_m, y_draw_m


def update_visit_map_from_tracks(stats, confirmed_tracks, vision_tracks):
    for tr in confirmed_tracks.values():
        x = float(tr.get("x_px", 0.0))
        y = float(tr.get("y_px", 0.0))
        if point_inside_track(x, y):
            gx, gy = get_cell(x, y)
            stats["visit_map"][gy, gx] += 1

    for tr in vision_tracks.values():
        x = float(tr.get("x_px", 0.0))
        y = float(tr.get("y_px", 0.0))
        if point_inside_track(x, y):
            gx, gy = get_cell(x, y)
            stats["visit_map"][gy, gx] += 1


def update_detect_map_from_tags(stats, tags):
    for tag in tags:
        tag_id = int(tag.tag_id)
        x_meas, y_meas = map(float, tag.center)

        if tag_id == TARGET_TAG_ID and point_inside_track(x_meas, y_meas):
            gx, gy = get_cell(x_meas, y_meas)
            stats["detect_map"][gy, gx] += 1


# =========================
# SETUP
# =========================
camera = Camera()

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)

track_mask = build_track_mask()
blob_detector = VehicleBlobDetector(track_mask)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("[INFO] A carregar modelo YOLO...")
resolved_yolo_model_path = resolve_yolo_model_path(YOLO_MODEL_PATH)
if resolved_yolo_model_path is None:
    raise RuntimeError(
        f"Modelo YOLO invalido ou nao encontrado: {YOLO_MODEL_PATH}. "
        f"Motivo: {describe_yolo_model_path_issue(YOLO_MODEL_PATH)}"
    )
yolo_model = YOLO(resolved_yolo_model_path)

confirmed_tracks = {}   # key = tag_id
vision_tracks = {}      # key = temp_id (negativos)
next_temp_id = -1000

frame_counter = 0
t_prev = time.perf_counter()

stats = reset_detection_stats()

print("Sistema a correr... prima q para sair")
print("Teclas: q=sair, r=reset stats, s=guardar stats")

# =========================
# LOOP
# =========================
try:
    while True:
        t = time.perf_counter()
        dt_frame = clamp_dt(t - t_prev)
        t_prev = t
        frame_counter += 1
        stats["total_frames_count"] += 1

        frame = camera.capture_bgr()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 0) deteção visual: blobs + YOLO
        # --- DETEÇÃO VISUAL ---
        # 0) DETEÇÃO VISUAL (YOLO + opcionalmente blobs)
        blob_detections = []
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

        if USE_BLOB:
            blob_detections, mask = blob_detector.detect(frame)

        # YOLO deteção (modelo de Deep Learning)
        results = yolo_model(frame)
        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w / 2, y1 + h / 2
                if not point_inside_track(cx, cy):
                    continue
                blob_detections.append({
                    "bbox": (x1, y1, w, h),
                    "center": (cx, cy),
                    "area": w * h,
                    "used": False,
                    "source": "yolo"
                })


        # 1) preparar tracks confirmados
        for tr in confirmed_tracks.values():
            prepare_track_for_frame(tr)

        # 2) preparar tracks visuais
        for tr in vision_tracks.values():
            prepare_track_for_frame(tr)

        # 3) detetar AprilTags
        tags = detector.detect(gray)

        # stats globais de deteção
        if len(tags) > 0:
            stats["tag_detect_count_any"] += 1

        tag_target_seen_this_frame = False
        for tag in tags:
            if int(tag.tag_id) == TARGET_TAG_ID:
                tag_target_seen_this_frame = True
                stats["tag_detect_count_target"] += 1
                break

        if tag_target_seen_this_frame:
            stats["max_gap_target"] = max(stats["max_gap_target"], stats["current_gap_target"])
            stats["current_gap_target"] = 0
        else:
            stats["current_gap_target"] += 1

        # 4) atualizar / criar tracks por tag
        for tag in tags:
            tag_id = int(tag.tag_id)
            x_meas, y_meas = map(float, tag.center)
            x_meas_m, y_meas_m = px_to_m(x_meas, y_meas)
            corners = tag.corners.astype(np.int32)

            if tag_id in confirmed_tracks:
                tr = confirmed_tracks[tag_id]
                last_meas = tr.get("last_meas")
                if last_meas is not None:
                    x_ref, y_ref = float(last_meas[0]), float(last_meas[1])
                else:
                    x_ref = float(tr.get("x_px", 0.0))
                    y_ref = float(tr.get("y_px", 0.0))

                if dist2d(x_meas, y_meas, x_ref, y_ref) <= RESET_DIST_PX or tr["missed"] > 0:
                    update_track_with_tag(tr, x_meas, y_meas, t)
            else:
                temp_key, d_attach = find_best_track_for_tag(
                    x_meas, y_meas, vision_tracks, max_dist=TAG_ATTACH_DIST_PX
                )

                if temp_key is not None:
                    tr_vis = vision_tracks[temp_key]
                    tr_vis["tag_id"] = tag_id
                    update_track_with_tag(tr_vis, x_meas, y_meas, t)
                    confirmed_tracks[tag_id] = tr_vis
                    del vision_tracks[temp_key]
                else:
                    confirmed_tracks[tag_id] = make_track(
                        x_meas, y_meas, t,
                        state_label="MEAS_TAG",
                        source="tag",
                        temp_id=None,
                        tag_id=tag_id
                    )

            theta_raw_deg = 0.0
            if tag_id in confirmed_tracks:
                theta_raw_deg = float(
                    np.degrees(confirmed_tracks[tag_id].get("raw_theta_rad", 0.0))
                )

            if SHOW_WINDOW:
                cv2.polylines(frame, [corners], True, (0, 0, 255), 2)
                cv2.circle(frame, (int(x_meas), int(y_meas)), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"RAW ID:{tag_id} x={x_meas_m:.3f}m y={y_meas_m:.3f}m th={theta_raw_deg:.1f}deg",
                    (int(x_meas) + 10, int(y_meas) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

        vision_ready = (not USE_BLOB) or (frame_counter > WARMUP_FRAMES)

        # 5) fallback por visão para tracks confirmados sem tag neste frame
        if vision_ready:
            for tag_id, tr in confirmed_tracks.items():
                if tr["seen_this_frame"]:
                    continue

                det = find_best_blob_for_track(tr, blob_detections)
                if det is not None:
                    cx, cy = det["center"]
                    update_track_with_vision(tr, cx, cy, t)

        # 6) visão para tracks visuais existentes
        if vision_ready:
            for temp_id, tr in vision_tracks.items():
                if tr["seen_this_frame"]:
                    continue

                det = find_best_blob_for_track(tr, blob_detections)
                if det is not None:
                    cx, cy = det["center"]
                    update_track_with_vision(tr, cx, cy, t)

        # 7) criar novos tracks visuais a partir de blobs não usados
        if vision_ready:
            for det in blob_detections:
                if det["used"]:
                    continue

                cx, cy = det["center"]
                too_close = False

                for tr in confirmed_tracks.values():
                    last_meas = tr.get("last_meas")
                    if last_meas is not None:
                        x_ref, y_ref = float(last_meas[0]), float(last_meas[1])
                    else:
                        x_ref = float(tr.get("x_px", 0.0))
                        y_ref = float(tr.get("y_px", 0.0))
                    if dist2d(cx, cy, x_ref, y_ref) < VISION_GATE_BASE_PX:
                        too_close = True
                        break

                if not too_close:
                    for tr in vision_tracks.values():
                        last_meas = tr.get("last_meas")
                        if last_meas is not None:
                            x_ref, y_ref = float(last_meas[0]), float(last_meas[1])
                        else:
                            x_ref = float(tr.get("x_px", 0.0))
                            y_ref = float(tr.get("y_px", 0.0))
                        if dist2d(cx, cy, x_ref, y_ref) < VISION_GATE_BASE_PX:
                            too_close = True
                            break

                if too_close:
                    continue

                tr_new = make_track(
                    cx, cy, t,
                    state_label="TENTATIVE_VISION",
                    source="vision",
                    temp_id=next_temp_id,
                    tag_id=None
                )
                x_m_new, y_m_new = px_to_m(cx, cy)
                update_raw_theta(tr_new, x_m_new, y_m_new, t)
                vision_tracks[next_temp_id] = tr_new
                det["used"] = True
                next_temp_id -= 1

        # 8) gerir tracks confirmados
        for tag_id in list(confirmed_tracks.keys()):
            tr = confirmed_tracks[tag_id]

            if not tr["seen_this_frame"]:
                tr["missed"] += 1
                tr["t_last"] = t

            if tr["missed"] > MAX_MISSED_CONFIRMED:
                del confirmed_tracks[tag_id]

        # 9) gerir tracks visuais
        for temp_id in list(vision_tracks.keys()):
            tr = vision_tracks[temp_id]

            if not tr["seen_this_frame"]:
                tr["missed"] += 1
                tr["t_last"] = t

            if tr["missed"] > MAX_MISSED_VISION:
                del vision_tracks[temp_id]

        # 10) atualizar mapas espaciais
        update_visit_map_from_tracks(stats, confirmed_tracks, vision_tracks)
        update_detect_map_from_tags(stats, tags)

        # 11) output
        payload = {
            "t": t,
            "frame": frame_counter,
            "tracks": []
        }

        for tag_id, tr in confirmed_tracks.items():
            label = tr["state_label"]
            x_udp_px = float(tr.get("x_px", 0.0))
            y_udp_px = float(tr.get("y_px", 0.0))
            x_m_filt, y_m_filt = px_to_m(x_udp_px, y_udp_px)
            vx = float(tr.get("vx_px_s", 0.0))
            vy = float(tr.get("vy_px_s", 0.0))
            ax = float(tr.get("ax_px_s2", 0.0))
            ay = float(tr.get("ay_px_s2", 0.0))
            vx_m_s, vy_m_s = vel_px_to_m(vx, vy)
            ax_m_s2, ay_m_s2 = acc_px_to_m(ax, ay)
            theta_rad_track, theta_deg_track, speed_m_s_track = update_track_theta(tr)
            last_meas = tr.get("last_meas")
            if last_meas is not None:
                x_udp_px = float(last_meas[0])
                y_udp_px = float(last_meas[1])
                x_m, y_m = px_to_m(x_udp_px, y_udp_px)
            else:
                x_m, y_m = x_m_filt, y_m_filt
            theta_rad = float(tr.get("raw_theta_rad", theta_rad_track))
            theta_deg = float(np.degrees(theta_rad))
            speed_m_s = tr.get("raw_speed_m_s", speed_m_s_track)

            payload["tracks"].append({
                "id": int(tag_id),
                "x_px": float(x_udp_px),
                "y_px": float(y_udp_px),
                "x_m": x_m,
                "y_m": y_m,
                "vx_px_s": float(vx),
                "vy_px_s": float(vy),
                "vx_m_s": vx_m_s,
                "vy_m_s": vy_m_s,
                "ax_px_s2": float(ax),
                "ay_px_s2": float(ay),
                "ax_m_s2": ax_m_s2,
                "ay_m_s2": ay_m_s2,
                "speed_m_s": float(speed_m_s),
                "theta_rad": float(theta_rad),
                "theta_deg": float(theta_deg),
                "mode": label,
                "missed": int(tr["missed"]),
                "source": tr["last_source"]
            })

        for temp_id, tr in vision_tracks.items():
            label = tr["state_label"]
            x_udp_px = float(tr.get("x_px", 0.0))
            y_udp_px = float(tr.get("y_px", 0.0))
            last_meas = tr.get("last_meas")
            if last_meas is not None:
                x_udp_px = float(last_meas[0])
                y_udp_px = float(last_meas[1])
                x_m, y_m = px_to_m(x_udp_px, y_udp_px)
            else:
                x_m, y_m = px_to_m(x_udp_px, y_udp_px)
            vx = float(tr.get("vx_px_s", 0.0))
            vy = float(tr.get("vy_px_s", 0.0))
            ax = float(tr.get("ax_px_s2", 0.0))
            ay = float(tr.get("ay_px_s2", 0.0))
            vx_m_s, vy_m_s = vel_px_to_m(vx, vy)
            ax_m_s2, ay_m_s2 = acc_px_to_m(ax, ay)
            theta_rad_track, theta_deg_track, speed_m_s = update_track_theta(tr)
            theta_rad = float(tr.get("raw_theta_rad", theta_rad_track))
            theta_deg = float(np.degrees(theta_rad))

            payload["tracks"].append({
                "id": int(temp_id),
                "x_px": float(x_udp_px),
                "y_px": float(y_udp_px),
                "x_m": x_m,
                "y_m": y_m,
                "vx_px_s": float(vx),
                "vy_px_s": float(vy),
                "vx_m_s": vx_m_s,
                "vy_m_s": vy_m_s,
                "ax_px_s2": float(ax),
                "ay_px_s2": float(ay),
                "ax_m_s2": ax_m_s2,
                "ay_m_s2": ay_m_s2,
                "speed_m_s": float(speed_m_s),
                "theta_rad": float(theta_rad),
                "theta_deg": float(theta_deg),
                "mode": label,
                "missed": int(tr["missed"]),
                "source": tr["last_source"]
            })

        # 12) draw tracks
        if SHOW_WINDOW:
            cv2.polylines(frame, [TRACK_POLYGON], True, (255, 255, 255), 2)

            if SHOW_REFERENCE_PATH:
                spline_points_m = build_rounded_polyline(
                    MARK_POINTS_M,
                    corner_radius=ROUND_CORNER_RADIUS_M,
                    corner_samples=ROUND_CORNER_SAMPLES,
                    closed=True
                )
                spline_points_px = []
                for sx_m, sy_m in spline_points_m:
                    sx_px, sy_px = m_to_px(sx_m, sy_m)
                    spline_points_px.append([int(round(sx_px)), int(round(sy_px))])

                if len(spline_points_px) >= 2:
                    cv2.polylines(
                        frame,
                        [np.array(spline_points_px, dtype=np.int32)],
                        True,
                        (0, 200, 255),
                        2,
                        cv2.LINE_AA
                    )

            for obstacle in DRAW_OBSTACLES_M:
                ox_px, oy_px = m_to_px(obstacle["x"], obstacle["y"])
                r_px = int(round(obstacle["r"] / PIXEL_TO_METER))
                if 0 <= ox_px < WIDTH and 0 <= oy_px < HEIGHT:
                    cv2.circle(frame, (int(round(ox_px)), int(round(oy_px))), r_px, (255, 140, 0), 2)
                    cv2.circle(frame, (int(round(ox_px)), int(round(oy_px))), 4, (255, 140, 0), -1)
                    cv2.putText(
                        frame,
                        f"OBS r={obstacle['r']:.2f}m",
                        (int(round(ox_px)) + 8, int(round(oy_px)) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 140, 0),
                        1
                    )

            if SHOW_REFERENCE_PATH:
                for i, (mx, my) in enumerate(MARK_POINTS_M):
                    px, py = m_to_px(mx, my)
                    if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                        cv2.circle(frame, (int(px), int(py)), 5, (0, 128, 255), -1)
                        cv2.putText(
                            frame,
                            f"P{i+1}",
                            (int(px) + 6, int(py) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 128, 255),
                            1
                        )

            if SHOW_TRACK_TRAIL:
                for tr in confirmed_tracks.values():
                    trail = tr.get("trail_px", [])
                    if len(trail) >= 2:
                        cv2.polylines(
                            frame,
                            [np.array(np.round(trail), dtype=np.int32)],
                            False,
                            (80, 255, 80),
                            2,
                            cv2.LINE_AA
                        )

                for tr in vision_tracks.values():
                    if tr["hits"] < VISION_MIN_HITS_TO_DRAW:
                        continue

                    trail = tr.get("trail_px", [])
                    if len(trail) >= 2:
                        cv2.polylines(
                            frame,
                            [np.array(np.round(trail), dtype=np.int32)],
                            False,
                            (255, 120, 255),
                            1,
                            cv2.LINE_AA
                        )

            for tag_id, tr in confirmed_tracks.items():
                label = tr["state_label"]
                x_draw_px, y_draw_px, x_m, y_m = get_track_display_pose(tr)
                theta_deg = float(np.degrees(tr.get("raw_theta_rad", 0.0)))
                vx = float(tr.get("vx_px_s", 0.0))
                vy = float(tr.get("vy_px_s", 0.0))
                ax = float(tr.get("ax_px_s2", 0.0))
                ay = float(tr.get("ay_px_s2", 0.0))

                if label == "MEAS_TAG":
                    color = (0, 255, 0)
                elif label == "MEAS_VISION":
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 255)

                cv2.circle(frame, (int(round(x_draw_px)), int(round(y_draw_px))), 6, color, -1)

                cv2.putText(
                    frame,
                    f"ID:{tag_id} {label}",
                    (int(round(x_draw_px)) + 10, int(round(y_draw_px)) - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"x={x_m:.3f}m y={y_m:.3f}m th={theta_deg:.1f}deg",
                    (int(round(x_draw_px)) + 10, int(round(y_draw_px)) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

                cv2.putText(
                    frame,
                    f"vx={vx:.1f} vy={vy:.1f} px/s",
                    (int(round(x_draw_px)) + 10, int(round(y_draw_px)) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

                cv2.putText(
                    frame,
                    f"ax={ax:.1f} ay={ay:.1f} px/s2",
                    (int(round(x_draw_px)) + 10, int(round(y_draw_px)) + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

            for temp_id, tr in vision_tracks.items():
                if tr["hits"] < VISION_MIN_HITS_TO_DRAW:
                    continue

                label = tr["state_label"]
                x_draw_px, y_draw_px, x_m, y_m = get_track_display_pose(tr)
                theta_deg = float(np.degrees(tr.get("raw_theta_rad", 0.0)))
                vx = float(tr.get("vx_px_s", 0.0))
                vy = float(tr.get("vy_px_s", 0.0))
                ax = float(tr.get("ax_px_s2", 0.0))
                ay = float(tr.get("ay_px_s2", 0.0))

                if label == "TENTATIVE_VISION":
                    color = (255, 0, 255)
                elif label == "CONFIRMED_VISION":
                    color = (255, 255, 0)
                else:
                    color = (0, 165, 255)

                cv2.circle(frame, (int(round(x_draw_px)), int(round(y_draw_px))), 5, color, -1)

                cv2.putText(
                    frame,
                    f"TMP:{abs(temp_id)} {label}",
                    (int(round(x_draw_px)) + 10, int(round(y_draw_px)) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"x={x_m:.3f}m y={y_m:.3f}m th={theta_deg:.1f}deg",
                    (int(round(x_draw_px)) + 10, int(round(y_draw_px)) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1
                )

            for det in blob_detections:
                x, y, w, h = det["bbox"]
                cx, cy = det["center"]
                src = det.get("source", "blob")

                if src == "yolo":
                    base_color = (0, 255, 0)
                else:
                    base_color = (255, 0, 0)

                blob_color = base_color if not det["used"] else (0, 255, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), blob_color, 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, blob_color, -1)
                if src == "yolo":
                    yolo_radius_px = int(round(YOLO_DRAW_RADIUS_M / PIXEL_TO_METER))
                    cv2.circle(frame, (int(cx), int(cy)), yolo_radius_px, blob_color, 2)
                cv2.putText(
                    frame,
                    f"{src.upper()} a={det['area']:.0f}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    blob_color,
                    1
                )

        if SEND_UDP and payload["tracks"]:
            sock.sendto(json.dumps(payload).encode("utf-8"), (PC_IP, PC_PORT))

        # 13) HUD
        if SHOW_WINDOW:
            fps = 1.0 / dt_frame if dt_frame > 0 else 0.0

            rate_any = stats["tag_detect_count_any"] / max(1, stats["total_frames_count"])
            rate_target = stats["tag_detect_count_target"] / max(1, stats["total_frames_count"])

            cv2.putText(
                frame,
                f"FPS:{fps:.1f}  tags:{len(tags)}  detections:{len(blob_detections)}  conf:{len(confirmed_tracks)}  vis:{len(vision_tracks)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"tag_any_rate={rate_any:.3f}  tag_{TARGET_TAG_ID}_rate={rate_target:.3f}  max_gap={stats['max_gap_target']}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.imshow("Hybrid Tracking", frame)
            if SHOW_MASK:
                cv2.imshow("Blob Mask", mask)

            if SHOW_HEATMAPS:
                visit_img = build_heatmap_image(stats["visit_map"], normalize=True)
                detect_img = build_heatmap_image(stats["detect_map"], normalize=True)
                rate_img, _ = build_rate_map_image(stats["detect_map"], stats["visit_map"])

                cv2.putText(
                    visit_img,
                    "VISIT MAP",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    detect_img,
                    "DETECT MAP",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    rate_img,
                    "DETECTION RATE MAP",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Visit Heatmap", visit_img)
                cv2.imshow("Detect Heatmap", detect_img)
                cv2.imshow("Detection Rate Heatmap", rate_img)

            if stats["total_frames_count"] % 100 == 0:
                print(
                    f"[STATS] frames={stats['total_frames_count']}  "
                    f"tag_any_rate={rate_any:.3f}  "
                    f"tag_{TARGET_TAG_ID}_rate={rate_target:.3f}  "
                    f"max_gap={stats['max_gap_target']}"
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                stats = reset_detection_stats()
                print("[INFO] Estatísticas resetadas")
            elif key == ord("s"):
                save_detection_stats(SAVE_STATS_FILE, stats)

except KeyboardInterrupt:
    pass

finally:
    stats["max_gap_target"] = max(stats["max_gap_target"], stats["current_gap_target"])
    print("\n===== RESUMO FINAL =====")
    print("frames totais =", stats["total_frames_count"])
    print("tag any rate =", stats["tag_detect_count_any"] / max(1, stats["total_frames_count"]))
    print(f"tag {TARGET_TAG_ID} rate =", stats["tag_detect_count_target"] / max(1, stats["total_frames_count"]))
    print("max gap target =", stats["max_gap_target"])

    camera.stop()
    sock.close()
    cv2.destroyAllWindows()
