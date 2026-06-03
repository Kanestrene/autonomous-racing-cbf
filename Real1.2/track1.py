"""
Tracking a (blue) car through a live video feed

Input: Live video feed through a USB camera

Output: Video with the car's current location, as well as the lap number, lap time and best lap time overall

Lap info is displayed in a bottom border

"""
import cv2
import numpy as np
import os
import time
import json
import socket
import re
from pathlib import Path
from shared_config import (
    BARRIER_INNER_M,
    BARRIER_OUTER_M,
    BARRIER_ROUND_CORNER_RADIUS_M,
    BARRIER_ROUND_CORNER_SAMPLES,
    BLUE_HSV_LOWER,
    BLUE_HSV_UPPER,
    OBSTACLES_M,
    ROBOT_ELLIPSE_A_M,
    ROBOT_ELLIPSE_B_M,
    TRACK_POINTS_M,
)
try:
    from obstacle_utils import obstacle_angle, obstacle_axes, obstacle_is_ellipse
except ImportError:
    from .obstacle_utils import obstacle_angle, obstacle_axes, obstacle_is_ellipse

#Global Variables
kernel = np.ones((3,3),np.uint8)
contour = None
lap_number = 0 
lap_start_time = 0 
lap_time = 1001
batota = 0
best_lap_time = 1000
#Aux variables
aux = 0 #flag for passing the goal
f_red = 0 #flag for going out of bounds
red = None
click_count = 0
edit_drag_target = None
edited_points_dirty = False

POINT_PICK_RADIUS_PX = 14
SHARED_CONFIG_PATH = Path(__file__).with_name("shared_config.py")

# Crop constants
frame_height = 0
frame_width = 0
raw_frame_height = 0
raw_frame_width = 0
CROP_HEIGHT_FACTOR = 8
CROP_WIDTH_START = None
CROP_WIDTH_END = None
RESIZE_FACTOR = 0.9
INVERT_IMAGE = True
INVERT_FLIP_CODE = -1  # 1=horizontal, 0=vertical, -1=ambas
TRACK_TRAIL_MAX_POINTS = 120
TRACK_TRAIL_MIN_STEP_PX = 6.0
track_trail = []
ROUND_CORNER_RADIUS_M = 0.2
ROUND_CORNER_SAMPLES = 24
RAW_SPEED_UPDATE_CYCLES = 1
RAW_THETA_MIN_UPDATE_M = 0.03

raw_theta_rad = 3.14
raw_speed_m_s = 0.0
raw_speed_anchor_pos_m = None
raw_speed_anchor_t = None
raw_speed_cycle_count = 0
raw_theta_anchor_pos_m = None
raw_x_m = None
raw_y_m = None

# Referencial em metros (mesmo do cam2.py, antes das transformacoes do frame)
REF_PX_ORIGIN_RAW = (616.0, 600.0)
REF_PX_X_AXIS_RAW = (892.0, 600.0)
REF_X_METERS = 0.9
REF_PX_ORIGIN = None
REF_PX_X_AXIS = None
PIXEL_TO_METER = None

#Finishing line coordinates
FINISHING_LINE_MIN_X = None
FINISHING_LINE_MAX_X = None
FINISHING_LINE_MIN_Y = None
FINISHING_LINE_MAX_Y = None
border_height = 40
rectangle_width = 0

# Color range 
#LOWER_BLUE = np.array([100, 100, 50])
#UPPER_BLUE = np.array([140, 255, 255])

LOWER_BLUE = np.array(BLUE_HSV_LOWER, dtype=np.uint8)
UPPER_BLUE = np.array(BLUE_HSV_UPPER, dtype=np.uint8)

#Text constants
font = cv2.FONT_HERSHEY_SIMPLEX #font
fontScale = 0.5
thickness = 1
#blend alpha and beta
alpha = 0.3
beta = 1.0 - alpha

# UDP output
PC_IP = "127.0.0.1"
PC_PORT = 5005
SEND_UDP = True
TRACK_ID = 2

#Get the overlay image
#Parameters: the shape of the video feed
def overlay(frame_shape):
    """
    Load the red overlay
    """
    global red

    if red is None:
        red = cv2.imread("Images/red.png")
        if red is not None:
            red = cv2.resize(red, (frame_shape[1], frame_shape[0]))


def transform_raw_point_to_processed(x_raw, y_raw):
    x_proc = float(x_raw) - float(CROP_WIDTH_START)
    y_proc = float(y_raw)

    x_proc *= RESIZE_FACTOR
    y_proc *= RESIZE_FACTOR

    x_proc = float(frame_width - 1) - x_proc
    y_proc = float(frame_height - 1) - y_proc

    if INVERT_IMAGE:
        if INVERT_FLIP_CODE in (1, -1):
            x_proc = float(frame_width - 1) - x_proc
        if INVERT_FLIP_CODE in (0, -1):
            y_proc = float(frame_height - 1) - y_proc

    return float(x_proc), float(y_proc)


def update_metric_reference():
    global REF_PX_ORIGIN, REF_PX_X_AXIS, PIXEL_TO_METER

    if CROP_WIDTH_START is None or frame_width <= 0 or frame_height <= 0:
        return

    REF_PX_ORIGIN = transform_raw_point_to_processed(*REF_PX_ORIGIN_RAW)
    REF_PX_X_AXIS = transform_raw_point_to_processed(*REF_PX_X_AXIS_RAW)

    dx_px = REF_PX_X_AXIS[0] - REF_PX_ORIGIN[0]
    if abs(dx_px) < 1e-9:
        raise ValueError("Referencia invalida apos transformacao do frame.")

    PIXEL_TO_METER = REF_X_METERS / dx_px


def px_to_m(x_px, y_px):
    if REF_PX_ORIGIN is None or PIXEL_TO_METER is None:
        return float("nan"), float("nan")

    x_m = (float(x_px) - REF_PX_ORIGIN[0]) * PIXEL_TO_METER
    y_m = (REF_PX_ORIGIN[1] - float(y_px)) * PIXEL_TO_METER
    return float(x_m), float(y_m)


def wrap_angle_pi(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def angle_diff_abs(a, b):
    return abs(wrap_angle_pi(a - b))


def update_raw_theta(x_m, y_m, t_now):
    global raw_theta_rad, raw_speed_m_s, raw_speed_anchor_pos_m, raw_speed_anchor_t
    global raw_speed_cycle_count, raw_theta_anchor_pos_m, raw_x_m, raw_y_m

    raw_x_m = float(x_m)
    raw_y_m = float(y_m)

    if raw_speed_anchor_pos_m is None or raw_speed_anchor_t is None:
        raw_speed_anchor_pos_m = (x_m, y_m)
        raw_speed_anchor_t = float(t_now)
        raw_speed_cycle_count = 0
    else:
        raw_speed_cycle_count += 1
        if raw_speed_cycle_count >= RAW_SPEED_UPDATE_CYCLES:
            dx_speed = x_m - raw_speed_anchor_pos_m[0]
            dy_speed = y_m - raw_speed_anchor_pos_m[1]
            dt_speed = max(1e-6, float(t_now - raw_speed_anchor_t))
            raw_speed_m_s = float(np.hypot(dx_speed, dy_speed)) / dt_speed
            raw_speed_anchor_pos_m = (x_m, y_m)
            raw_speed_anchor_t = float(t_now)
            raw_speed_cycle_count = 0

    if raw_theta_anchor_pos_m is None:
        raw_theta_anchor_pos_m = (x_m, y_m)
    else:
        dx = x_m - raw_theta_anchor_pos_m[0]
        dy = y_m - raw_theta_anchor_pos_m[1]
        step_m = float(np.hypot(dx, dy))

        if step_m >= RAW_THETA_MIN_UPDATE_M:
            theta_raw = float(np.arctan2(dy, dx))
            theta_alt = wrap_angle_pi(theta_raw + np.pi)

            if angle_diff_abs(theta_raw, raw_theta_rad) <= angle_diff_abs(theta_alt, raw_theta_rad):
                raw_theta_rad = theta_raw
            else:
                raw_theta_rad = theta_alt

            raw_theta_anchor_pos_m = (x_m, y_m)

    return float(raw_theta_rad)


def m_to_px(x_m, y_m):
    if REF_PX_ORIGIN is None or PIXEL_TO_METER is None:
        return float("nan"), float("nan")

    x_px = REF_PX_ORIGIN[0] + (float(x_m) / PIXEL_TO_METER)
    y_px = REF_PX_ORIGIN[1] - (float(y_m) / PIXEL_TO_METER)
    return float(x_px), float(y_px)


def build_rounded_polyline(points, corner_radius=0.08, corner_samples=10, closed=True):
    n = len(points)
    if n < 2:
        return list(points)
    if n == 2:
        return list(points)

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
        out.append((float(p_start[0]), float(p_start[1])))
        for t in np.linspace(0.0, 1.0, corner_samples + 2)[1:-1]:
            pt = ((1.0 - t) ** 2) * p_start + 2.0 * (1.0 - t) * t * p_curr + (t ** 2) * p_end
            out.append((float(pt[0]), float(pt[1])))
        out.append((float(p_end[0]), float(p_end[1])))

    return out


def append_track_trail(x_px, y_px):
    point = (float(x_px), float(y_px))

    if track_trail:
        last_x, last_y = track_trail[-1]
        if np.hypot(point[0] - last_x, point[1] - last_y) < TRACK_TRAIL_MIN_STEP_PX:
            track_trail[-1] = point
        else:
            track_trail.append(point)
    else:
        track_trail.append(point)

    if len(track_trail) > TRACK_TRAIL_MAX_POINTS:
        del track_trail[:-TRACK_TRAIL_MAX_POINTS]


def draw_track_trail(img):
    if len(track_trail) >= 2:
        cv2.polylines(
            img,
            [np.array(np.round(track_trail), dtype=np.int32)],
            False,
            (0, 200, 255),
            2,
            cv2.LINE_AA
        )


def draw_reference_path(img):
    if REF_PX_ORIGIN is None or PIXEL_TO_METER is None:
        return

    path_points_m = list(TRACK_POINTS_M)
    spline_points_m = build_rounded_polyline(
        path_points_m,
        corner_radius=ROUND_CORNER_RADIUS_M,
        corner_samples=ROUND_CORNER_SAMPLES,
        closed=True
    )

    spline_points_px = []
    for sx_m, sy_m in spline_points_m:
        sx_px, sy_px = m_to_px(sx_m, sy_m)
        if np.isfinite(sx_px) and np.isfinite(sy_px):
            spline_points_px.append([int(round(sx_px)), int(round(sy_px))])

    if len(spline_points_px) >= 2:
        cv2.polylines(
            img,
            [np.array(spline_points_px, dtype=np.int32)],
            True,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    for sx_m, sy_m in TRACK_POINTS_M:
        sx_px, sy_px = m_to_px(sx_m, sy_m)
        if np.isfinite(sx_px) and np.isfinite(sy_px):
            cv2.circle(
                img,
                (int(round(sx_px)), int(round(sy_px))),
                4,
                (255, 0, 0),
                -1,
                cv2.LINE_AA
            )


def _points_by_name(points_name):
    if points_name == "track":
        return TRACK_POINTS_M
    if points_name == "inner":
        return BARRIER_INNER_M
    if points_name == "outer":
        return BARRIER_OUTER_M
    return None


def _round_edit_coord(value):
    value = round(float(value), 2)
    if abs(value) < 0.005:
        value = 0.0
    return value


def _find_nearest_control_point(x_px, y_px):
    if REF_PX_ORIGIN is None or PIXEL_TO_METER is None:
        return None

    best = None
    best_dist = float("inf")
    px_per_meter = abs(1.0 / PIXEL_TO_METER)

    for points_name, points in (
        ("track", TRACK_POINTS_M),
        ("inner", BARRIER_INNER_M),
        ("outer", BARRIER_OUTER_M),
    ):
        for idx, (x_m, y_m) in enumerate(points):
            px, py = m_to_px(x_m, y_m)
            if not (np.isfinite(px) and np.isfinite(py)):
                continue

            dist = float(np.hypot(float(x_px) - px, float(y_px) - py))
            if dist <= POINT_PICK_RADIUS_PX and dist < best_dist:
                best_dist = dist
                best = (points_name, idx)

    for idx, obstacle in enumerate(OBSTACLES_M):
        ox_px, oy_px = m_to_px(float(obstacle["x"]), float(obstacle["y"]))
        if not (np.isfinite(ox_px) and np.isfinite(oy_px)):
            continue

        pick_radius_m = max(obstacle_axes(obstacle))
        pick_radius_px = max(POINT_PICK_RADIUS_PX, int(round(pick_radius_m * px_per_meter)))
        dist = float(np.hypot(float(x_px) - ox_px, float(y_px) - oy_px))
        if dist <= pick_radius_px and dist < best_dist:
            best_dist = dist
            best = ("obstacle", idx)

    return best


def _move_selected_control_point(x_px, y_px):
    global edited_points_dirty

    if edit_drag_target is None:
        return

    if y_px < 0 or y_px >= frame_height or x_px < 0 or x_px >= frame_width:
        return

    x_m, y_m = px_to_m(x_px, y_px)
    if not (np.isfinite(x_m) and np.isfinite(y_m)):
        return

    points_name, idx = edit_drag_target
    if points_name == "obstacle":
        obstacle = OBSTACLES_M[idx]
        obstacle["x"] = _round_edit_coord(x_m)
        obstacle["y"] = _round_edit_coord(y_m)
        edited_points_dirty = True
        return

    points = _points_by_name(points_name)
    if points is None:
        return

    points[idx] = (
        _round_edit_coord(x_m),
        _round_edit_coord(y_m),
    )
    edited_points_dirty = True


def edit_control_points(event, x, y, flags, param):
    global edit_drag_target

    if event == cv2.EVENT_LBUTTONDOWN:
        edit_drag_target = _find_nearest_control_point(x, y)
        _move_selected_control_point(x, y)
    elif event == cv2.EVENT_MOUSEMOVE and edit_drag_target is not None:
        _move_selected_control_point(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _move_selected_control_point(x, y)
        edit_drag_target = None


def _format_points_block(points):
    lines = ["["]
    for x_m, y_m in points:
        lines.append(f"    ({float(x_m):.2f}, {float(y_m):.2f}),")
    lines.append("]")
    return "\n".join(lines)


def _format_obstacles_block(obstacles):
    lines = ["["]
    for obstacle in obstacles:
        if obstacle_is_ellipse(obstacle):
            a_m, b_m = obstacle_axes(obstacle)
            theta = obstacle_angle(obstacle)
            lines.append(
                "    "
                f'{{"x": {float(obstacle["x"]):.2f}, '
                f'"y": {float(obstacle["y"]):.2f}, '
                f'"a": {a_m:.2f}, '
                f'"b": {b_m:.2f}, '
                f'"theta": {theta:.2f}}},'
            )
        else:
            r_m = obstacle_axes(obstacle)[0]
            lines.append(
                "    "
                f'{{"x": {float(obstacle["x"]):.2f}, '
                f'"y": {float(obstacle["y"]):.2f}, '
                f'"r": {r_m:.2f}}},'
            )
    lines.append("]")
    return "\n".join(lines)


def _replace_points_block(text, name, points, followed_by=None):
    replacement = f"{name} = {_format_points_block(points)}"
    pattern = rf"{name}\s*=\s*\[\n.*?\n\]"
    if followed_by is not None:
        pattern += rf"(?=\s*{followed_by})"
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Nao encontrei o bloco {name} em {SHARED_CONFIG_PATH.name}.")
    return text


def _replace_points_block_before_marker(text, name, points, marker):
    marker_idx = text.find(marker)
    if marker_idx < 0:
        raise RuntimeError(f"Nao encontrei o marcador {marker} em {SHARED_CONFIG_PATH.name}.")

    block_start = text.rfind(name, 0, marker_idx)
    if block_start < 0:
        raise RuntimeError(f"Nao encontrei o bloco ativo {name} em {SHARED_CONFIG_PATH.name}.")

    block_end = text.find("\n]", block_start, marker_idx)
    if block_end < 0:
        raise RuntimeError(f"Nao encontrei o fim do bloco ativo {name} em {SHARED_CONFIG_PATH.name}.")
    block_end += len("\n]")

    replacement = f"{name} = {_format_points_block(points)}"
    return text[:block_start] + replacement + text[block_end:]


def _replace_obstacles_block(text, name, obstacles):
    replacement = f"{name} = {_format_obstacles_block(obstacles)}"
    pattern = rf"{name}\s*=\s*\[\n.*?\n\]"
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Nao encontrei o bloco {name} em {SHARED_CONFIG_PATH.name}.")
    return text


def save_control_points_to_shared_config():
    global edited_points_dirty

    text = SHARED_CONFIG_PATH.read_text(encoding="utf-8")
    text = _replace_points_block_before_marker(
        text,
        "TRACK_POINTS_M",
        TRACK_POINTS_M,
        marker="BARRIER_ROUND_CORNER_RADIUS_M",
    )
    text = _replace_points_block(text, "BARRIER_INNER_M", BARRIER_INNER_M)
    text = _replace_points_block(text, "BARRIER_OUTER_M", BARRIER_OUTER_M)
    text = _replace_obstacles_block(text, "OBSTACLES_M", OBSTACLES_M)
    SHARED_CONFIG_PATH.write_text(text, encoding="utf-8")
    edited_points_dirty = False
    print(f"Pontos e obstaculos guardados em {SHARED_CONFIG_PATH.name}.")


def handle_tracking_key(key):
    if key in (ord("q"), ord("Q")):
        if edited_points_dirty:
            print("Tens alteracoes por guardar. Carrega em S para guardar, ou Esc para sair sem guardar.")
            return False
        return True

    if key == 27:
        return True

    if key in (ord("s"), ord("S")):
        save_control_points_to_shared_config()

    return False


def draw_barrier(img, barrier_points_m, color):
    if REF_PX_ORIGIN is None or PIXEL_TO_METER is None:
        return

    rounded_points_m = build_rounded_polyline(
        list(barrier_points_m),
        corner_radius=BARRIER_ROUND_CORNER_RADIUS_M,
        corner_samples=BARRIER_ROUND_CORNER_SAMPLES,
        closed=True
    )

    barrier_points_px = []
    for sx_m, sy_m in rounded_points_m:
        sx_px, sy_px = m_to_px(sx_m, sy_m)
        if np.isfinite(sx_px) and np.isfinite(sy_px):
            barrier_points_px.append([int(round(sx_px)), int(round(sy_px))])

    if len(barrier_points_px) >= 2:
        points = np.array(barrier_points_px, dtype=np.int32)
        cv2.polylines(img, [points], True, color, 2, cv2.LINE_AA)

    for sx_m, sy_m in barrier_points_m:
        sx_px, sy_px = m_to_px(sx_m, sy_m)
        if np.isfinite(sx_px) and np.isfinite(sy_px):
            cv2.circle(
                img,
                (int(round(sx_px)), int(round(sy_px))),
                3,
                color,
                -1,
                cv2.LINE_AA
            )


def draw_barriers(img):
    draw_barrier(img, BARRIER_INNER_M, (0, 255, 0))
    draw_barrier(img, BARRIER_OUTER_M, (0, 0, 255))


def draw_obstacles(img):
    if REF_PX_ORIGIN is None or PIXEL_TO_METER is None:
        return

    px_per_meter = abs(1.0 / PIXEL_TO_METER)

    for obstacle in OBSTACLES_M:
        ox_m = float(obstacle["x"])
        oy_m = float(obstacle["y"])
        a_m, b_m = obstacle_axes(obstacle)

        ox_px, oy_px = m_to_px(ox_m, oy_m)
        axes = (
            max(1, int(round(a_m * px_per_meter))),
            max(1, int(round(b_m * px_per_meter))),
        )

        if not (np.isfinite(ox_px) and np.isfinite(oy_px)):
            continue

        center = (int(round(ox_px)), int(round(oy_px)))
        angle_deg = -float(np.degrees(obstacle_angle(obstacle)))
        cv2.ellipse(img, center, axes, angle_deg, 0, 360, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.circle(img, center, 2, (0, 140, 255), -1, cv2.LINE_AA)


def draw_robot_ellipse(img, x_m, y_m, theta_rad, color):
    if REF_PX_ORIGIN is None or PIXEL_TO_METER is None:
        return

    cx_px, cy_px = m_to_px(x_m, y_m)
    if not (np.isfinite(cx_px) and np.isfinite(cy_px)):
        return

    px_per_meter = abs(1.0 / PIXEL_TO_METER)
    axes = (
        max(1, int(round(float(ROBOT_ELLIPSE_A_M) * px_per_meter))),
        max(1, int(round(float(ROBOT_ELLIPSE_B_M) * px_per_meter))),
    )
    center = (int(round(cx_px)), int(round(cy_px)))
    angle_deg = -float(np.degrees(theta_rad))
    cv2.ellipse(img, center, axes, angle_deg, 0, 360, color, 2, cv2.LINE_AA)


def build_udp_payload(frame_counter, x_px, y_px, x_m, y_m):
    theta_rad = float(raw_theta_rad)
    theta_deg = float(np.degrees(theta_rad))
    speed_m_s = float(raw_speed_m_s)

    return {
        "t": time.time(),
        "frame": int(frame_counter),
        "tracks": [{
            "id": int(TRACK_ID),
            "x_px": float(x_px),
            "y_px": float(y_px),
            "x_m": float(x_m),
            "y_m": float(y_m),
            "vx_px_s": 0.0,
            "vy_px_s": 0.0,
            "vx_m_s": 0.0,
            "vy_m_s": 0.0,
            "ax_px_s2": 0.0,
            "ay_px_s2": 0.0,
            "ax_m_s2": 0.0,
            "ay_m_s2": 0.0,
            "speed_m_s": speed_m_s,
            "theta_rad": theta_rad,
            "theta_deg": theta_deg,
            "mode": "MEAS_VISION",
            "missed": 0,
            "source": "track1"
        }]
    }

#Process the frame
#Parameters: Frame to process
#Returns: The frame cropped and rotated
def process_frame(frame):

    global CROP_WIDTH_START, CROP_WIDTH_END, frame_width, frame_height, rectangle_width, raw_frame_width, raw_frame_height
    
    raw_frame_height, raw_frame_width = frame.shape[:2]
    frame_height, frame_width = frame.shape[:2]

    #Get crop dimensions
    if CROP_WIDTH_START is None:
        CROP_WIDTH_START = frame_width//3 - 160
        CROP_WIDTH_END = frame_width//3*2 + 150
    
    #Crop and rotate the frame
    frame = frame[:frame_height//10*8 + 75, CROP_WIDTH_START:CROP_WIDTH_END]
    
    #Resize and rotate
    frame = cv2.resize(frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    if INVERT_IMAGE:
        frame = cv2.flip(frame, INVERT_FLIP_CODE)

    #Define border constants
    frame_height, frame_width = frame.shape[:2]
    rectangle_width = frame_width//3
    update_metric_reference()
    
    return frame

#Apply bottom border to the window
#Parameters: window with car centre and a list to lap info
#Return: Img with border and lap info
def apply_border(img, phrases):

    global border_height, frame_height, frame_width, rectangle_width

    #Add bottom border
    img = cv2.copyMakeBorder(img, 0, border_height, 0 , 0,
                             cv2.BORDER_CONSTANT, (0, 0, 0))

    for i in range(3):

        x_start = i * rectangle_width
        if i < 2:
            x_end = (i + 1) * rectangle_width
        else:
            frame_width - 2  # Last section stops at the end
    
        top_left_corner = (x_start, frame_height)
        bottom_right_corner = (x_end, frame_height + border_height - 2)
        
        #Draw rectangle
        cv2.rectangle(img, top_left_corner, bottom_right_corner,
                      (255, 255, 255), 2)
        
        # Put text inside rectangle
        (text_width, text_height), _ = cv2.getTextSize(phrases[i], font, fontScale, thickness)

        #Calculate centered text position
        text_x = x_start + (rectangle_width - text_width) // 2
        text_y = frame_height + (border_height + text_height) // 2 - 4

        #Add text
        cv2.putText(img, phrases[i], (text_x, text_y), font, fontScale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    return img

#Extracts the track's edges
#Parameters: Frame with the track
#Returns: The limits of the track and the same frame grayscaled, but with the edges found highlighted in white
def limits (frame):

    if frame is None:
        return -1

    #Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Apply histogram equalization to boost contrast
    gray = cv2.equalizeHist(gray)
    #Apply blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    #Use Canny edge detection
    edged = cv2.Canny(blurred, 70, 170, apertureSize=3)

    #Close all contours
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    #Find all contours
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame, None

    #Extract largest contour
    largest = max(contours, key=cv2.contourArea)

    #Draw on frame copy
    img = frame.copy()
    cv2.drawContours(img, [largest], -1, (0, 0, 255), 2)
        
    return img, largest


def _format_hsv_tuple(values):
    return f"({int(values[0])}, {int(values[1])}, {int(values[2])})"


def _upsert_config_constant(text, name, value_text, after_name):
    replacement = f"{name} = {value_text}"
    pattern = rf"^{name}\s*=\s*[\(\[][^\n]*[\)\]]"
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
    if count == 1:
        return text

    after_pattern = rf"^({after_name}\s*=\s*[^\n]*\n)"
    text, count = re.subn(
        after_pattern,
        lambda match: match.group(1) + replacement + "\n",
        text,
        count=1,
        flags=re.M,
    )
    if count == 1:
        return text

    return text.rstrip() + "\n\n" + replacement + "\n"


def save_blue_hsv_to_shared_config(lower, upper):
    text = SHARED_CONFIG_PATH.read_text(encoding="utf-8")
    text = _upsert_config_constant(
        text,
        "BLUE_HSV_LOWER",
        _format_hsv_tuple(lower),
        after_name="ROBOT_ELLIPSE_B_M",
    )
    text = _upsert_config_constant(
        text,
        "BLUE_HSV_UPPER",
        _format_hsv_tuple(upper),
        after_name="BLUE_HSV_LOWER",
    )
    SHARED_CONFIG_PATH.write_text(text, encoding="utf-8")
    print(f"HSV azul guardado em {SHARED_CONFIG_PATH.name}.")


def calibrate_blue_color(cap):
    global LOWER_BLUE, UPPER_BLUE

    window_name = "Calibrar azul"

    def _nothing(_value):
        pass

    cv2.namedWindow(window_name)
    cv2.createTrackbar("H min", window_name, int(LOWER_BLUE[0]), 179, _nothing)
    cv2.createTrackbar("H max", window_name, int(UPPER_BLUE[0]), 179, _nothing)
    cv2.createTrackbar("S min", window_name, int(LOWER_BLUE[1]), 255, _nothing)
    cv2.createTrackbar("S max", window_name, int(UPPER_BLUE[1]), 255, _nothing)
    cv2.createTrackbar("V min", window_name, int(LOWER_BLUE[2]), 255, _nothing)
    cv2.createTrackbar("V max", window_name, int(UPPER_BLUE[2]), 255, _nothing)

    print("Calibracao azul: ajusta os seletores HSV ate a mascara apanhar so o carro.")
    print("Enter/Espaco confirma os valores. Q sai.")

    while True:
        ret, frame = cap.read()
        if not ret:
            check(cap)

        frame = process_frame(frame)
        h_min = cv2.getTrackbarPos("H min", window_name)
        h_max = cv2.getTrackbarPos("H max", window_name)
        s_min = cv2.getTrackbarPos("S min", window_name)
        s_max = cv2.getTrackbarPos("S max", window_name)
        v_min = cv2.getTrackbarPos("V min", window_name)
        v_max = cv2.getTrackbarPos("V max", window_name)

        lower = np.array([
            min(h_min, h_max),
            min(s_min, s_max),
            min(v_min, v_max),
        ], dtype=np.uint8)
        upper = np.array([
            max(h_min, h_max),
            max(s_min, s_max),
            max(v_min, v_max),
        ], dtype=np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        preview = np.hstack((frame, masked))

        cv2.putText(
            preview,
            "Original | Mascara HSV    Enter/Espaco: confirmar | Q: sair",
            (10, 25),
            font,
            fontScale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            cv2.destroyWindow(window_name)
            cap.release()
            cv2.destroyAllWindows()
            exit()

        if key in (13, 32):
            LOWER_BLUE = lower
            UPPER_BLUE = upper
            save_blue_hsv_to_shared_config(LOWER_BLUE, UPPER_BLUE)
            cv2.destroyWindow(window_name)
            print(f"Azul calibrado: lower={LOWER_BLUE.tolist()} upper={UPPER_BLUE.tolist()}")
            return


#Get coordinates of the finishing line
def set_finishing_line(event, x, y, flags, param):

    global click_count, FINISHING_LINE_MIN_X, FINISHING_LINE_MAX_X, FINISHING_LINE_MIN_Y, FINISHING_LINE_MAX_Y

    if event == cv2.EVENT_LBUTTONDOWN:

        #If this is first set of coordinates
        if click_count == 0:

            FINISHING_LINE_MIN_X = x
            FINISHING_LINE_MIN_Y = y
            click_count += 1
            print("Select the bottom right corner of the finishing line")
        
        #If this is the second set of coordinates        
        elif click_count == 1:

            FINISHING_LINE_MAX_X = x
            FINISHING_LINE_MAX_Y = y
            click_count += 1

#Extracts the car contour
#Parameters: Frame with the car in view, draw_flag -> 0 if visual representation not needed. draw_flag -> 1 otherwise
#Returns: Car's contour and (if draw_flag = 1) the same frame with the contour drawn on
def carcontour(frame, draw_flag=0):

    if frame is None:
        return None if not draw_flag else (None,None)

    #Convert into HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #Binary image
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    #Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None if not draw_flag else (None,None)

    #Get largest contour
    largest = max(contours, key=cv2.contourArea)

    #Draw on image
    if draw_flag:
        img = frame.copy()
        cv2.drawContours(img, [largest], -1, (0, 252, 124), 2)
        return largest, img

    return largest

#Compute the cars' center
#Parameters: Car's contour, track's contour, draw_flag and current frame
#Returns: Car's centre (x, y) and (if draw_flag -> 1) the frame with center drawn
#Updates lap time and lap number, and gives window a slight red tint if car out of bounds
def carcentre(car_contour, track_contour, draw_flag=0, img=None):

    global FINISHING_LINE_MIN_X, FINISHING_LINE_MAX_X, FINISHING_LINE_MIN_Y, FINISHING_LINE_MAX_Y, red, lap_number, lap_start_time, lap_time, batota, f_red, aux

    if car_contour is None or track_contour is None:
        return (-1, -1, img) if draw_flag else (-1, -1)

    # Calculate moments and validate
    M = cv2.moments(car_contour)
    if M["m00"] == 0:
        return (-1, -1, img) if draw_flag else (-1, -1)
    
    #Get centre coordinates
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    x_m, y_m = px_to_m(x, y)
    theta_rad = update_raw_theta(x_m, y_m, time.time())
    theta_deg = float(np.degrees(theta_rad))

    #Check to see if centre in limits or out
    InorOut = cv2.pointPolygonTest(track_contour, (x,y), False)

    if not draw_flag:
        return x, y
    
    if img is None:
        return -1, -1, None  


    if draw_flag:

        #On track
        if InorOut == 1:

            #If it just left the finishing line
            if aux == 1:
                lap_start_time = time.time()

            aux = 0
            append_track_trail(x, y)
            draw_track_trail(img)
            draw_robot_ellipse(img, x_m, y_m, theta_rad, (0, 255, 0))
            cv2.circle(img, (x,y), radius = 2, color=[0, 255, 0],
                    thickness=5)
            cv2.putText(
                img,
                f"x={x_m:.3f}m y={y_m:.3f}m th={theta_deg:.1f}deg",
                (x + 10, y - 10),
                font,
                fontScale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA
            )
            
            f_red = 0

            return x, y, img

        #Crossing the finishing line
        elif FINISHING_LINE_MIN_X <= x <= FINISHING_LINE_MAX_X and FINISHING_LINE_MIN_Y <= y <= FINISHING_LINE_MAX_Y:

            #First time crossing it
            if aux == 0:

                lap_time = time.time() - lap_start_time

                #If it went out of bounds
                if batota != 0:
                    lap_time = round(lap_time + 3 * batota, 2)
                else:
                    lap_time = round(lap_time, 2)
                batota = 0
                lap_number += 1
                aux = 1
            
            append_track_trail(x, y)
            draw_track_trail(img)
            draw_robot_ellipse(img, x_m, y_m, theta_rad, (0, 255, 255))
            cv2.circle(img, (x, y), 2, (0, 255, 255), 5)
            cv2.putText(
                img,
                f"x={x_m:.3f}m y={y_m:.3f}m th={theta_deg:.1f}deg",
                (x + 10, y - 10),
                font,
                fontScale,
                (0, 255, 255),
                thickness,
                cv2.LINE_AA
            )
            return x, y, img
            
        #Out of Bounds
        else:

            #if it wasnt already out of bounds
            if f_red == 0:
                batota += 1
                f_red = 1
            
            if red is not None:
                cv2.addWeighted(img, alpha, red, beta, 0.0, img)

            append_track_trail(x, y)
            draw_track_trail(img)
            draw_robot_ellipse(img, x_m, y_m, theta_rad, (0, 0, 255))
            cv2.putText(
                img,
                f"x={x_m:.3f}m y={y_m:.3f}m th={theta_deg:.1f}deg",
                (x + 10, y - 10),
                font,
                fontScale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA
            )
            return x, y, img

#View specific frames
#Parameters: name of window and frame to be displayed
def show(name, frame):

    cv2.imshow(name, frame)
    return cv2.waitKey(1) & 0xFF

#check if the function worked
def check(cap):

    print("Camera error.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

#Load the calibration parameters
if os.path.exists("calibration_data.npz"):
    with np.load("calibration_data.npz") as data:
        mtx = data['camera_matrix']
        dist = data['dist_coeff']
else:
    print("Calibration file not found.")
    exit()

def main():

    global click_count, FINISHING_LINE_MIN_X, FINISHING_LINE_MAX_X, FINISHING_LINE_MIN_Y, FINISHING_LINE_MAX_Y, contour, lap_number, lap_time, best_lap_time

    #Open the camera
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    frame_counter = 0
    ret, frame = cap.read()
    if not ret:
        check(cap)    

    calibrate_blue_color(cap)

    #Get Track Limits
    while True:

        ret, frame = cap.read()
        if not ret:
            check(cap)
         
        #Process frame
        frame = process_frame(frame)

        #Get the tracks limits
        img, contour = limits(frame)
        cv2.imshow("limits", img)
            
        key = cv2.waitKey(0) & 0xFF
        if key in [ord('c'), ord('C')]:
            break

        if key in [ord('q'), ord('Q')]:
            cv2.destroyAllWindows()
            cap.release()
            exit()
        
    cv2.destroyAllWindows()

    #Get finishing line coordinates
    cv2.namedWindow("Live")
    cv2.setMouseCallback("Live", set_finishing_line)

    print("Select the upper left corner of the finishing line")

    #Loop until all coordinatess are set
    while click_count < 2:

        ret, frame = cap.read()
        if not ret:
            check(cap)
        
        frame = process_frame(frame)
        if frame is None:
            continue

        cv2.imshow("Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    #Load red image for overlay
    overlay(frame.shape)
    if red is None:
        print("Warning: Red overlay not loaded. Skipping overlay functionality.")

    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", edit_control_points)
    print("Editor: arrasta os pontos do caminho/barreiras ou os obstaculos na janela Tracking e carrega em S para guardar.")

    #Tracking the car
    while True:

        ret, frame = cap.read()
        if not ret:
            check(cap)
        frame_counter += 1
        
        #Process each frame
        frame = process_frame(frame)
        cv2.imshow("Live", frame)

        key = cv2.waitKey(10) & 0xFF
        if handle_tracking_key(key):
            break

        #Get car contour
        car_contour = carcontour(frame)

        display_frame = frame.copy()
        draw_reference_path(display_frame)
        draw_barriers(display_frame)
        draw_obstacles(display_frame)

        if car_contour is None:
            key = show("Tracking", display_frame)
            if handle_tracking_key(key):
                break
            continue

        #Get car center
        x, y, img = carcentre(car_contour, contour, draw_flag=1, img=display_frame)

        if img is not None:
            x_m, y_m = px_to_m(x, y)
            if SEND_UDP and np.isfinite(x_m) and np.isfinite(y_m):
                payload = build_udp_payload(frame_counter, x, y, x_m, y_m)
                sock.sendto(json.dumps(payload).encode("utf-8"), (PC_IP, PC_PORT))

            #Update best lap time
            if lap_time < best_lap_time:
                best_lap_time = lap_time

            #Update lap info
            phrases = [
                f"LAP: {lap_number}",
                f"TIME:{lap_time:.2f}s",
                f"BEST:{best_lap_time:.2f}s"
            ]

            #Display border and lap info
            img = apply_border(img, phrases)
            key = show("Tracking", img)
            if handle_tracking_key(key):
                break

    cap.release()
    sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
