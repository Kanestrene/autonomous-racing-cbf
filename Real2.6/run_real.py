import time
import numpy as np
import qp
import socket
import json
from shared_config import OBSTACLES_M, TRACK_POINTS_M

from controller import (
    build_spline_path,
    pure_pursuit_control,
    omega_to_delta,
    rate_limit,
)

from car_interface import CarController


# ============================================================
# Estado real via UDP (cam.py)
# ============================================================
VISION_UDP_IP = "0.0.0.0"
VISION_UDP_PORT = 5005
VISION_TRACK_ID = 2
ROUND_CORNER_RADIUS_M = 0.08
ROUND_CORNER_SAMPLES = 10


class VisionPoseReceiver:
    def __init__(self, bind_ip=VISION_UDP_IP, bind_port=VISION_UDP_PORT, track_id=VISION_TRACK_ID):
        self.track_id = track_id
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_ip, bind_port))
        self.sock.settimeout(0.02)

        self.has_state = False
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.prev_x = None
        self.prev_y = None
        self.prev_t = None

    def _select_track(self, tracks):
        for tr in tracks:
            if int(tr.get("id", -999999)) == self.track_id:
                return tr

        positive_ids = [tr for tr in tracks if int(tr.get("id", -1)) >= 0]
        if positive_ids:
            return positive_ids[0]

        return tracks[0] if tracks else None

    def _process_packet(self, payload):
        tracks = payload.get("tracks", [])
        tr = self._select_track(tracks)
        if tr is None:
            return

        x = float(tr["x_m"])
        y = float(tr["y_m"])
        theta = float(tr.get("theta_rad", self.theta))
        v = float(tr.get("speed_m_s", self.v))
        now = time.time()

        if not self.has_state:
            self.x = x
            self.y = y
            self.theta = theta
            self.v = v
            self.prev_x = x
            self.prev_y = y
            self.prev_t = now
            self.has_state = True
            return

        self.x = x
        self.y = y
        self.theta = theta
        self.v = v

        self.prev_x = x
        self.prev_y = y
        self.prev_t = now

    def get_state(self):
        got_new_packet = False

        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
                payload = json.loads(data.decode("utf-8"))
                self._process_packet(payload)
                got_new_packet = True
            except socket.timeout:
                break
            except Exception:
                break

        if not self.has_state:
            raise RuntimeError("Sem estado da visao ainda (aguardando UDP na porta 5005).")

        if not got_new_packet:
            self.v = 0.0

        return self.x, self.y, self.theta, self.v


_vision_receiver = VisionPoseReceiver()


def get_robot_state():
    """
    Devolve x, y, yaw, v.
    Valores vindos diretamente do payload UDP do cam.py.
    """
    return _vision_receiver.get_state()


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


def wait_for_initial_state():
    while True:
        try:
            return get_robot_state()
        except Exception as e:
            print(f"A aguardar estado inicial... {e}")
            time.sleep(0.1)


# ============================================================
# Loop principal (SEM plots)
# ============================================================

def run_real():
    waypoints = build_rounded_polyline(
        TRACK_POINTS_M,
        corner_radius=ROUND_CORNER_RADIUS_M,
        corner_samples=ROUND_CORNER_SAMPLES,
        closed=True,
    )
    if waypoints and waypoints[0] != waypoints[-1]:
        waypoints.append(waypoints[0])

    px, py, pyaw, s = build_spline_path(waypoints, ds=0.01)

    # obstáculos
    n_obs = 5
    idxs = np.linspace(0, len(px) - 1, n_obs + 2, dtype=int)[1:-1]

    
    obstacles = [dict(obstacle) for obstacle in OBSTACLES_M]
    '''
    for k, idx in enumerate(idxs):
        x_path = px[idx]
        y_path = py[idx]
        yaw_path = pyaw[idx]

        nx = -np.sin(yaw_path)
        ny = np.cos(yaw_path)

        side = (-1) ** k
        offset = 0.3

        obstacles.append({
            "x": x_path + side * offset * nx,
            "y": y_path + side * offset * ny,
            "r": 0.35
        })
    '''

    # parâmetros
    dt = 0.02
    v_ref = 0.35
    L0 = 0.3
    kv = 0.4

    v_max = 0.47
    w_max = 2.5
    a_max = 2.0

    a_ell, b_ell = 0.03, 0.03
    margin = 0.001

    last_near = 0

    # Steering state
    delta = 0.0

    # Parâmetros bicycle/servo
    L = 0.04                         # entre-eixos (m)
    delta_max = np.deg2rad(16.5)       # limite do servo
    delta_rate_max = np.deg2rad(300) # rad/s
    kappa_max = np.tan(delta_max) / L

    car = CarController(v_max=v_max, delta_max=delta_max)
    car.send_heartbeat()
    time.sleep(0.02)

    #car.send_cmd(v=0.3, delta=0.15)              
    #car.send_cmd(v=1, delta=0)

    # estado inicial
    x, y, yaw, v = wait_for_initial_state()
    print(
        f"Estado inicial recebido: x={x:.3f} m, y={y:.3f} m, yaw={yaw:.3f} rad, v={v:.3f} m/s"
    )

    try:
        while True:
            t0 = time.time()

            # ler estado REAL
            x, y, yaw, v = get_robot_state()

            # nominal simples
            v_nom = v_ref
            w_nom = 0.0         
            
            # limitar aceleração
            #dv = np.clip(v_cmd - v, -a_max * dt, a_max * dt)
            #v_cmd = v + dv

            
            # CBF + CLF (Lyapunov), igual ao Pack2
            (u_safe, clf_info) = qp.cbf_clf_qp_filter(
                u_nom=(v_nom, w_nom),
                robot_state=(x, y, yaw),
                obstacles=obstacles,
                px=px, py=py, pyaw=pyaw, s=s,
                last_path_idx=last_near,
                ellipse_ab=(a_ell, b_ell),
                margin=margin,
                lookahead_l=0.1,
                alpha=2.5,
                eps_clf=0.5,
                q_clf=(1.0, 10.0, 0.01),
                W=(2500.0, 1.0),
                p_slack=50.0,
                v_ref=v_ref,
                v_bounds=(0.0, 2.0),
                kappa_max=kappa_max,
            )

            v_safe, w_safe = u_safe
            last_near = clf_info["idx"]
            cte = clf_info["ey"]
            v_safe = np.clip(v_safe, -v_max, v_max)

            kappa_max = np.tan(delta_max) / L
            w_max_speed = abs(v_safe) * kappa_max
            w_safe = np.clip(w_safe, -w_max_speed, w_max_speed)

            delta_cmd = omega_to_delta(w_safe, v_safe, L, v_min=0.2)
            delta_cmd_unclipped = delta_cmd
            delta_cmd = np.clip(delta_cmd, -delta_max, delta_max)

            # limita taxa do servo
            #delta = rate_limit(delta_cmd, delta, du_max=delta_rate_max * dt)
            delta = delta_cmd

            # enviar para o carro
            #car.send_heartbeat()
            car.send_cmd(v=v_safe, delta=delta)
            #car.send_cmd(v=0.3, delta=0.15)
            
            # debug
            print(
                f"estado: x={x:.3f} m, y={y:.3f} m, yaw={yaw:.3f} rad, v={v:.3f} m/s | "
                f"ref: idx={clf_info['idx']} psi_r={clf_info['psi_r']:.3f} ey={clf_info['ey']:.3f} "
                f"epsi={clf_info['epsi']:.3f} V={clf_info['V']:.3f} | "
                f"cmd: v={v_safe:.2f} m/s, w={w_safe:.3f} rad/s, delta_raw={delta_cmd_unclipped:.3f} rad, "
                f"delta={delta:.3f} rad, cte={cte:.3f}"
            )

            # manter frequência
            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("Parado pelo utilizador")

    finally:
        car.stop()


if __name__ == "__main__":
    run_real()
