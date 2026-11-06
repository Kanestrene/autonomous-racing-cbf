import time
import numpy as np
import qp
import socket
import json
import asyncio

from pathlib import Path

from shared_config import OBSTACLES_M, TRACK_POINTS_M
from delay_comp import RunLogger, StatePredictor

from controller import (
    build_spline_path,
    omega_to_delta,
    rate_limit,
)


# ============================================================
# Estado real via UDP (cam.py)
# ============================================================
VISION_UDP_IP = "0.0.0.0"
VISION_UDP_PORT = 5005
VISION_TRACK_ID = 2
ROUND_CORNER_RADIUS_M = 0.08
ROUND_CORNER_SAMPLES = 10


# ============================================================
# Compensacao do atraso do estado (delay_comp.py)
# ============================================================
DELAY_COMP = True
LOOP_DELAY_S = 0.10            # atraso nao medido: camara + comunicacao + ESP32 + motor (identify_delay.py)
STEER_GAIN = 1.0               # th' = STEER_GAIN * v * tan(delta) / L (identify_delay.py)
PRED_SPEED_SOURCE = "measured" # "measured" (visao) ou "command"
LOG_RUN = True                 # CSV em logs/ para o identify_delay.py


# ============================================================
# Comandos para a ESP32 via WiFi (UDP)
# ============================================================
# Hostname mDNS definido no firmware (HOSTNAME = "xiao-c3-car").
# Se o .local nao resolver, poe aqui o IP que a ESP32 imprime no Serial.
ESP32_HOST = "xiao-c3-car.local"
ESP32_FALLBACK_IP = "10.232.18.146"  # rede "Lopes" (DHCP: pode mudar); em modo AP e "192.168.4.1"
ESP32_UDP_PORT = 5006


class VisionPoseReceiver:
    def __init__(self, bind_ip=VISION_UDP_IP, bind_port=VISION_UDP_PORT, track_id=VISION_TRACK_ID):
        self.track_id = track_id
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_ip, bind_port))
        self.sock.setblocking(False)  # nao bloquear: o estado mais recente entra logo
        self.on_measurement = None    # callback(t_capture, x, y, theta) por cada pacote
        self.t_capture = None

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
        t_capture = float(payload.get("t_capture", payload.get("t", now)))
        self.t_capture = t_capture
        if self.on_measurement is not None:
            self.on_measurement(t_capture, x, y, theta)

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

            except (BlockingIOError, socket.timeout):
                break

            except Exception:
                break

        if not self.has_state:
            raise RuntimeError("Sem estado da visao ainda (aguardando UDP na porta 5005).")

        if not got_new_packet and time.time() - self.prev_t > 0.2:
            self.v = 0.0

        return self.x, self.y, self.theta, self.v


_vision_receiver = VisionPoseReceiver()


class Esp32CommandSender:
    def __init__(self, host=ESP32_HOST, port=ESP32_UDP_PORT, fallback_ip=ESP32_FALLBACK_IP):
        self.host = host
        self.port = port
        self.fallback_ip = fallback_ip
        self.target = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @staticmethod
    def format_command(v, delta):
        return f"V:{v:.3f} D:{delta:.3f}"

    async def connect(self):
        print(f"A resolver ESP32: {self.host}")

        try:
            ip = socket.gethostbyname(self.host)
        except socket.gaierror:
            if self.fallback_ip is None:
                raise RuntimeError(
                    f"Nao consegui resolver '{self.host}'. "
                    "Define ESP32_FALLBACK_IP com o IP que a ESP32 imprime no Serial."
                )
            ip = self.fallback_ip

        self.target = (ip, self.port)
        print(f"A enviar comandos UDP para a ESP32 em {ip}:{self.port}")

    async def send_cmd(self, v, delta):
        msg = self.format_command(v, delta)

        if self.target is None:
            raise RuntimeError("ESP32 sem endereco (chama connect primeiro)")

        self.sock.sendto(msg.encode("utf-8"), self.target)
        return msg

    async def stop(self):
        try:
            if self.target is not None:
                # UDP pode perder pacotes: envia o stop varias vezes
                for _ in range(5):
                    await self.send_cmd(0.0, 0.0)
                    await asyncio.sleep(0.02)
        except Exception:
            pass
        finally:
            self.sock.close()


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
                pt = (
                    ((1.0 - t) ** 2) * p_start
                    + 2.0 * (1.0 - t) * t * p_curr
                    + (t ** 2) * p_end
                )
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
            pt = (
                ((1.0 - t) ** 2) * p_start
                + 2.0 * (1.0 - t) * t * p_curr
                + (t ** 2) * p_end
            )
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
# Loop principal
# ============================================================

async def run_real():
    waypoints = build_rounded_polyline(
        TRACK_POINTS_M,
        corner_radius=ROUND_CORNER_RADIUS_M,
        corner_samples=ROUND_CORNER_SAMPLES,
        closed=True,
    )

    if waypoints and waypoints[0] != waypoints[-1]:
        waypoints.append(waypoints[0])

    px, py, pyaw, s = build_spline_path(waypoints, ds=0.01)

    obstacles = [dict(obstacle) for obstacle in OBSTACLES_M]

    # parâmetros
    dt = 0.02
    v_ref = 0.3

    v_max = 0.47
    a_max = 2.0

    a_ell, b_ell = 0.03, 0.03
    margin = 0.01

    last_near = 0

    # Steering state
    delta = 0.0

    # Parâmetros bicycle/servo
    L = 0.06
    delta_max = np.deg2rad(13)
    delta_rate_max = np.deg2rad(300)
    kappa_max = np.tan(delta_max) / L

    predictor = StatePredictor(
        L,
        loop_delay_s=LOOP_DELAY_S,
        steer_gain=STEER_GAIN,
        speed_source=PRED_SPEED_SOURCE,
    )
    _vision_receiver.on_measurement = predictor.add_measurement
    logger = RunLogger(Path(__file__).with_name("logs")) if LOG_RUN else None

    esp32 = Esp32CommandSender()
    await esp32.connect()

    x, y, yaw, v = wait_for_initial_state()

    print(
        f"Estado inicial recebido: x={x:.3f} m, y={y:.3f} m, "
        f"yaw={yaw:.3f} rad, v={v:.3f} m/s"
    )

    try:
        while True:
            t0 = time.time()

            x_meas, y_meas, yaw_meas, v = get_robot_state()
            x, y, yaw = x_meas, y_meas, yaw_meas
            horizon = 0.0
            if DELAY_COMP:
                # estado previsto para quando este comando atuar no carro
                pred = predictor.predict(time.time())
                if pred is not None:
                    x, y, yaw, v, pred_info = pred
                    horizon = pred_info["horizon"]

            v_nom = v_ref
            w_nom = 0.0

            (u_safe, clf_info) = qp.cbf_clf_qp_filter(
                u_nom=(v_nom, w_nom),
                robot_state=(x, y, yaw),
                obstacles=obstacles,
                px=px, py=py, pyaw=pyaw, s=s,
                last_path_idx=last_near,
                ellipse_ab=(a_ell, b_ell),
                margin=margin,
                lookahead_l=0.15, #0.15
                alpha=3, #3
                eps_clf=6,
                q_clf=(0.01, 10.0, 0.01),
                W=(2500000.0, 1.0),
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

            # Se quiseres suavizar servo, troca para:
            #delta = rate_limit(delta_cmd, delta, du_max=delta_rate_max * dt)
            delta = delta_cmd

            if 0.0 < delta < 0.05:
                delta_send = 0.05
            elif 0.1 < delta < 0.25:
                delta_send = delta
            elif -0.25 < delta < -0.1:
                delta_send = delta
            elif -0.05 < delta < 0.0:
                delta_send = -0.05               
            else:
                delta_send = delta

            esp32_msg = await esp32.send_cmd(v=v_safe, delta= - 1 * delta)
            t_cmd = time.time()
            predictor.record_command(t_cmd, v_safe, delta)
            if logger is not None:
                logger.log(
                    t_now=t0, t_capture=_vision_receiver.t_capture, t_cmd=t_cmd,
                    x_meas=x_meas, y_meas=y_meas, th_meas=yaw_meas,
                    x_ctrl=x, y_ctrl=y, th_ctrl=yaw,
                    v_meas=v, v_cmd=v_safe, w_cmd=w_safe, delta_cmd=delta,
                    horizon=horizon, ey=clf_info["ey"], epsi=clf_info["epsi"],
                )
            #esp32_msg = await esp32.send_cmd(v=0.0, delta=0.3)

            print(
                f"estado: x={x:.3f} m, y={y:.3f} m, yaw={yaw:.3f} rad, "
                f"v={v:.3f} m/s pred_h={horizon * 1000:.0f}ms | "
                f"ref: idx={clf_info['idx']} psi_r={clf_info['psi_r']:.3f} ey={clf_info['ey']:.3f} "
                f"epsi={clf_info['epsi']:.3f} V={clf_info['V']:.3f} | "
                f"cmd: v={v_safe:.2f} m/s, w={w_safe:.3f} rad/s, delta_raw={delta_cmd_unclipped:.3f} rad, "
                f"delta={delta:.3f} rad, cte={cte:.3f} | "
                f"wifi='{esp32_msg}'"
            )

            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("Parado pelo utilizador")

    finally:
        await esp32.stop()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    asyncio.run(run_real())
