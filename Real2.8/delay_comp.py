"""
Compensacao do atraso da estimacao de estado (preditor tipo Smith).

A medida da visao chega atrasada: o frame foi capturado em t_capture e o controlador
so a usa em t_now (processamento + UDP + espera do ciclo). Alem disso, o comando
calculado agora so atua no carro LOOP_DELAY_S mais tarde.

O preditor integra o modelo bicicleta desde t_capture ate t_now + loop_delay_s,
com os comandos ja enviados (um comando enviado em t_k atua em t_k + loop_delay_s):

    x' = v cos(th)    y' = v sin(th)    th' = steer_gain * v * tan(delta) / L

Atrasos:
  - t_now - t_capture (software da visao): medido em cada passo, mesmo relogio do PC
  - loop_delay_s: o que nao se mede por software, somado (exposicao/buffer da camara
    + WiFi/BLE + ciclo da ESP32 + resposta do motor/direcao). Basta a soma: dividi-lo
    entre camara e atuacao da a mesma predicao. Estima-se com identify_delay.py.
"""
import csv
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np


def wrap_angle_pi(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class StatePredictor:
    def __init__(
        self,
        L,
        loop_delay_s=0.10,
        steer_gain=1.0,
        speed_source="measured",   # "measured" (visao) ou "command" (speed_gain * v enviado)
        speed_gain=1.0,
        speed_window_s=0.25,
        max_horizon_s=0.5,
        int_dt=0.005,
        history_s=2.0,
    ):
        self.L = float(L)
        self.loop_delay_s = float(loop_delay_s)
        self.steer_gain = float(steer_gain)
        self.speed_source = speed_source
        self.speed_gain = float(speed_gain)
        self.speed_window_s = float(speed_window_s)
        self.max_horizon_s = float(max_horizon_s)
        self.int_dt = float(int_dt)
        self.history_s = float(history_s)

        self.cmd_t = []
        self.cmd_v = []
        self.cmd_d = []
        self.meas = []   # (t_capture, x, y, th)

    def record_command(self, t_sent, v, delta):
        """delta na convencao do modelo (o que o controlador quer, antes da troca de sinal da ESP)."""
        self.cmd_t.append(float(t_sent))
        self.cmd_v.append(float(v))
        self.cmd_d.append(float(delta))

        # guarda o historico necessario (o horizonte nunca passa de max_horizon_s + atraso)
        t_min = float(t_sent) - self.history_s
        k = bisect_right(self.cmd_t, t_min) - 1
        if k > 0:
            del self.cmd_t[:k], self.cmd_v[:k], self.cmd_d[:k]

    def add_measurement(self, t_capture, x, y, th):
        t_capture = float(t_capture)
        if self.meas and t_capture <= self.meas[-1][0]:
            return
        self.meas.append((t_capture, float(x), float(y), float(th)))

        t_min = t_capture - 4.0 * self.speed_window_s
        while len(self.meas) > 2 and self.meas[0][0] < t_min:
            self.meas.pop(0)

    def measured_speed(self):
        if len(self.meas) < 2:
            return 0.0
        t1, x1, y1, _ = self.meas[-1]
        for t0, x0, y0, _ in self.meas:
            if t0 >= t1 - self.speed_window_s:
                break
        if t1 - t0 < 1e-3:
            t0, x0, y0, _ = self.meas[-2]
        return float(np.hypot(x1 - x0, y1 - y0) / max(1e-3, t1 - t0))

    def _command_at(self, t):
        k = bisect_right(self.cmd_t, t) - 1
        if k < 0:
            return None
        return self.cmd_v[k], self.cmd_d[k]

    def predict(self, t_now):
        """
        Estado previsto para t_now + loop_delay_s (quando o comando deste passo atuar).
        Devolve (x, y, th, v, info) ou None se ainda nao ha medidas.
        """
        if not self.meas:
            return None

        t0, x, y, th = self.meas[-1]
        horizon = float(np.clip(t_now + self.loop_delay_s - t0, 0.0, self.max_horizon_s))
        v_meas = self.measured_speed()

        tau = t0
        t_end = t0 + horizon
        while tau < t_end - 1e-9:
            h = min(self.int_dt, t_end - tau)
            cmd = self._command_at(tau - self.loop_delay_s)
            if cmd is None:
                v_c, d_c = v_meas, 0.0
            else:
                v_c, d_c = cmd

            v = self.speed_gain * v_c if self.speed_source == "command" else v_meas
            w = self.steer_gain * v * np.tan(d_c) / self.L

            th_mid = th + 0.5 * w * h
            x += v * np.cos(th_mid) * h
            y += v * np.sin(th_mid) * h
            th += w * h
            tau += h

        info = {"horizon": horizon, "vision_age": float(t_now - t0), "t_capture": t0}
        return x, y, wrap_angle_pi(th), v_meas, info


class RunLogger:
    """CSV de cada passo do controlo (serve para o identify_delay.py)."""

    FIELDS = [
        "t_now", "t_capture", "t_cmd",
        "x_meas", "y_meas", "th_meas",
        "x_ctrl", "y_ctrl", "th_ctrl",
        "v_meas", "v_cmd", "w_cmd", "delta_cmd",
        "horizon", "ey", "epsi",
    ]

    def __init__(self, folder, prefix="run"):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        print(f"A gravar log do controlo em {self.path}")

    def log(self, **row):
        self._w.writerow({k: row.get(k, "") for k in self.FIELDS})

    def close(self):
        self._f.close()
        print(f"Log guardado em {self.path}")
