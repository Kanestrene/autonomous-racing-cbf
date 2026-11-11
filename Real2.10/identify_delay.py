"""
Identifica LOOP_DELAY_S e STEER_GAIN a partir de um log do run_real_esp_*.py (pasta logs/).

Ideia: a velocidade angular medida pela visao (no instante t_capture) responde ao
comando de direcao enviado tau segundos antes:

    w_medido(t) ~= STEER_GAIN * v_medido(t) * tan(delta_cmd(t - tau)) / L

Para cada tau testa-se o ajuste (minimos quadrados no ganho) e escolhe-se o tau com
menor erro. Esse tau e o atraso que a visao nao mede (camara + comunicacao + motor),
ou seja, o LOOP_DELAY_S do runner.

Uso:  py -3.13 identify_delay.py logs/run_YYYYmmdd_HHMMSS.csv [--L 0.06] [--plot]
Dica: quanto mais a direcao variar durante o log, melhor (curvas, slalom, desvios).
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def load_log(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Log vazio.")

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.asarray(out)

    return {k: col(k) for k in rows[0].keys()}


def moving_average(a, n):
    if n <= 1:
        return a
    k = np.ones(n) / n
    pad = n // 2
    a_pad = np.concatenate([np.full(pad, a[0]), a, np.full(n - 1 - pad, a[-1])])
    return np.convolve(a_pad, k, mode="valid")


def zoh(t_query, t_samples, values):
    """Valor do comando em vigor (ultimo enviado ate t_query)."""
    k = np.searchsorted(t_samples, t_query, side="right") - 1
    out = np.where(k >= 0, values[np.clip(k, 0, None)], np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--L", type=float, default=0.06, help="distancia entre eixos (m), igual ao runner")
    ap.add_argument("--fs", type=float, default=100.0, help="frequencia da grelha de reamostragem (Hz)")
    ap.add_argument("--smooth-s", type=float, default=0.10, help="janela da media movel (s), centrada")
    ap.add_argument("--window-s", type=float, default=0.30,
                    help="compara a variacao do angulo nesta janela (mesmo filtro nos dois sinais); "
                         "tira a correlacao falsa do ruido que a malha fechada mete no comando")
    ap.add_argument("--max-delay", type=float, default=0.5)
    ap.add_argument("--min-speed", type=float, default=0.05)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    d = load_log(args.log)

    # medidas unicas (o runner corre mais depressa que a camara: ha repetidas)
    tm = d["t_capture"]
    ok = np.isfinite(tm) & np.isfinite(d["x_meas"]) & np.isfinite(d["th_meas"])
    tm, idx = np.unique(tm[ok], return_index=True)
    xm = d["x_meas"][ok][idx]
    ym = d["y_meas"][ok][idx]
    thm = np.unwrap(d["th_meas"][ok][idx])

    # comandos
    tc = d["t_cmd"]
    okc = np.isfinite(tc) & np.isfinite(d["delta_cmd"])
    order = np.argsort(tc[okc])
    tc = tc[okc][order]
    dc = d["delta_cmd"][okc][order]
    vc = d["v_cmd"][okc][order]

    if len(tm) < 50 or len(tc) < 50:
        raise SystemExit("Log demasiado curto.")

    # grelha uniforme, suavizacao centrada (nao introduz atraso) e derivadas
    dt = 1.0 / args.fs
    t = np.arange(max(tm[0], tc[0]) + args.max_delay, tm[-1], dt)
    n_s = max(1, int(round(args.smooth_s * args.fs)))
    x = moving_average(np.interp(t, tm, xm), n_s)
    y = moving_average(np.interp(t, tm, ym), n_s)
    th = moving_average(np.interp(t, tm, thm), n_s)
    w_meas = np.gradient(th, dt)
    v_meas = np.hypot(np.gradient(x, dt), np.gradient(y, dt))

    # media da janela = variacao do angulo na janela / janela (igual para os dois sinais)
    n_w = max(1, int(round(args.window_s * args.fs)))
    w_meas_w = moving_average(w_meas, n_w)

    # bordas das medias moveis e trocos parados ficam de fora
    edge = n_s + n_w
    valid = moving_average((v_meas > args.min_speed).astype(float), n_w) > 0.999
    valid[:edge] = False
    valid[-edge:] = False

    taus = np.arange(0.0, args.max_delay + 1e-9, 0.005)
    costs, gains = [], []
    for tau in taus:
        delta = zoh(t - tau, tc, dc)
        w_model = moving_average(np.nan_to_num(v_meas * np.tan(delta) / args.L), n_w)
        m = valid & np.isfinite(delta)
        a, b = w_model[m], w_meas_w[m]
        g = float(a @ b / max(1e-12, a @ a))
        costs.append(float(np.mean((b - g * a) ** 2) / max(1e-12, np.var(b))))
        gains.append(g)

    costs = np.asarray(costs)
    k = int(np.argmin(costs))
    tau_best, g_best = float(taus[k]), float(gains[k])

    # ganho de velocidade (para PRED_SPEED_SOURCE = "command")
    v_cmd_d = zoh(t - tau_best, tc, vc)
    m = valid & np.isfinite(v_cmd_d) & (v_cmd_d > 1e-3)
    k_v = float(v_cmd_d[m] @ v_meas[m] / max(1e-12, v_cmd_d[m] @ v_cmd_d[m])) if m.any() else float("nan")

    vision_age = d["t_now"] - d["t_capture"]
    vision_age = vision_age[np.isfinite(vision_age)]

    print(f"Log: {args.log}  ({len(tm)} medidas, {len(tc)} comandos, {t[-1] - t[0]:.1f} s)")
    print(f"Atraso da visao medido (t_now - t_capture): mediana {1000 * np.median(vision_age):.0f} ms, "
          f"p95 {1000 * np.percentile(vision_age, 95):.0f} ms")
    print(f"LOOP_DELAY_S = {tau_best:.3f}   (erro relativo do ajuste {costs[k]:.3f}, R2 = {1 - costs[k]:.3f})")
    print(f"STEER_GAIN   = {g_best:.3f}")
    print(f"ganho de velocidade v_real / v_cmd = {k_v:.3f}")
    if k in (0, len(taus) - 1):
        print("Aviso: o minimo ficou no limite da procura; aumenta --max-delay ou ve se o log tem curvas.")
    if costs[k] > 0.6:
        print("Aviso: ajuste fraco (pouca variacao da direcao no log?); o atraso pode nao ser fiavel.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        delta = zoh(t - tau_best, tc, dc)
        w_model = g_best * moving_average(np.nan_to_num(v_meas * np.tan(delta) / args.L), n_w)
        w_meas = w_meas_w
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
        ax1.plot(taus * 1000, costs)
        ax1.axvline(tau_best * 1000, ls="--", c="k")
        ax1.set_xlabel("atraso testado (ms)")
        ax1.set_ylabel("erro relativo")
        ax2.plot(t - t[0], np.where(valid, w_meas, np.nan), label="w medido (visao)")
        ax2.plot(t - t[0], np.where(valid, w_model, np.nan), label=f"modelo, atraso {tau_best * 1000:.0f} ms")
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("w (rad/s)")
        ax2.legend()
        fig.tight_layout()
        out = Path(args.log).with_suffix(".delay.png")
        fig.savefig(out, dpi=120)
        print(f"Grafico guardado em {out}")


if __name__ == "__main__":
    main()
