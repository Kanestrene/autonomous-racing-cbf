"""
Comparação simulate1 (qp.cbf_clf_qp_filter) vs simulate1_custo_paper
(qp.cbf_clf_qp_paper_cost) na mesma pista.

Corre as duas simulações sem animação (cada uma com os seus próprios
parâmetros, definidos no respetivo ficheiro) e junta os resultados nos
mesmos gráficos.

Uso:  python compare_simulate1.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch

import simulate1
import simulate1_custo_paper

OUT_DIR = Path(__file__).resolve().parent / "figuras" / "compare_simulate1.1"

COLOR_A = "green"            # simulate1
COLOR_B = "red"              # simulate1_custo_paper
COLOR_BARRIER = "black"
LABEL_A = "Proposed"
LABEL_B = "Ref. [9]"

LINE_WIDTH = 2.2
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 12

N_REPS = 5  # repetições por método para as medições de tempo


def print_stats(label, res):
    step_ms, qp_ms, lat = res["step_ms"], res["qp_ms"], res["lat"]
    print(f"[{label}] volta completa: {'sim' if res['lap_completed'] else 'NAO'} | "
          f"duracao={len(lat) * res['dt']:.2f} s")
    print(
        f"  Tempo CPU controlador [ms]: media={step_ms.mean():.3f} | "
        f"mediana={np.median(step_ms):.3f} | max={step_ms.max():.3f} | N={len(step_ms)}"
    )
    print(
        f"  Tempo CPU so solve_qp [ms]: media={qp_ms.mean():.3f} | "
        f"mediana={np.median(qp_ms):.3f} | max={qp_ms.max():.3f}"
    )
    print(
        f"  Erro lateral [m]: medio |e|={np.mean(np.abs(lat)):.4f} | "
        f"rms={np.sqrt(np.mean(lat ** 2)):.4f} | max={np.max(np.abs(lat)):.4f}"
    )


def control_effort(res):
    """J = sum (v_k * w_k)^2 * dt, com a_lat = v * w (modelo uniciclo)."""
    a_lat = np.asarray(res["v_qp"]) * np.asarray(res["w_qp"])
    J = np.sum(a_lat ** 2) * res["dt"]
    T = len(a_lat) * res["dt"]
    return J, T


def print_effort(res_a, res_b):
    # caso 1 = referência (B), caso 2 = proposto (A)
    J2, T2 = control_effort(res_a)
    J1, T1 = control_effort(res_b)
    print("\nEsforco de controlo  J = sum (v*w)^2 dt:")
    for label, J, T in ((LABEL_B, J1, T1), (LABEL_A, J2, T2)):
        print(f"  [{label}] J={J:.3f} m^2/s^3 | T={T:.2f} s | "
              f"J/T={J / T:.4f} m^2/s^4 | a_lat rms={np.sqrt(J / T):.4f} m/s^2")
    print(f"  Eficiencia relativa  eta = J1/J2 * 100 = {J1 / J2 * 100:.2f} %")
    print(f"  Reducao de esforco  (1 - J2/J1) * 100 = {(1 - J2 / J1) * 100:.2f} %")
    # versão normalizada pelo tempo de volta (os dois runs têm durações diferentes)
    P1, P2 = J1 / T1, J2 / T2
    print(f"  [normalizado J/T] eta = {P1 / P2 * 100:.2f} % | "
          f"reducao = {(1 - P2 / P1) * 100:.2f} %")


def plot_effort_table(res_a, res_b, pdf_path):
    J2, T2 = control_effort(res_a)
    J1, T1 = control_effort(res_b)
    P1, P2 = J1 / T1, J2 / T2

    col_labels = ["", r"$J$ [m$^2$/s$^3$]", r"$T$ [s]", r"$J/T$ [m$^2$/s$^4$]",
                  r"$a_{lat}$ RMS [m/s$^2$]"]
    # rótulos das linhas na 1.ª coluna (rowLabels ficaria fora dos eixos e
    # cortado na janela do plt.show)
    cells = [
        [f"{LABEL_B} (1)", f"{J1:.2f}", f"{T1:.2f}", f"{P1:.3f}", f"{np.sqrt(P1):.3f}"],
        [f"{LABEL_A} (2)", f"{J2:.2f}", f"{T2:.2f}", f"{P2:.3f}", f"{np.sqrt(P2):.3f}"],
        ["", "", "", "", ""],
        [r"$\eta_\% = J_1/J_2 \times 100$",
         f"{J1 / J2 * 100:.2f} %", "", f"{P1 / P2 * 100:.2f} %", ""],
        [r"Reduction $= (1 - J_2/J_1) \times 100$",
         f"{(1 - J2 / J1) * 100:.2f} %", "", f"{(1 - P2 / P1) * 100:.2f} %", ""],
    ]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=col_labels, cellLoc="center",
                     colWidths=[0.32, 0.17, 0.17, 0.17, 0.17], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(LEGEND_FONTSIZE)
    for (row, col), cell in table.get_celld().items():
        if row == 3 or (row, col) == (0, 0):  # separação / canto vazio
            cell.set_visible(False)
        elif row == 0 or col == 0:
            cell.set_text_props(weight="bold")
    table[1, 0].get_text().set_color(COLOR_B)
    table[2, 0].get_text().set_color(COLOR_A)

    ax.set_title(r"Control effort  $J = \sum_k (v_k \omega_k)^2 \Delta t$",
                 fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def plot_track(res_a, res_b, pdf_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for obs in res_a["obstacles"]:
        ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"], fill=False))

    # zorder maior: a trajetória verde fica por cima da vermelha
    ax.plot(res_a["hx"], res_a["hy"], "-", color=COLOR_A, linewidth=1.8, label=LABEL_A,
            zorder=3)
    ax.plot(res_b["hx"], res_b["hy"], "-", color=COLOR_B, linewidth=1.8, label=LABEL_B,
            zorder=2)
    # desenhada depois só pela ordem da legenda; zorder baixo mantém-na por baixo
    ax.plot(res_a["px"], res_a["py"], "--", color="gray", label="Spline (reference)",
            zorder=1)

    ax.plot(res_a["inner_bar"][:, 0], res_a["inner_bar"][:, 1], "-",
            color=COLOR_BARRIER, linewidth=2, label="Track limits")
    ax.plot(res_a["outer_bar"][:, 0], res_a["outer_bar"][:, 1], "-",
            color=COLOR_BARRIER, linewidth=2)

    ax.set_aspect("equal", "box")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=LEGEND_FONTSIZE)

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")


def plot_signal(res_a, res_b, key, title, ylabel, pdf_path):
    t_a = np.arange(len(res_a[key + "_qp"])) * res_a["dt"]
    t_b = np.arange(len(res_b[key + "_qp"])) * res_b["dt"]
    # a referência nominal é igual nos dois: desenha-se a do run mais longo
    res_long, t_long = (res_a, t_a) if len(t_a) >= len(t_b) else (res_b, t_b)

    fig, ax = plt.subplots()
    ax.plot(t_long, res_long[key + "_nom"], "--", color="black",
            label="Controller", linewidth=LINE_WIDTH)
    ax.plot(t_a, res_a[key + "_qp"], color=COLOR_A, label=f"QP ({LABEL_A})",
            linewidth=LINE_WIDTH)
    ax.plot(t_b, res_b[key + "_qp"], color=COLOR_B, label=f"QP ({LABEL_B})",
            linewidth=LINE_WIDTH)
    style_axes(ax, title, "Time [s]", ylabel)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")


def plot_box(data_a, data_b, title, ylabel, decimals, pdf_path):
    fig, ax = plt.subplots()
    bp = ax.boxplot(
        [data_a, data_b], widths=0.5, showmeans=False, showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=LINE_WIDTH),
    )
    for i, color in enumerate((COLOR_A, COLOR_B)):
        bp["boxes"][i].set(facecolor=color, alpha=0.35, edgecolor=color, linewidth=1.5)
        for artist in bp["whiskers"][2 * i:2 * i + 2] + bp["caps"][2 * i:2 * i + 2]:
            artist.set(color=color, linewidth=1.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([LABEL_A, LABEL_B])
    ax.margins(y=0.2)  # espaço para a legenda não tapar os bigodes
    style_axes(ax, title, None, ylabel)
    ax.legend(
        [Patch(facecolor=COLOR_A, alpha=0.35, edgecolor=COLOR_A),
         Patch(facecolor=COLOR_B, alpha=0.35, edgecolor=COLOR_B)],
        [f"Median = {np.median(data_a):.{decimals}f} ms",
         f"Median = {np.median(data_b):.{decimals}f} ms"],
        fontsize=LEGEND_FONTSIZE,
    )
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")


def plot_lateral_error(res_a, res_b, pdf_path):
    fig, ax = plt.subplots()
    for res, color, label in ((res_a, COLOR_A, LABEL_A), (res_b, COLOR_B, LABEL_B)):
        t = np.arange(len(res["lat"])) * res["dt"]
        ax.plot(t, res["lat"], color=color, linewidth=LINE_WIDTH,
                label=f"{label} (mean |e| = {np.mean(np.abs(res['lat'])):.3f} m)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    style_axes(ax, "Lateral error to path", "Time [s]", "e_lat [m]")
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")


def run_timed(n_reps=N_REPS):
    """
    Corre cada método n_reps vezes (após 1 corrida de aquecimento descartada),
    alternando a ordem em cada repetição para nenhum método ficar sempre com a
    máquina "a frio". Os tempos de todas as repetições são juntos; o resto dos
    resultados (trajetória, sinais, erro) vem da 1.a repetição (é determinístico).
    """
    sims = {LABEL_A: simulate1.simulate, LABEL_B: simulate1_custo_paper.simulate}
    print("Aquecimento (descartado)...")
    for sim in sims.values():
        sim(animate=False)

    runs = {label: [] for label in sims}
    for rep in range(n_reps):
        order = list(sims) if rep % 2 == 0 else list(sims)[::-1]
        for label in order:
            print(f"Repeticao {rep + 1}/{n_reps}: {label}...")
            runs[label].append(sims[label](animate=False))

    results = []
    for label in sims:
        res = dict(runs[label][0])
        res["step_ms"] = np.concatenate([r["step_ms"] for r in runs[label]])
        res["qp_ms"] = np.concatenate([r["qp_ms"] for r in runs[label]])
        results.append(res)
    return results


def main():
    res_a, res_b = run_timed()

    print()
    print_stats(LABEL_A, res_a)
    print_stats(LABEL_B, res_b)
    print_effort(res_a, res_b)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "track": OUT_DIR / "compare1_volta_completa.pdf",
        "v": OUT_DIR / "compare1_velocidade_linear.pdf",
        "w": OUT_DIR / "compare1_velocidade_angular.pdf",
        "delta": OUT_DIR / "compare1_delta.pdf",
        "cpu": OUT_DIR / "compare1_tempo_cpu.pdf",
        "qp": OUT_DIR / "compare1_tempo_qp.pdf",
        "lat": OUT_DIR / "compare1_erro_lateral.pdf",
        "effort": OUT_DIR / "compare1_esforco_controlo.pdf",
    }

    plot_track(res_a, res_b, paths["track"])
    plot_signal(res_a, res_b, "v", "Linear velocity: Controller vs QP",
                "v [m/s]", paths["v"])
    plot_signal(res_a, res_b, "w", "Angular velocity: Controller vs QP",
                "w [rad/s]", paths["w"])
    plot_signal(res_a, res_b, "delta", "Steering angle: Controller vs QP",
                "delta [rad]", paths["delta"])
    plot_box(res_a["step_ms"], res_b["step_ms"], "Total time per step",
             "Time [ms]", 2, paths["cpu"])
    plot_box(res_a["qp_ms"], res_b["qp_ms"], "QP solve time per step",
             "QP time [ms]", 3, paths["qp"])
    plot_lateral_error(res_a, res_b, paths["lat"])
    plot_effort_table(res_a, res_b, paths["effort"])

    print("\nGraficos guardados em:")
    for p in paths.values():
        print(f"  {p}")
    plt.show()


if __name__ == "__main__":
    main()
