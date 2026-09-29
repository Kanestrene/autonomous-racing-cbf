"""
Comparação na pista do simulate1 (uniciclo puro, sem animação):
  - "simulate1": qp.cbf_clf_qp_filter (CLF de path tracking + CBF lookahead/elipse)
  - "paper":     paper_qp.paper_clf_cbf_qp (Huang et al. 2023, objetivo móvel)

Mede tempo de CPU por passo e erro de seguimento (CTE) medido da mesma forma
para os dois métodos: centro do robô vs spline de referência.

Uso:  python compare.py [--runs 5]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import qp
import paper_qp
from clf import nearest_path_point_closed
from controller import wrap_to_pi, build_spline_path
from simulate1 import WAYPOINTS

SCRIPT_DIR = Path(__file__).resolve().parent

DT = 0.02
T_MAX = 50.0
V_REF = 2.0
V_BOUNDS = (0.0, 2.0)
W_BOUNDS = (-2.5, 2.5)
X0 = (2.0, 6.0, np.deg2rad(90))

# parâmetros do método do paper (H, Q, P_clf e D_GOAL não vêm no paper: afinados aqui)
PAPER_L = 0.15
PAPER_D_GOAL = 1.5
PAPER_PARAMS = dict(
    l=PAPER_L,
    r_robot=0.30,
    margin=0.05,
    alpha=1.5,
    gamma=4.0,   # paper: 0.5 -> com objetivo móvel dá v ~ gamma*d/2 ~ 0.4 m/s
    p_slack=1000.0,
    H=(1.0, 1.0),
    Q=(10.0, 10.0),
    P_clf=((1.0, 0.0, 0.0),
           (0.0, 4.0, 0.5),
           (0.0, 0.5, 0.5)),
    goal_frame=True,
    v_bounds=V_BOUNDS,
    w_bounds=W_BOUNDS,
)


def build_scenario():
    px, py, pyaw, s = build_spline_path(WAYPOINTS, ds=0.01)
    n_path = len(px)

    n_obs = 5
    idxs = np.linspace(0, n_path - 1, n_obs + 2, dtype=int)[1:-1]
    obstacles = []
    for k, idx in enumerate(idxs):
        nx, ny = -np.sin(pyaw[idx]), np.cos(pyaw[idx])
        side = (-1) ** k
        obstacles.append({
            "x": px[idx] + side * 0.3 * nx,
            "y": py[idx] + side * 0.3 * ny,
            "r": 0.35,
        })

    inner = np.loadtxt(SCRIPT_DIR / "barreira_suavizada_interna.txt")
    outer = np.loadtxt(SCRIPT_DIR / "barreira_suavizada_externa.txt")
    return dict(px=px, py=py, pyaw=pyaw, s=s, obstacles=obstacles, inner=inner, outer=outer)


def make_controller(method, sc):
    """Devolve step(state) -> (v, w, center_xy, info)."""
    mem = {"last_idx": 0, "u_pre": np.array([0.0, 0.0])}

    if method == "simulate1":
        def step(state):
            u, info = qp.cbf_clf_qp_filter(
                u_nom=(V_REF, 0.0),
                robot_state=state,
                obstacles=sc["obstacles"],
                px=sc["px"], py=sc["py"], pyaw=sc["pyaw"], s=sc["s"],
                last_path_idx=mem["last_idx"],
                ellipse_ab=(0.30, 0.20),
                margin=0.05,
                lookahead_l=0.1,
                alpha=5,
                eps_clf=3,
                q_clf=(1.0, 10.0, 1),
                W=(100000.0, 1.0),
                p_slack=50.0,
                v_ref=V_REF,
                v_bounds=V_BOUNDS,
                w_bounds=W_BOUNDS,
                barrier_inner=sc["inner"],
                barrier_outer=sc["outer"],
            )
            mem["last_idx"] = info["idx"]
            return u[0], u[1], (state[0], state[1]), info
        return step

    if method == "paper":
        def step(state):
            goal, idx = paper_qp.paper_goal_on_path(
                sc["px"], sc["py"], sc["pyaw"], sc["s"], state,
                PAPER_L, mem["last_idx"], PAPER_D_GOAL,
            )
            mem["last_idx"] = idx
            u, info = paper_qp.paper_clf_cbf_qp(
                mem["u_pre"], state, goal, sc["obstacles"],
                sc["inner"], sc["outer"], **PAPER_PARAMS,
            )
            mem["u_pre"] = u
            x, y, th = state
            center = (x + PAPER_L * np.cos(th), y + PAPER_L * np.sin(th))
            return u[0], u[1], center, info
        return step

    raise ValueError(method)


def run(method, sc):
    step = make_controller(method, sc)
    px, py, pyaw = sc["px"], sc["py"], sc["pyaw"]
    n_path = len(px)

    x, y, th = X0
    ref_idx = 0
    prev_idx = None
    progress = 0.0

    log = {k: [] for k in ("t", "x", "y", "cx", "cy", "v", "w", "cte", "epsi",
                           "s_ref", "t_step", "t_solve", "delta", "infeasible")}

    cpu0 = time.process_time()
    steps = int(T_MAX / DT)
    lap_completed = False
    for k in range(steps):
        t0 = time.perf_counter()
        v, w, center, info = step((x, y, th))
        t_step = time.perf_counter() - t0

        # erro de seguimento: centro do robô vs spline (igual para os dois métodos)
        ref_idx, _ = nearest_path_point_closed(px, py, center[0], center[1], last_idx=ref_idx)
        dx = center[0] - px[ref_idx]
        dy = center[1] - py[ref_idx]
        cte = -np.sin(pyaw[ref_idx]) * dx + np.cos(pyaw[ref_idx]) * dy
        epsi = wrap_to_pi(th - pyaw[ref_idx])

        if prev_idx is None:
            prev_idx = ref_idx
        else:
            d_idx = ref_idx - prev_idx
            if d_idx < -n_path / 2:
                d_idx += n_path
            elif d_idx > n_path / 2:
                d_idx -= n_path
            progress += max(0.0, float(d_idx))
            prev_idx = ref_idx

        log["t"].append(k * DT)
        log["x"].append(x)
        log["y"].append(y)
        log["cx"].append(center[0])
        log["cy"].append(center[1])
        log["v"].append(v)
        log["w"].append(w)
        log["cte"].append(cte)
        log["epsi"].append(epsi)
        log["s_ref"].append(progress * (sc["s"][1] - sc["s"][0]))
        log["t_step"].append(t_step)
        log["t_solve"].append(info["qp_time"])
        log["delta"].append(info.get("delta", np.nan))
        log["infeasible"].append(info.get("infeasible", False))

        # uniciclo puro: aplica u do QP diretamente
        x += v * np.cos(th) * DT
        y += v * np.sin(th) * DT
        th = wrap_to_pi(th + w * DT)

        if progress >= n_path - 1:
            lap_completed = True
            break

    cpu_total = time.process_time() - cpu0
    log = {k: np.array(val) for k, val in log.items()}
    log["lap_completed"] = lap_completed
    log["cpu_total"] = cpu_total
    return log


def min_clearance(log, sc):
    c = np.stack([log["cx"], log["cy"]], axis=1)
    d_obs = min(np.min(np.hypot(c[:, 0] - o["x"], c[:, 1] - o["y"]) - o["r"])
                for o in sc["obstacles"])

    d_wall = np.inf
    for poly in (sc["inner"], sc["outer"]):
        A, B = poly[:-1], poly[1:]
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-12)
        for p in c[::5]:  # subamostragem chega para a distância mínima
            t = np.clip(((p - A) * AB).sum(1) / L2, 0.0, 1.0)
            Q = A + t[:, None] * AB
            d_wall = min(d_wall, np.min(np.hypot(*(p - Q).T)))
    return d_obs, d_wall


def summarize(method, logs, sc):
    ref = logs[0]  # trajetória é determinística, só o tempo varia entre runs
    t_step = np.concatenate([lg["t_step"][1:] for lg in logs]) * 1e3
    t_solve = np.concatenate([lg["t_solve"][1:] for lg in logs]) * 1e3
    cpu_per_step = np.median([lg["cpu_total"] / len(lg["t"]) for lg in logs]) * 1e3
    d_obs, d_wall = min_clearance(ref, sc)
    delta = ref["delta"]

    return {
        "metodo": method,
        "volta_completa": ref["lap_completed"],
        "tempo_volta_s": ref["t"][-1] + DT,
        "v_media": np.mean(ref["v"]),
        "cte_rms": np.sqrt(np.mean(ref["cte"] ** 2)),
        "cte_abs_medio": np.mean(np.abs(ref["cte"])),
        "cte_max": np.max(np.abs(ref["cte"])),
        "epsi_rms_deg": np.rad2deg(np.sqrt(np.mean(ref["epsi"] ** 2))),
        "dist_min_obst": d_obs,
        "dist_min_parede": d_wall,
        "passos_infeasible": int(np.sum(ref["infeasible"])),
        "passos_slack_ativa": int(np.sum(np.abs(np.nan_to_num(delta)) > 1e-6)),
        "t_step_mediana_ms": np.median(t_step),
        "t_step_media_ms": np.mean(t_step),
        "t_step_p95_ms": np.percentile(t_step, 95),
        "t_step_max_ms": np.max(t_step),
        "t_solve_mediana_ms": np.median(t_solve),
        "t_solve_media_ms": np.mean(t_solve),
        "t_solve_p95_ms": np.percentile(t_solve, 95),
        "t_solve_max_ms": np.max(t_solve),
        "cpu_por_passo_ms": cpu_per_step,
    }, t_step, t_solve


def plot_results(sc, logs, times):
    labels = list(logs.keys())

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(sc["inner"][:, 0], sc["inner"][:, 1], "k-", lw=1)
    ax.plot(sc["outer"][:, 0], sc["outer"][:, 1], "k-", lw=1)
    ax.plot(sc["px"], sc["py"], "k--", lw=0.8, label="Referência")
    for o in sc["obstacles"]:
        ax.add_patch(plt.Circle((o["x"], o["y"]), o["r"], color="gray"))
    for m in labels:
        ax.plot(logs[m]["cx"], logs[m]["cy"], lw=1.8, label=m)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "compare_trajetorias.pdf", bbox_inches="tight")

    fig, ax = plt.subplots()
    for m in labels:
        ax.plot(logs[m]["s_ref"], logs[m]["cte"], lw=1.5, label=m)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("CTE [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "compare_cte.pdf", bbox_inches="tight")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.boxplot([times[m][0] for m in labels], tick_labels=labels, showfliers=False)
    ax1.set_title("Passo completo")
    ax1.set_ylabel("tempo [ms]")
    ax2.boxplot([times[m][1] for m in labels], tick_labels=labels, showfliers=False)
    ax2.set_title("Só solve_qp")
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "compare_tempo_cpu.pdf", bbox_inches="tight")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for m in labels:
        ax1.plot(logs[m]["t"], logs[m]["v"], label=m)
        ax2.plot(logs[m]["t"], logs[m]["w"], label=m)
    ax1.set_ylabel("v [m/s]")
    ax2.set_ylabel("w [rad/s]")
    ax2.set_xlabel("Tempo [s]")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "compare_velocidades.pdf", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    sc = build_scenario()
    rows, first_logs, times = [], {}, {}

    for method in ("simulate1", "paper"):
        logs = []
        for r in range(args.runs):
            lg = run(method, sc)
            print(f"[{method}] run {r + 1}/{args.runs}: {len(lg['t'])} passos, "
                  f"volta={'sim' if lg['lap_completed'] else 'NAO'}")
            logs.append(lg)
        row, t_step, t_solve = summarize(method, logs, sc)
        rows.append(row)
        first_logs[method] = logs[0]
        times[method] = (t_step, t_solve)

    print()
    for key in rows[0]:
        vals = [f"{r[key]:.4f}" if isinstance(r[key], float) else str(r[key]) for r in rows]
        print(f"{key:22s} " + "  ".join(f"{v:>12s}" for v in vals))

    with open(SCRIPT_DIR / "compare_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_results(sc, first_logs, times)
    print(f"\nResultados em {SCRIPT_DIR / 'compare_results.csv'} e compare_*.pdf")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
