import argparse
import csv
from pathlib import Path

import numpy as np

import ml_cbf
from ml_cbf_experiment import DEFAULT_BASELINE_PARAMS, run_episode


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina SVM + FGO para parametros ML-CBF continuos no Pack6."
    )
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument(
        "--episodes-per-sample",
        "--episodes-per-candidate",
        dest="episodes_per_sample",
        type=int,
        default=1,
    )
    parser.add_argument("--horizon", type=float, default=55.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--record-stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feasible-min-progress", type=float, default=0.95)
    parser.add_argument("--feasible-min-clearance", type=float, default=-3.0)
    parser.add_argument("--max-qp-failures", type=int, default=0)
    parser.add_argument("--svm-C", type=float, default=10.0)
    parser.add_argument("--svm-degree", type=int, default=7)
    parser.add_argument("--svm-c1", type=float, default=0.8)
    parser.add_argument("--svm-c2", type=float, default=0.5)
    parser.add_argument("--fgo-random-samples", type=int, default=2500)
    parser.add_argument("--fgo-iterations", type=int, default=10)
    parser.add_argument("--fgo-population", type=int, default=80)
    parser.add_argument("--model-out", type=Path, default=ml_cbf.DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--report-out",
        type=Path,
        default=BASE_DIR / "ml_cbf_training_report.csv",
    )
    parser.add_argument(
        "--selected-out",
        type=Path,
        default=BASE_DIR / "ml_cbf_fgo_params.csv",
    )
    return parser.parse_args()


def is_feasible(metrics, args):
    min_clearance = min(metrics["min_obstacle_clearance"], metrics["min_barrier_clearance"])
    return (
        metrics["progress_ratio"] >= args.feasible_min_progress
        and not metrics["collided"]
        and min_clearance >= args.feasible_min_clearance
        and int(metrics["qp_failures"]) <= args.max_qp_failures
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    sampled_params = ml_cbf.sample_parameter_matrix(args.samples, seed=args.seed)
    baseline = np.array(
        [[DEFAULT_BASELINE_PARAMS[name] for name in ml_cbf.PARAM_NAMES]],
        dtype=float,
    )
    param_matrix = np.vstack([baseline, sampled_params])

    train_params = []
    train_scores = []
    train_labels = []
    report_rows = []

    for sample_id, vector in enumerate(param_matrix):
        params = ml_cbf.vector_to_params(vector)
        episode_scores = []
        episode_labels = []

        for episode_idx in range(args.episodes_per_sample):
            variation_id = sample_id * args.episodes_per_sample + episode_idx
            metrics = run_episode(
                params=params,
                variation_id=variation_id,
                horizon=args.horizon,
                dt=args.dt,
                record_stride=args.record_stride,
                record_history=False,
            )

            label = 1 if is_feasible(metrics, args) else -1
            score = float(metrics["score"])
            episode_scores.append(score)
            episode_labels.append(label)
            train_params.append(vector)
            train_scores.append(score)
            train_labels.append(label)

            report_rows.append(
                {
                    "sample_id": sample_id,
                    "episode": episode_idx,
                    "variation_id": variation_id,
                    "score": score,
                    "label": label,
                    "completed": metrics["completed"],
                    "collided": metrics["collided"],
                    "progress_ratio": metrics["progress_ratio"],
                    "qp_failures": metrics["qp_failures"],
                    "mean_abs_cte": metrics["mean_abs_cte"],
                    "mean_abs_dv": metrics["mean_abs_dv"],
                    "total_abs_dv": metrics["total_abs_dv"],
                    "min_obstacle_clearance": metrics["min_obstacle_clearance"],
                    "min_barrier_clearance": metrics["min_barrier_clearance"],
                    **params,
                }
            )

        feasible_count = sum(1 for label in episode_labels if label > 0)
        print(
            f"sample {sample_id:03d}: mean_score={np.mean(episode_scores):.2f} "
            f"feasible={feasible_count}/{len(episode_labels)} params={params}"
        )

    train_params = np.asarray(train_params, dtype=float)
    train_scores = np.asarray(train_scores, dtype=float)
    train_labels = np.asarray(train_labels, dtype=int)

    if len(np.unique(train_labels)) < 2:
        print(
            "Aviso: o treino so encontrou uma classe. "
            "Ajusta --feasible-min-progress ou aumenta --samples para uma SVM melhor."
        )

    model = ml_cbf.MLCBFParameterModel.fit(
        param_matrix=train_params,
        scores=train_scores,
        labels=train_labels,
        svm_degree=args.svm_degree,
        svm_c1=args.svm_c1,
        svm_c2=args.svm_c2,
        svm_C=args.svm_C,
        seed=args.seed,
        fgo_random_samples=args.fgo_random_samples,
        fgo_iterations=args.fgo_iterations,
        fgo_population=args.fgo_population,
    )
    model.save(args.model_out)
    write_csv(args.report_out, report_rows)

    best_params = model.best_params
    selected_row = {
        "fgo_predicted_score": float(model._predict_score(model.best_normal_params)[0]),
        "svm_feasibility": float(model.svm.decision_function(model.best_normal_params)[0]),
        **best_params,
    }
    write_csv(args.selected_out, [selected_row])

    print(f"modelo guardado em: {args.model_out}")
    print(f"relatorio guardado em: {args.report_out}")
    print(f"parametros FGO guardados em: {args.selected_out}")
    print(f"amostras avaliadas: {len(train_params)}")
    print(f"viaveis: {int(np.sum(train_labels > 0))} | inviaveis: {int(np.sum(train_labels < 0))}")
    print(f"parametros escolhidos pela FGO: {best_params}")


if __name__ == "__main__":
    main()
