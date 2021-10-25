def update_times_taken(dataset, method, times, times_taken):
    for column in times:
        if column not in times_taken[dataset]:
            times_taken[dataset][column] = {}
        times_taken[dataset][column][method] = times[column]

    return times_taken


def run(data_paths, datasets, methods):
    type_predictions = {d: {} for d in datasets}
    unit_predictions = {d: {} for d in datasets}
    times_taken = {dataset: {} for dataset in datasets}

    for dataset in datasets:
        print("running on ", dataset, "...")

        df = read_dataset(dataset, ALL_PATHS=DATA_PATHS)
        cols = data_paths[dataset]["columns"]

        for method in methods:
            if method == "PUC":
                t, u, times = run_dimension_experiments(df, cols)
            else:
                t, u, times = run_competitor_column_experiments(df, cols, method)

            type_predictions[dataset][method] = t
            unit_predictions[dataset][method] = u
            times_taken = update_times_taken(dataset, method, times, times_taken)

    return type_predictions, unit_predictions, times_taken
    # ccut_type_predictions = {}
    # grobid_type_predictions = {}
    # quantulum_type_predictions = {}
    # pint_type_predictions = {}
    # ner_type_type_predictions = {}
    # spacy_type_type_predictions = {}

    # # print(type_predictions, unit_predictions)
    # method = "quantulum"
    # t, times = run_competitor_column_experiments(df, columns, method)
    # quantulum_type_predictions[dataset] = t
    # times_taken = update_times_taken(dataset, method, times, times_taken)
    #
    # method = "ccut"
    # t, times = run_competitor_column_experiments(df, columns, method)
    # ccut_type_predictions[dataset] = t
    # times_taken = update_times_taken(dataset, method, times, times_taken)
    #
    # method = "grobid"
    # t, times = run_competitor_column_experiments(df, columns, method)
    # grobid_type_predictions[dataset] = t
    # times_taken = update_times_taken(dataset, method, times, times_taken)
    #
    # method = "pint"
    # t, times = run_competitor_column_experiments(df, columns, method)
    # pint_type_type_predictions[dataset] = t
    # times_taken = update_times_taken(dataset, method, times, times_taken)
    #
    # method = "ner"
    # t, times = run_competitor_column_experiments(df, columns, method)
    # ner_type_type_predictions[dataset] = t
    # times_taken = update_times_taken(dataset, method, times, times_taken)

    # method = 'spacy'
    # t, times = run_competitor_column_experiments(df, columns, method)
    # spacy_type_type_predictions[dataset] = t
    # times_taken = update_times_taken(dataset, method, times, times_taken)


def evaluate_predictions(predictions, input_path):
    # Evaluate dimension predictions
    with open(input_path + "dimensions.json", "r") as fp:
        annotations = json.load(fp)

    evaluations = {dataset: {} for dataset in predictions}
    for d in predictions:
        columns = list(predictions[d]["PUC"].keys())
        for c in columns:
            evaluations[d][c] = {"correct": annotations[d][c]}
            for m in predictions[d]:
                evaluations[d][c][m] = predictions[d][m][c]

    return evaluations


def report_results(predictions, times_taken, evaluations, methods, output_path):
    # Save results
    np.save(output_path + "dimension_predictions.npy", predictions)
    np.save(output_path + "times_taken.npy", times_taken)
    np.save(output_path + "dimension_evaluations.npy", evaluations)

    # Calculate the results and putting them in dataframes
    evaluations_df = as_table(evaluations, methods)
    times_df = as_table_times(times_taken, methods)

    # Save the performance
    metrics_df = calculate_metrics(evaluations_df, methods)
    metrics_df.to_latex(output_path + "evaluations.tex")

    # Vizualize the results (normalized confusion matrices and runtimes)
    plot_hintons(evaluations_df, output_path, methods)
    plot_runtimes(times_df, output_path)


def main(data_paths, datasets, methods, input_path, output_path):

    predictions, _, times_taken = run(data_paths, datasets, methods)
    evaluations = evaluate_predictions(predictions, input_path)
    report_results(predictions, times_taken, evaluations, methods, output_path)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "../src/")
    from experiments.utils_evals import as_table, as_table_times, calculate_metrics
    from experiments.utils_experiment import (
        run_dimension_experiments,
        run_competitor_column_experiments,
        # evaluate_column_measurement_type_experiment,
        read_dataset,
    )
    from experiments.utils_viz import plot_hintons, plot_runtimes
    from experiments.Constants import DATA_PATHS, DATASETS, INPUT_ROOT, OUTPUT_ROOT

    import json
    import numpy as np

    methods = ["Quantulum", "Pint", "PUC"]
    main(DATA_PATHS, DATASETS, methods, INPUT_ROOT, OUTPUT_ROOT)
