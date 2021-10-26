from fractions import Fraction
from experiments.Constants import DATA_PATHS
from experiments.utils_IO import read_dataset

import numpy as np
import pandas as pd


def not_vector(X):
    return np.array([not x for x in X])


def as_table(evaluations, methods):
    evaluations_df = []
    for dataset in evaluations:
        for col in evaluations[dataset]:
            temp = [dataset, col, evaluations[dataset][col]["correct"]]
            temp = temp + [evaluations[dataset][col][method] for method in methods]
            evaluations_df.append(temp)
    return pd.DataFrame.from_records(
        evaluations_df,
        columns=[
            "dataset",
            "column",
            "annotation",
        ]
        + methods,
    )


def as_table_cell(evaluations, methods):
    evaluations_df = []
    for d in evaluations:
        for c in evaluations[d]:
            temp = [d, c] + [evaluations[d][c][m]["correct"] for m in methods]
            temp.append(
                evaluations[d][c][methods[0]]["correct"]
                + evaluations[d][c][methods[0]]["false"]
            )
            evaluations_df.append(temp)
    return pd.DataFrame.from_records(
        evaluations_df,
        columns=[
            "dataset",
            "column",
        ]
        + methods
        + ["total # unique entries"],
    )


def calculate_dim_rates(dim_indices, true_values, predicted_values):
    dim_rates = {t: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for t in dim_indices}
    for t in dim_indices:
        y_true = np.array(true_values) == t
        y_score = np.array(predicted_values) == t
        dim_rates[t]["TP"] = sum(y_true * y_score)
        dim_rates[t]["FP"] = sum(not_vector(y_true) * y_score)
        dim_rates[t]["TN"] = sum(not_vector(y_true) * not_vector(y_score))
        dim_rates[t]["FN"] = sum(y_true * not_vector(y_score))
    return dim_rates


def calculate_metrics(df, methods):
    dim_indices = {
        "currency": 0,
        "data storage": 1,
        "length": 2,
        "mass": 3,
        "volume": 4,
    }
    dims = [
        "Overall Accuracy",
        "Currency",
        "Data storage",
        "Length",
        "Mass",
        "Volume",
    ]
    metrics = pd.DataFrame(columns=methods, index=dims)
    true_values = df["annotation"].tolist()
    for method in methods:
        predicted_values = df[method].tolist()
        dim_rates = calculate_dim_rates(dim_indices, true_values, predicted_values)
        jaccard_indices = []
        acc = {"tp": 0, "tp+fn": 0}
        for t in dim_indices:
            tp = dim_rates[t]["TP"]
            fp = dim_rates[t]["FP"]
            fn = dim_rates[t]["FN"]
            acc["tp"] += tp
            acc["tp+fn"] += tp + fn
            jaccard_indices.append(float("{:.2f}".format(tp / (tp + fp + fn))))

        acc = float("{:.2f}".format(float(acc["tp"]) / (acc["tp+fn"])))
        metrics[method] = [acc] + jaccard_indices
    return metrics


def calculate_metrics_cells(df, methods):
    datasets = list(df["dataset"].unique())
    datasets.sort()
    if "2015ReportedTaserData" in datasets:
        datasets.remove("2015ReportedTaserData")
    N = len(datasets)
    acc_table = np.zeros((N, len(methods)))
    total_table = np.zeros((N, 1))

    for index, row in df.iterrows():
        dataset = row["dataset"]
        if dataset in datasets:
            dataset_index = datasets.index(dataset)
            for i, method in enumerate(methods):
                acc_table[dataset_index, i] += row[method]
            total_table[dataset_index, 0] += row["total # unique entries"]

    for dataset in datasets:
        if dataset in datasets:
            dataset_index = datasets.index(dataset)
            for i in range(len(methods)):
                acc_table[dataset_index, i] = (
                    acc_table[dataset_index, i] / total_table[dataset_index, 0]
                )

    acc_table = np.around(acc_table, decimals=2)
    accuracy_df = pd.DataFrame.from_records(
        acc_table.T, columns=datasets, index=methods
    )
    return accuracy_df


def as_table_times(times, methods):
    times_table = []
    for dataset in times:
        for column in times[dataset]:
            temp = [dataset, column] + [times[dataset][column][m] for m in methods]
            times_table.append(temp)
    df = pd.DataFrame.from_records(times_table, columns=["dataset", "column"] + methods)
    times_df = df[methods].T.stack().reset_index(name="Runtime (sec.)")
    times_df = times_df[["level_0", "Runtime (sec.)"]]
    times_df.columns = ["Method", "Runtime (sec.)"]
    times_df["Runtime (sec.)"] = times_df["Runtime (sec.)"].apply(lambda x: np.log10(x))

    return times_df


def evaluate_prediction(truth, prediction):
    # to convert '1/2' to 0.5
    if (type(prediction) == dict) and type(prediction["magnitude"]) == str:

        if prediction["magnitude"] == "":
            prediction[
                "magnitude"
            ] = 1.0  # missing magnitude and filling it as 1 are assumed to be both correct.
        else:
            prediction["magnitude"] = float(Fraction(prediction["magnitude"]))

    if (type(prediction) == dict) and type(prediction["unit"]) == list:
        if len(prediction["unit"]) == 1:
            if type(truth["unit"]) == list:
                return (
                    (type(prediction) == dict)
                    and (truth["magnitude"] == float(prediction["magnitude"]))
                    and (
                        len(set(prediction["unit"]).intersection(set(truth["unit"])))
                        > 0
                    )
                )
            else:
                return (
                    (type(prediction) == dict)
                    and (truth["magnitude"] == float(prediction["magnitude"]))
                    and (truth["unit"] in prediction["unit"][0])
                )
        else:
            # if len(prediction["unit"]) > 1:
            #     print("multiple matches", prediction["unit"])
            return False

    else:
        return (
            (type(prediction) == dict)
            and (truth["magnitude"] == float(prediction["magnitude"]))
            and (prediction["unit"] in truth["unit"])
        )


def evaluate_identification_experiment(
    dataset, columns, method, annotations, predictions
):
    evaluations = {col: {} for col in columns}
    df = read_dataset(dataset, ALL_PATHS=DATA_PATHS)
    for col in columns:
        correct = 0
        false = 0
        unique_vals = np.unique(df[col].values)
        if method == "PUC":
            preds = predictions[col]
        else:
            preds = predictions

        for unique_val in unique_vals:
            if unique_val in annotations:
                if preds == "no unit":
                    false += 1
                elif evaluate_prediction(annotations[unique_val], preds[unique_val]):
                    correct += 1
                else:
                    false += 1
            # else:
            #     print("Not annotated!", unique_val, len(unique_val))
        evaluations[col]["correct"] = correct
        evaluations[col]["false"] = false

    return evaluations
