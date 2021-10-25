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


def as_table_times(times, methods):
    # methods = {
    # "ccut": "CCUT",
    # "grobid": "GQ",
    # "ner": "S-NER",
    # "pint": "Pint",
    # "puc": "PUC",
    # "quantulum": "Quantulum",
    # }

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
