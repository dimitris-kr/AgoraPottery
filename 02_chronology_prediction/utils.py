import json

import pandas as pd
import numpy as np
from IPython.core.display_functions import display
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
import seaborn as sns
import os
import time
from itertools import product

from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, max_error, \
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier

import torch

# CONSTANTS

f_types = {
    "vectors": ("csv", pd.read_csv, {}),
    "tensors": ("pt", torch.load, {"weights_only": True})
}

d_types_methods = {
    "text": ("tfidf", "bert"),
    "image": ("cannyhog", "resnet", "vit")
}

metrics_r = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "r2": r2_score,
    "medae": median_absolute_error,
    "maxerror:": max_error,
}

metrics_c = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

metric_params = {metric: {"average": "macro", "zero_division": 0} for metric in ["precision", "recall", "f1"]}


# READ FEATURES

def read_features(path, f_type="vectors"):
    if f_type not in f_types:
        return None

    ext, loader, params = f_types[f_type]

    # path = "../data/chronology_prediction/" + f_type + "/"
    path = os.path.abspath(os.path.join(path, f_type))

    subsets = ["train", "test"]
    methods = ["tfidf", "bert", "cannyhog", "resnet", "vit"]

    X = {}
    for subset in subsets:
        X[subset] = {}
        for method in methods:
            filename = f"X_{subset}_{method}.{ext}"
            file_path = os.path.join(path, filename)
            if os.path.exists(file_path):
                X[subset][method] = loader(file_path, **params)
                print(f"Loaded X_{subset}_{method}")

    return X


# READ TARGETS

def read_targets(path, targets, f_type="df"):
    path = os.path.abspath(os.path.join(path, "targets"))

    subsets = ["train", "test"]

    y = {}
    for subset in subsets:
        filename = f"y_{subset}.csv"
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            y[subset] = pd.read_csv(file_path)
            print(f"Loaded y_{subset}")

    targets = [target for target in targets if target in y["train"]]
    if len(targets) == 0:
        return None

    y = {
        subset: _y[targets]
        for subset, _y in y.items()
    }

    if f_type == "df":
        return y
    elif f_type == "np":
        return {subset: _y.to_numpy() for subset, _y in y.items()}
    else:
        return None

# PRINT INFO

def print_info_features(X):
    print("{")
    for subset in X.keys():
        indent = "\t"
        print(f"{indent}{subset}: " + "{")
        for method in X[subset].keys():
            indent = 2 * "\t"
            print(f"{indent}{method}: ")
            indent = 3 * "\t"
            print(f"{indent}{type(X[subset][method])}")
            print(f"{indent}shape = {X[subset][method].shape}, ")
        print("\t},")
    print("}")


def print_info_targets(y):
    print("{")
    for subset in y.keys():
        indent = "\t"
        print(f"{indent}{subset}: ", end="")

        y_type = type(y[subset])
        print()
        indent = 2 * "\t"
        print(f"{indent}{y_type}")
        print(f"{indent}shape   = {y[subset].shape}")

        if y_type is pd.DataFrame:
            print(f"{indent}columns = {list(y[subset].columns)},")
    print("}")


# LOAD BEST PARAMS

def load_best_params(path):
    # Initialize if not exists
    if not os.path.exists(path):
        return {}

    # Load
    with open(path, "r") as f:
        best_params = json.load(f)

        # Convert stringified tuples back to tuple keys
        return {
            model: {
                eval(k): v for k, v in param_dict.items()
            } for model, param_dict in best_params.items()
        }


# SAVE BEST PARAMS

def save_best_params(path, best_params, flag_new_model):
    if not flag_new_model:
        print("✅ No new tuning needed — using existing parameters.")
        return

    # Convert tuple keys to strings to make JSON serializable
    serializable_params = {
        model: {
            str(k): v for k, v in param_dict.items()
        } for model, param_dict in best_params.items()
    }

    with open(path, "w") as f:
        json.dump(serializable_params, f, indent=2)

    print(f"✅ Saved best parameters to {path}")


# COMBINE FEATURES

def combine_features(feature_sets):
    col_offset = 0
    for i in range(0, len(feature_sets)):
        n = feature_sets[i].shape[1]
        feature_sets[i].columns = [f"F{i}" for i in range(col_offset, col_offset + n)]
        col_offset += n
    return pd.concat(feature_sets, axis=1)


# CROSS VALIDATION

def cross_validation(model, folds, metrics, X, y):
    scores = {metric: [] for metric in metrics.keys()}

    convert_to_numpy = isinstance(model, XGBRegressor) or isinstance(model, XGBClassifier)

    for train_idx, val_idx in folds:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if convert_to_numpy:
            X_train, X_val, y_train, y_val = [df.to_numpy() for df in (X_train, X_val, y_train, y_val)]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        fold_scores = evaluate(y_val, y_pred, metrics)
        for metric, value in fold_scores.items():
            scores[metric].append(value)

    return {metric: (np.mean(values), values) for metric, values in scores.items()}


# SAVE CROSS VALIDATION RESULTS TO SCOREBOARD DF

def update_scoreboard(scoreboard, new_entries):
    if scoreboard.empty:
        return new_entries

    scoreboard = pd.concat([scoreboard, new_entries], ignore_index=True)
    scoreboard = scoreboard.drop_duplicates(subset=["model", "target", "features"], keep="last")
    return scoreboard


# PLOT CROSS VALIDATION RESULTS

def plt_title(model, target, features):
    title = ""
    if model:
        title += f"Model: '{model}'"
        if target or features:
            title += " | "
    if target:
        title += f"Target: '{target}'"
        if features:
            title += " | "
    if features:
        title += f"Features: '{features.upper()}'"
    return title


subplots_r = [
    {
        'metrics': ["mae", "rmse", "medae"],
        'colors': ['blue', 'green', 'orange'],
        'ylabel': 'Years',
    },
    {
        'metrics': ["r2"],
        'colors': ['purple'],
        'ylabel': 'Score',
        'ylim': 1

    }
]

subplots_c = [
    {
        'metrics': list(metrics_c.keys()),
        'colors': ['blue', 'green', 'orange', 'purple'],
        'ylabel': 'Score',
        'ylim': 1
    }
]


def plot_cv_scores(scores, model, target, method):
    plt.figure(figsize=(14, 5))

    if set(scores.keys()).issubset(metrics_r.keys()):
        subplots = subplots_r
    elif set(scores.keys()).issubset(metrics_c.keys()):
        subplots = subplots_c
    else:
        return

    for idx, subplot in enumerate(subplots):
        # rows = len(subplots) // cols + (1 if len(subplots) % 2 > 0 else 0)
        plt.subplot(1, len(subplots), idx + 1)

        min_y = 0
        max_y = 0
        for metric, color in zip(subplot['metrics'], subplot['colors']):
            if metric in scores:
                mean_val, vals = scores[metric]
                plt.plot(range(1, len(vals) + 1), vals, marker='o', label=f"{metric.upper()} per Fold", color=color)
                plt.axhline(mean_val, color=color, alpha=0.8, linestyle='--',
                            label=f"Mean {metric.upper()}: {mean_val:.2f}")

                max_y = max(max_y, max(vals))
                min_y = min(min_y, min(vals))

        plt.axhline(0, color='gray', alpha=0.5, linestyle='--')
        plt.title(', '.join(subplot['metrics']).upper() + " Across Folds", fontsize="12")
        plt.xlabel("Fold")
        plt.ylabel(subplot['ylabel'])
        plt.ylim(min_y * 1.1, subplot['ylim'] if 'ylim' in subplot else max_y * 1.2)
        plt.xticks(range(1, len(vals) + 1))
        plt.legend(ncol=len(subplot['metrics']), fontsize="8")
        plt.grid(True)

    plt.suptitle(plt_title(model, target, method), fontsize="16")
    plt.tight_layout()
    plt.show()
    # print("\n")


def plot_compare_feature_scores(model_scoreboard):
    current_metrics = set(model_scoreboard.columns) - {'model', 'target', 'features'}
    if current_metrics.issubset(metrics_r.keys()):
        subplots = subplots_r
    elif current_metrics.issubset(metrics_c.keys()):
        subplots = subplots_c
    else:
        return

    print("Compare Mean Cross Validation Scores of Feature Sets for One Model")
    model = model_scoreboard["model"].unique()[0]

    for target in model_scoreboard["target"].unique():
        df = model_scoreboard[model_scoreboard["target"] == target]

        plt.figure(figsize=(14, 6))
        x = np.arange(len(df))

        for idx, subplot in enumerate(subplots):
            # rows = len(subplots) // cols + (1 if len(subplots) % 2 > 0 else 0)
            plt.subplot(1, len(subplots), idx + 1)

            bar_width = 0.6 / len(subplot['metrics'])
            offsets = [i - (len(subplot['metrics']) - 1) / 2 for i in range(len(subplot['metrics']))]

            all_values = df[subplot['metrics']].to_numpy()
            min_y = min(0, all_values.min() * 1.2)
            max_y = subplot['ylim'] if 'ylim' in subplot else all_values.max() * 1.2

            text_gap = (max_y - min_y) * 0.02

            for metric, color, offset in zip(subplot['metrics'], subplot['colors'], offsets):
                bars = plt.bar(x + offset * bar_width, df[metric], width=bar_width, label=metric.upper(), color=color)

                for bar in bars:
                    height = bar.get_height()
                    va = 'bottom' if height > 0 else 'top'
                    pos_y = height + (text_gap if height > 0 else - text_gap)
                    pos_x = bar.get_x() + bar.get_width() / 2
                    plt.text(pos_x, pos_y, f"{height:.2f}", ha='center', va=va, fontsize=9, rotation=90)

            plt.axhline(0, color='gray', alpha=0.5, linestyle='--')
            plt.xticks(ticks=x, labels=df["features"], rotation=45, ha='right')
            plt.ylabel(subplot['ylabel'])
            plt.ylim(min_y, max_y)
            plt.title(', '.join(subplot['metrics']).upper(), fontsize="12")
            plt.legend()
            plt.grid(True)

        plt.suptitle(plt_title(model, target, ""), fontsize="16")
        plt.tight_layout()
        plt.show()


def plot_prediction_scores(scores, model, target, features):
    print("\n")
    if set(scores.keys()).issubset(metrics_r.keys()):
        subplots = subplots_r
    elif set(scores.keys()).issubset(metrics_c.keys()):
        subplots = subplots_c
    else:
        return
    plt.figure(figsize=(8 * len(subplots), 5))
    for idx, subplot in enumerate(subplots):
        plt.subplot(1, len(subplots), idx + 1)

        metrics = subplot['metrics']
        colors = subplot['colors']
        values = [scores[metric] for metric in subplot['metrics']]

        bars = plt.barh(metrics, values, color=colors)

        for bar, val in zip(bars, values):
            plt.text(
                val + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va='center', fontsize=10
            )

        plt.xlabel(subplot['ylabel'])
        plt.ylabel("Metrics")
        plt.xlim(min([0] + values) * 1.1, subplot['ylim'] if 'ylim' in subplot else max(values) * 1.2)

    plt.suptitle("Prediction Evaluation | " + plt_title(model, target, features), fontsize="12")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()


# RUN CROSS VALIDATION

def run_cv(model_name, model_class, best_params, folds, metrics, X, y, method, target, enable_plots=True):
    if best_params:
        if (method, target) not in best_params: return
        _best_params = best_params[(method, target)]["params"]

        model = model_class(**_best_params)
    else:
        model = model_class()
    scores = cross_validation(model, folds, metrics, X, y)
    if enable_plots: plot_cv_scores(scores, model_name, target, method)

    return {"model": model_name, "target": target, "features": method,
            **{metric: scores[metric][0] for metric in metrics}}


# RUN CROSS VALIDATION FOR ALL FEATURE SETS AND TARGETS

def run_cv_all(model_name, model_class, best_params, folds, metrics, X, y, enable_plots=True):
    if enable_plots: print("Cross Validation Score Progression")
    model_scoreboard = []
    for target, _y in y.items():
        for method, _X in X.items():
            scores = run_cv(model_name, model_class, best_params, folds, metrics, _X, _y, method, target, enable_plots)
            model_scoreboard.append(scores)

    return pd.DataFrame(model_scoreboard)


## HYPERPARAMETER TUNING

def hyperparameter_tuning(model_class, param_grid, folds, metrics, X, y, deciding_metric):
    start_time = time.time()
    param_scores = []
    for params in ParameterGrid(param_grid):
        model = model_class(**params)
        s = cross_validation(model, folds, metrics, X, y)
        param_scores.append((params, s[deciding_metric][0]))  # [0] is the mean across folds

    calc_best = max if deciding_metric == "r2" or deciding_metric in metrics_c else min
    best_params, best_score = calc_best(param_scores, key=lambda x: x[1])

    execution_time = time.time() - start_time

    return {
        "params": best_params,
        deciding_metric: best_score,
        "time": execution_time,
    }


def get_column_width(values):
    values = list(map(str, values))
    return len(max(values, key=len))


def get_column_widths(targets, feature_sets, deciding_metric, param_grid):
    column_widths = {
        "target": get_column_width(targets + ["target"]),
        "feature_set": get_column_width(feature_sets + ["feature_set"]),
        deciding_metric: get_column_width([deciding_metric, "0000.0000"]),
    }

    for key, values in param_grid.items():
        if len(values) > 1:
            column_widths[key] = get_column_width(values + [key])

    return column_widths


def print_row(column_widths, values, col_divider="|", padding_char=" "):
    for col, width in column_widths.items():
        value = str(values[col])
        padding = width - len(value) + 1
        print(col_divider + (padding_char * padding) + value, end=padding_char)
    print(col_divider)


def print_row_divider(column_widths):
    print_row(
        column_widths,
        {col: "" for col in column_widths.keys()},
        col_divider="+",
        padding_char="-"
    )


def print_row_header(column_widths):
    print_row_divider(column_widths)
    print_row(column_widths, {header: header for header in column_widths.keys()})
    print_row_divider(column_widths)


def get_print_values(feature_set, target, deciding_metric, hp_result):
    print_values = hp_result["params"].copy()
    print_values["target"] = target
    print_values["feature_set"] = feature_set
    print_values[deciding_metric] = f"{hp_result[deciding_metric]:.4f}"
    return print_values


def print_best_params(column_widths, best_params, deciding_metric):
    print_row_header(column_widths)
    target_prev = ""
    for (method, target), hp_result in best_params.items():
        if target_prev and target_prev != target: print_row_divider(column_widths)
        print_row(column_widths, get_print_values(method, target, deciding_metric, hp_result))
        target_prev = target
    print_row_divider(column_widths)


def fmt_time(seconds):
    time_str = ""
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    if hour > 0: time_str += f"{int(hour)}h "
    if min > 0: time_str += f"{int(min)}m "
    time_str += f"{int(sec)}s"
    return time_str


def get_execution_time(best_params):
    return fmt_time(sum([hp_result["time"] for hp_result in best_params.values()]))


def run_hp_all(model_class, param_grid, folds, metrics, X, y, deciding_metric, column_widths, verbose=False):
    model_best_params = {}

    if verbose: print_row_header(column_widths)

    execution_time = 0
    target_prev = ""
    for target, _y in y.items():
        if verbose and target_prev and target_prev != target: print_row_divider(column_widths)
        for method, _X in X.items():
            hp_result = hyperparameter_tuning(model_class, param_grid, folds, metrics, _X, _y, deciding_metric)
            model_best_params[(method, target)] = hp_result.copy()

            execution_time += hp_result["time"]
            if verbose: print_row(column_widths, get_print_values(method, target, deciding_metric, hp_result))

        target_prev = target
    if verbose: print_row_divider(column_widths)
    if verbose: print(f"Finished in {fmt_time(execution_time)}")

    return model_best_params


def combine_features_all(X, scale=False):
    X_combos = {}
    for text_method in d_types_methods["text"]:
        for image_method in d_types_methods["image"]:
            X_combo = combine_features([X[text_method], X[image_method]])
            if scale: X_combo = scale_feature_set(X_combo)
            X_combos[f"{text_method} + {image_method}"] = X_combo
    return X_combos


def scale_feature_set(X):
    scaler = StandardScaler()
    return pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )


def scale_all(X):
    return {
        subset: {
            method: scale_feature_set(_X) if method != "tfidf" else _X
            for method, _X in X[subset].items()
        } for subset in X.keys()
    }


def reduce_components(X, reducer_fitted=None, n_components=0.95, random_state=42):
    X = scale_feature_set(X)
    if reducer_fitted:
        reducer = reducer_fitted
        X_reduced = reducer.transform(X)
    else:
        reducer = PCA(n_components=n_components, random_state=random_state)
        X_reduced = reducer.fit_transform(X)

    X_reduced = pd.DataFrame(
        X_reduced,
        columns=X.columns[:X_reduced.shape[1]],
        index=X.index
    )
    return X_reduced, reducer


def reduce_all(X, n_components=0.95, random_state=42):
    reducers = {}
    X_reduced = {subset: {} for subset in X.keys()}
    for method in d_types_methods["text"] + d_types_methods["image"]:
        if method == "tfidf":
            for subset in X.keys(): X_reduced[subset][method] = X[subset][method]
            continue

        X_reduced["train"][method], reducers[method] = reduce_components(X["train"][method], n_components=n_components,
                                                                         random_state=random_state)

        X_reduced["test"][method], _ = reduce_components(X["test"][method], reducer_fitted=reducers[method])

    return X_reduced


def add_all_feature_combos(X, scale=False):
    for subset in X.keys():
        X_combos = combine_features_all(X[subset], scale=scale)

        X[subset].update(X_combos)
    return X


def encode_labels(y, target):
    le = LabelEncoder()

    target_enc = f"{target}_encoded"

    y["train"][target_enc] = le.fit_transform(y["train"][target])
    y["test"][target_enc] = le.transform(y["test"][target])

    for label, encoding in zip(le.classes_, le.transform(le.classes_)):
        print(f"{encoding} --> {label}")

    return y, target_enc, le


# EVALUATE SCOREBOARD

def get_unique(df, columns):
    return (df[column].unique().tolist() for column in columns)


def get_color_map(values, palette_name):
    palette = sns.color_palette(palette_name, len(values))
    return dict(zip(values, palette))


def plot_metric_in_groups(df, metric, group_elems, single_elems, palette, lim=None, step=None):
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df,
        x=metric,
        y=group_elems,
        hue=single_elems,
        palette=palette,
        errorbar=None
    )

    if not lim: lim = np.max(df[metric])
    if not step:
        plt.xlim(0, 1.1 * lim)
    else:
        plt.xticks(np.arange(0, lim + step, step))

    plt.title(
        f"{metric.upper()} per {group_elems.upper()} | Grouped by {single_elems.upper()} | Target = {df['target'].values[0]}")
    plt.xlabel(metric.upper())
    plt.ylabel(group_elems.upper())
    plt.legend(title=single_elems.upper())
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metric_points(df, metric, group_elems, point_elems, palette, lim=None, step=None):
    plt.figure(figsize=(12, 8))
    sns.pointplot(data=df, x=group_elems, y=metric, hue=point_elems,
                  palette=palette, markers="o")
    if not lim: lim = np.max(df[metric])
    if not step:
        plt.ylim(1.1 * np.min(df[metric].tolist() + [0.0]), 1.1 * lim)
    else:
        plt.yticks(np.arange(1.1 * np.min(df[metric].tolist() + [0.0]), lim + step, step))
    plt.title(
        f"{metric.upper()} per {group_elems.upper()} and {point_elems.upper()} | Target = {df['target'].values[0]}")
    plt.xticks(rotation=45)
    plt.xlabel(metric.upper())
    plt.ylabel(group_elems.upper())
    plt.legend(title=point_elems.upper())
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pivot_table(df, metric, models, features):
    # Create pivot table with explicit order
    pivot = df.pivot_table(
        index="model",
        columns="features",
        values=metric
    ).reindex(index=models, columns=features)

    # Plot
    plt.figure(figsize=(10, 6))

    cmap = "mako"
    if metric in metrics_r and metric != "r2": cmap += "_r"

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5)
    better_dir = ""
    if metric in metrics_c or metric == "r2":
        better_dir = "Higher"
    elif metric in metrics_r and metric != "r2":
        better_dir = "Lower"
    plt.title(
        f"{metric.upper()} Pivot Table (Heatmap)" + f" | {better_dir} is Better" if better_dir else "" + f" | Target = {df['target'].values[0]}")

    plt.xlabel("FEATURES")
    plt.ylabel("MODELS")
    plt.tight_layout()
    plt.show()


def plot_best_of_each(df, metric, group_elems, find_best, palette, lim=None, step=None):
    if metric in metrics_r and metric != "r2":
        best_of_each = df.loc[df.groupby(group_elems)[metric].idxmin()]
        asc = True
    elif metric in metrics_c or metric == "r2":
        best_of_each = df.loc[df.groupby(group_elems)[metric].idxmax()]
        asc = False
    else:
        return

    best_of_each = best_of_each.sort_values(metric, ascending=asc)

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        data=best_of_each,
        x=metric,
        y=group_elems,
        hue=find_best,
        palette=palette
    )

    # Add model name as a label on top of each bar
    for container, label in zip(ax.containers, best_of_each[find_best].unique()):
        ax.bar_label(container, labels=[label] * len(container), fontsize=9, label_type='edge')

    if not lim: lim = np.max(best_of_each[metric])
    if not step:
        plt.xlim(0, 1.1 * lim)
    else:
        plt.xticks(np.arange(0, lim + step, step))

    plt.title(f"Best {find_best.upper()} per {group_elems.upper()} | Target = {df['target'].values[0]}")
    plt.xlabel(metric.upper())
    plt.ylabel(group_elems.upper())
    plt.legend(title=find_best.upper())
    plt.tight_layout()
    plt.show()


def plot_best_by_data_type(df, metric, palette, top_n=5, lim=None, step=None):
    d_types_methods_full = d_types_methods.copy()
    d_types_methods_full["combo"] = tuple(
        f"{t} + {i}" for t in d_types_methods["text"] for i in d_types_methods["image"])
    for d_type, features in d_types_methods_full.items():
        subset = df.loc[df["features"].isin(features)]
        if subset.empty:
            continue

        top_n_set = subset.sort_values(metric, ascending=metric in metrics_r and metric != "r2").head(top_n)
        top_n_set["pair"] = top_n_set["model"] + " w/ " + top_n_set["features"]

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_n_set, x=metric, y="pair", hue="model", palette=palette)
        plt.title(f"Top {top_n} MODEL & FEATURE pair | {d_type.upper()} Data | Target = {df['target'].values[0]}")
        plt.xlabel(metric.upper())
        plt.ylabel("MODEL & FEATURE pair")

        lim_curr = lim if lim else np.max(top_n_set[metric])
        if not step:
            plt.xlim(0, 1.1 * lim_curr)
        else:
            plt.xticks(np.arange(0, lim_curr + step, step))

        plt.grid(True, axis="x", linestyle="--", alpha=0.6)
        for index, value in enumerate(top_n_set[metric]):
            plt.text(value + 0.01, index, f"{value:.2f}", va='center', fontsize=9)
        plt.tight_layout()
        plt.show()


def evaluate(y_true, y_pred, metrics):
    scores = {}
    for metric, get_metric_score in metrics.items():
        metric_score = get_metric_score(y_true, y_pred, **metric_params.get(metric, {}))

        if metric == "rmse": metric_score = np.sqrt(metric_score)
        scores[metric] = float(metric_score)
    return scores


def plot_confusion_matrix(cm, le, model_name, features):
    print("\n")
    # Normalize confusion matrix by rows (true labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Calculate true class counts
    true_counts = cm.sum(axis=1)
    true_labels = [f"{label}\n({count} samples)" for label, count in zip(le.classes_, true_counts)]

    # Predicted class labels
    predicted_labels = le.classes_

    # Plot
    plt.subplots(figsize=(8, 6))
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".0%",
        cmap="Blues",
        xticklabels=predicted_labels,
        yticklabels=true_labels
    )

    # Rotate y-axis labels to horizontal
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Move x-axis labels to the top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Customize colorbar to show % ticks
    cbar = ax.collections[0].colorbar
    ticks = np.arange(0, 1.1, 0.2)
    cbar.ax.set_yticks(ticks)
    cbar.ax.set_yticklabels([f"{int(t * 100)}%" for t in ticks])

    plt.grid(False)
    plt.title(f"Confusion Matrix | Model: {model_name} | Features: {features.upper()}", pad=20)
    plt.xlabel("Predicted Label", labelpad=15)
    plt.ylabel("True Label (Count)", labelpad=15)
    plt.tight_layout()
    plt.show()


def autopct_with_counts(values):
    def inner_autopct(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f"{pct:.0f}%\n({count})"

    return inner_autopct


def plot_pie(ax, values, label_prefixes, class_name, reverse=False):
    labels = [f"Predicted as {class_name}", f"Not Predicted as {class_name}"]
    if reverse:
        labels = reversed(labels)
    labels = [f"{prefix}: {label}" for prefix, label in zip(label_prefixes, labels)]

    wedges, _, _ = ax.pie(
        values,
        colors=["#4a90e2", "#e74c3c"],
        startangle=90,
        autopct=autopct_with_counts(values),
        wedgeprops={"edgecolor": "white"}
    )
    title = f"{class_name} Samples"
    if reverse:
        title = "Non-" + title
    ax.set_title(title, fontsize=12, pad=20, weight="bold")
    ax.legend(wedges, labels, loc="upper center", fontsize=8, bbox_to_anchor=(0.5, 1.08))


def plot_cm_pies(cm, le):
    print("\n")
    classes = le.classes_
    total_samples = cm.sum()

    fig, axes = plt.subplots(nrows=len(classes), ncols=2, figsize=(8, 4.5 * len(classes)))

    for i, class_name in enumerate(classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total_samples - (TP + FN + FP)

        # --- TP/FN Pie Chart ---
        plot_pie(axes[i, 0], [TP, FN], ["TP", "FN"], class_name)

        # --- TN/FP Pie Chart ---
        plot_pie(axes[i, 1], [TN, FP], ["TN", "FP"], class_name, reverse=True)

    plt.show()


# Regression Models Init, Fit, Predict with Extra Info

def initialize_model(model_class, params):
    if model_class == RandomForestRegressor:
        return model_class(**params)
    elif model_class == LGBMRegressor:
        model_alphas = {
            "y_pred": 0.5,
            "CI_lower": 0.025,
            "CI_upper": 0.975
        }
        params["objective"] = "quantile"
        return {name: model_class(**params, alpha=alpha) for name, alpha in model_alphas.items()}
    elif model_class == Ridge:
        return BaggingRegressor(estimator=model_class(**params), n_estimators=50, bootstrap=True, n_jobs=-1)
    else:
        return None


# Probability that the true year lies within ±N years of prediction
def fixed_CI_probability(y_std, N):
    y_std = max(y_std, 1e-6)  # avoid division by zero
    z = N / y_std
    return norm.cdf(z) - norm.cdf(-z)


def predict_with_std(model, X_test, y_test):
    # Get predictions from all estimators
    estimators = model.estimators_
    all_preds = np.stack([est.predict(X_test) for est in estimators])

    results = pd.DataFrame({
        "y_true": y_test,
        "y_pred": np.mean(all_preds, axis=0),
        "y_std": np.std(all_preds, axis=0),
    })

    # Assume the prediction errors follow a normal distribution
    # Confidence Interval where prediction has 95% confidence
    z = norm.ppf(0.975)  # ≈ 1.96
    results["CI_lower"] = results["y_pred"] - z * results["y_std"]
    results["CI_upper"] = results["y_pred"] + z * results["y_std"]

    N = 10
    results[f"confidence_±{N}"] = results["y_std"].apply(
        lambda std: fixed_CI_probability(std, N)
    )

    # Absolute Error
    results["error"] = (results["y_pred"] - results["y_true"]).abs()

    return results


def fit_multimodel(models, X_train, y_train):
    for name in models.keys():
        models[name].fit(X_train, y_train)
    return models


def predict_multimodel(models, X_test, y_test):
    results = {"y_true": y_test}
    for name, model in models.items():
        results[name] = model.predict(X_test)

    # Absolute Error
    results["error"] = (results["y_pred"] - results["y_true"]).abs()

    return pd.DataFrame(results)


# Regression Display and Plot Prediction Results

def get_result_subsets(results, samples):
    results_rand = results.sample(n=samples, random_state=42)
    results_best = results.nsmallest(samples, "error").sort_values(by="error", ascending=True)
    results_worst = results.nlargest(samples, "error").sort_values(by="y_true", ascending=False)

    results_rand.insert(0, "SAMPLE", "RANDOM")
    results_best.insert(0, "SAMPLE", "BEST")
    results_worst.insert(0, "SAMPLE", "WORST")

    return [results_rand, results_best, results_worst]


def print_top_1(result_subsets):
    print("\n")
    print("Example Sample Predictions:")
    display(pd.concat([df.head(1) for df in result_subsets]))
    print("\n")


def plot_true_vs_pred(result_tables):
    plt.figure(figsize=(14, result_tables[0].shape[0] * 0.3 * 2))

    for idx, results in enumerate(result_tables):
        samples = len(results)
        # plt.subplot(1, len(result_tables), idx + 1)

        plt.subplot2grid((2, 2), (0, 0) if idx == 0 else (1, idx - 1), colspan=(2 if idx == 0 else 1))

        results_sorted = results.sort_values(by="y_true").reset_index(drop=True)
        for i, row in results_sorted.iterrows():
            # Confidence interval line
            plt.plot([row["CI_lower"], row["CI_upper"]], [i, i], color='gray', linewidth=1, alpha=0.3)

            # True year dot
            plt.scatter(row["y_true"], i, color='seagreen', label='True' if i == 0 else "", zorder=3, s=40, alpha=0.5)

            # Predicted year dot
            plt.scatter(row["y_pred"], i, color='royalblue', label='Predicted' if i == 0 else "", zorder=3, s=40,
                        alpha=0.5)

        if idx == 0 or idx == 1: plt.ylabel("Data Entries")
        if idx == 0: plt.legend(loc="upper right")
        plt.ylim(samples, -1)
        plt.yticks([])
        plt.xlabel("Year")
        plt.title(f"{samples} {results["SAMPLE"].unique()[0]} Sample Predictions")

    plt.suptitle("Prediction vs True Year with Confidence Interval")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# Train - Validation Split (Further Split)
def train_val_split(X, y):
    indices = np.arange(y["train"].shape[0])
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

    train_idx = torch.tensor(train_idx, dtype=torch.int64)
    val_idx = torch.tensor(val_idx, dtype=torch.int64)

    X = {
        "train": {method: tensors[train_idx] for method, tensors in X["train"].items()},
        "val": {method: tensors[val_idx] for method, tensors in X["train"].items()},
        "test": X["test"]
    }

    y = {
        "train": y["train"][train_idx],
        "val": y["train"][val_idx],
        "test": y["test"]
    }

    return X, y

# Get Data Dimensions
def get_dimensions(X, y, le=None, verbose=True):
    X_dimensions = {
        method: _X.shape[1]
        for method, _X in X["train"].items()
    }

    if le:
        # CLASSIFICATION → dims = number of classes
        y_dimensions = len(le.classes_)
    else:
        # REGRESSION -> dims = number of continuous variables
        y_dimensions = y["train"].shape[1]

    if verbose:
        print("X Dimensions:", X_dimensions)
        print("y Dimensions:", y_dimensions)

    return X_dimensions, y_dimensions

def get_device():
    print("PyTorch Version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available")
        print("GPU:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")
    print("Using Device:", device)
    return device