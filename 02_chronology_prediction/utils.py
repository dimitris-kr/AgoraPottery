import json

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, max_error, \
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

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
    "rmse": mean_squared_error,
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

def read_targets(path, targets):
    path = os.path.abspath(os.path.join(path, "targets"))

    subsets = ["train", "test"]

    y = {}
    for subset in subsets:
        filename = f"y_{subset}.csv"
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            y[subset] = pd.read_csv(file_path)
            print(f"Loaded y_{subset}")

    targets =  [target for target in targets if target in y["train"]]
    if len(targets) == 0:
        return None

    y = {subset: _y[targets] for subset, _y in y.items()}
    return y


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

    convert_to_numpy = isinstance(model, XGBRegressor)

    for train_idx, val_idx in folds:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if convert_to_numpy:
            X_train, X_val, y_train, y_val = [df.to_numpy() for df in (X_train, X_val, y_train, y_val)]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        for metric, get_metric_score in metrics.items():
            metric_score = get_metric_score(y_val, y_pred, **metric_params.get(metric, {}))

            if metric == "rmse": metric_score = np.sqrt(metric_score)
            scores[metric].append(metric_score)

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


subplots = [
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


def plot_cv_scores(scores, model, target, method, cols=2):
    plt.figure(figsize=(14, 5))

    for idx, subplot in enumerate(subplots):
        rows = len(subplots) // cols + (1 if len(subplots) % 2 > 0 else 0)
        plt.subplot(rows, cols, idx + 1)

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


def plot_compare_feature_scores(model_scoreboard, cols=2):
    print("Compare Mean Cross Validation Scores of Feature Sets for One Model")
    model = model_scoreboard["model"].unique()[0]

    for target in model_scoreboard["target"].unique():
        df = model_scoreboard[model_scoreboard["target"] == target]

        plt.figure(figsize=(14, 6))
        x = np.arange(len(df))

        for idx, subplot in enumerate(subplots):
            rows = len(subplots) // cols + (1 if len(subplots) % 2 > 0 else 0)
            plt.subplot(rows, cols, idx + 1)

            bar_width = 0.6 / len(subplot['metrics'])
            offsets = [i - (len(subplot['metrics']) - 1) / 2 for i in range(len(subplot['metrics']))]

            for metric, color, offset in zip(subplot['metrics'], subplot['colors'], offsets):
                plt.bar(x + offset * bar_width, df[metric], width=bar_width, label=metric.upper(), color=color)

            plt.axhline(0, color='gray', alpha=0.5, linestyle='--')
            plt.xticks(ticks=x, labels=df["features"], rotation=45, ha='right')
            plt.ylabel(subplot['ylabel'])
            plt.title(', '.join(subplot['metrics']).upper(), fontsize="12")
            plt.legend()
            plt.grid(True)

        plt.suptitle(plt_title(model, target, ""), fontsize="16")
        plt.tight_layout()
        plt.show()


# RUN CROSS VALIDATION

def run_cv(model_name, model_class, best_params, folds, metrics, X, y, method, target, enable_plots=True):
    if best_params:
        if (method, target) not in best_params: return
        model = model_class(**best_params[(method, target)])
    else:
        model = model_class()
    scores = cross_validation(model, folds, metrics, X, y)
    if enable_plots: plot_cv_scores(scores, model_name, target, method)

    return {"model": model_name, "target": target, "features": method, **{metric: scores[metric][0] for metric in metrics}}


# RUN CROSS VALIDATION FOR ALL FEATURE SETS AND TARGETS

def run_cv_all(model_name, model_class, best_params, folds, metrics, X, y, enable_plots=True):
    if enable_plots: print("Cross Validation Score Progression")
    model_scoreboard = []
    for target, _y in y["train"].items():
        # Single Feature Sets
        for method, _X in X["train"].items():
            scores = run_cv(model_name, model_class, best_params, folds, metrics, _X, _y, method, target, enable_plots)
            model_scoreboard.append(scores)

        # Combined Feature Sets (Text+Image)
        for text_method in d_types_methods["text"]:
            for image_method in d_types_methods["image"]:
                _X = combine_features([X["train"][text_method], X["train"][image_method]])
                method = f"{text_method} + {image_method}"

                scores = run_cv(model_name, model_class, best_params, folds, metrics, _X, _y, method, target, enable_plots)
                model_scoreboard.append(scores)

    return pd.DataFrame(model_scoreboard)


## HYPERPARAMETER TUNING

def hyperparameter_tuning(model_class, param_grid, folds, metrics, X, y, deciding_metric, verbose=False):
    param_scores = []

    for params in ParameterGrid(param_grid):
        model = model_class(**params)
        s = cross_validation(model, folds, metrics, X, y)
        param_scores.append((params, s[deciding_metric][0]))  # [0] is the mean across folds

    calc_best = max if deciding_metric == "r2" or deciding_metric in metrics_c else min
    best_params, best_score = calc_best(param_scores, key=lambda x: x[1])

    if verbose:
        print(f"✅ Best params: {best_params}")
        print(f"🎯 Best {deciding_metric.upper()}: {best_score:.4f}")

    return best_params

def run_hp_all(model_class, param_grid, folds, metrics, X, y, deciding_metric, verbose=False):
    model_best_params = {}
    for target, _y in y.items():
        for method, _X in X.items():
            if verbose: print(f"\nFeatures: {method} | Target: {target}")
            model_best_params[(method, target)] = hyperparameter_tuning(model_class, param_grid, folds, metrics, _X, _y, deciding_metric, verbose)

        for text_method in d_types_methods["text"]:
            for image_method in d_types_methods["image"]:
                _X = combine_features([X[text_method], X[image_method]])
                method = f"{text_method} + {image_method}"

                if verbose: print(f"\nFeatures: {method} | Target: {target}")
                model_best_params[(method, target)] = hyperparameter_tuning(model_class, param_grid, folds, metrics, _X, _y, deciding_metric, verbose)

    return model_best_params