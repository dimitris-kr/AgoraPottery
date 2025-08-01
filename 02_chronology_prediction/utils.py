import json

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from itertools import product

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, max_error, \
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
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

    targets = [target for target in targets if target in y["train"]]
    if len(targets) == 0:
        return None

    y = {subset: _y[targets] for subset, _y in y.items()}
    return y


# PRINT INFO

def print_info_features(X):
    print("X = {")
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
    print("y = {")
    for subset in y.keys():
        indent = "\t"
        print(f"{indent}{subset}: ")
        indent = 2 * "\t"
        print(f"{indent}{type(y[subset])}")
        print(f"{indent}shape {y[subset].shape}")
        print(f"{indent}columns {list(y[subset].columns)},")
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

            min_y = 0
            max_y = 0

            for metric, color, offset in zip(subplot['metrics'], subplot['colors'], offsets):
                bars = plt.bar(x + offset * bar_width, df[metric], width=bar_width, label=metric.upper(), color=color)
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}", ha='center', va='bottom',
                        fontsize=9, rotation=90)
                max_y = max(max_y, df[metric].max())
                min_y = min(min_y, df[metric].min())

            plt.axhline(0, color='gray', alpha=0.5, linestyle='--')
            plt.xticks(ticks=x, labels=df["features"], rotation=45, ha='right')
            plt.ylabel(subplot['ylabel'])
            plt.ylim(min_y * 1.1, subplot['ylim'] if 'ylim' in subplot else max_y * 1.2)
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

    return {"model": model_name, "target": target, "features": method,
            **{metric: scores[metric][0] for metric in metrics}}


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

                scores = run_cv(model_name, model_class, best_params, folds, metrics, _X, _y, method, target,
                                enable_plots)
                model_scoreboard.append(scores)

    return pd.DataFrame(model_scoreboard)


def run_cv_all_2(model_name, model_class, best_params, folds, metrics, X, y, enable_plots=True):
    if enable_plots: print("Cross Validation Score Progression")
    model_scoreboard = []
    for target, _y in y.items():
        for method, _X in X.items():
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
            model_best_params[(method, target)] = hyperparameter_tuning(model_class, param_grid, folds, metrics, _X, _y,
                                                                        deciding_metric, verbose)

        # for text_method in d_types_methods["text"]:
        #     for image_method in d_types_methods["image"]:
        #         _X = combine_features([X[text_method], X[image_method]])
        #         method = f"{text_method} + {image_method}"
        #
        #         if verbose: print(f"\nFeatures: {method} | Target: {target}")
        #         model_best_params[(method, target)] = hyperparameter_tuning(model_class, param_grid, folds, metrics, _X, _y, deciding_metric, verbose)

    return model_best_params


def combine_features_all_txt_img(X, scale=False):
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


# EVALUATE SCOREBOARD

def get_unique(df, columns):
    return (df[column].unique().tolist() for column in columns)


def get_color_map(values, palette_name):
    palette = sns.color_palette(palette_name, len(values))
    return dict(zip(values, palette))


def plot_metric_in_groups(df, metric, x_axis, group_by, palette, ylim=None):
    plt.figure(figsize=(16, 8))
    sns.barplot(
        data=df,
        x=x_axis,
        y=metric,
        hue=group_by,
        palette=palette
    )

    if ylim: plt.ylim(0, ylim)

    plt.xticks(rotation=45)
    plt.title(
        f"{metric.upper()} per {x_axis.upper()} | Grouped by {group_by.upper()} | Target = {df['target'].values[0]}")
    plt.ylabel(metric.upper())
    plt.xlabel(x_axis.upper())
    plt.legend(title=group_by.upper())
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
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="mako", linewidths=0.5)
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


def plot_best_of_each(df, metric, x_axis, find_best, palette, ylim=None):
    best_of_each = df.loc[df.groupby(x_axis)[metric].idxmax()]
    best_of_each = best_of_each.sort_values(metric, ascending=metric in metrics_r and metric != "r2")

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=best_of_each,
        x=x_axis,
        y=metric,
        hue=find_best,
        palette=palette
    )

    # Add model name as a label on top of each bar
    for container, label in zip(ax.containers, best_of_each[find_best].unique()):
        ax.bar_label(container, labels=[label] * len(container), fontsize=9, label_type='edge')

    plt.title(f"Best {find_best.upper()} per {x_axis.upper()} | Target = {df['target'].values[0]}")
    plt.ylabel(metric.upper())
    plt.xlabel(x_axis.upper())
    plt.xticks(rotation=45)
    if ylim: plt.ylim(0, ylim)
    plt.legend(title=find_best.upper())
    plt.tight_layout()
    plt.show()


def plot_best_by_data_type(df, metric, palette, lim=None, top_n=5):
    d_types_methods_full = d_types_methods.copy()
    d_types_methods_full["combo"] = tuple(f"{t} + {i}" for t in d_types_methods["text"] for i in d_types_methods["image"])
    for d_type, features in d_types_methods_full.items():
        subset = df.loc[df["features"].isin(features)]

        if subset.empty:
            continue

        subset = subset.sort_values(metric, ascending=metric in metrics_r and metric != "r2").head(top_n)
        subset["combo"] = subset["model"] + " w/ " + subset["features"]

        plt.figure(figsize=(10, 6))
        sns.barplot(data=subset, x=metric, y="combo", hue="model", palette=palette)
        plt.title(f"Top {top_n} MODEL & FEATURE pair | {d_type.upper()} Data | Target = {df['target'].values[0]}")
        plt.xlabel(metric.upper())
        plt.ylabel("MODEL & FEATURE pair")
        if lim: plt.xlim(0, lim)
        plt.grid(True, axis="x", linestyle="--", alpha=0.6)
        for index, value in enumerate(subset[metric]):
            plt.text(value + 0.01, index, f"{value:.2f}", va='center', fontsize=9)
        plt.tight_layout()
        plt.show()