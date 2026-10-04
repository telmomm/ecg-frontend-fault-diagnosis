"""Reference tabular classifiers with small hyper-parameter grids.

The grids are deliberately small: the study is about what the measurements allow,
not about squeezing a model. `tune` chooses within the grid on a validation split.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# models that stay fast on the full dataset (tens of thousands of rows)
FAST_MODELS = ("knn", "random_forest", "gradient_boosting")


def classifiers(seed: int = 0, balanced: bool = False) -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, hyper-parameter grid).

    `balanced` weights the classes inversely to their frequency in the models that
    support it (random forest, SVM, gradient boosting); k-NN and the MLP cannot.
    """
    weight = "balanced" if balanced else None
    return {
        "knn": (
            make_pipeline(StandardScaler(), KNeighborsClassifier()),
            {"kneighborsclassifier__n_neighbors": [3, 7, 15]},
        ),
        "random_forest": (
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=200, n_jobs=-1, class_weight=weight, random_state=seed
                )
            ),
            {"randomforestclassifier__max_features": ["sqrt", 0.5]},
        ),
        "svm": (
            make_pipeline(StandardScaler(), SVC(gamma="scale", class_weight=weight)),
            {"svc__C": [1.0, 10.0, 100.0]},
        ),
        "gradient_boosting": (
            make_pipeline(HistGradientBoostingClassifier(class_weight=weight, random_state=seed)),
            {"histgradientboostingclassifier__max_leaf_nodes": [15, 31]},
        ),
        "mlp": (
            make_pipeline(
                StandardScaler(),
                MLPClassifier(max_iter=300, early_stopping=True, random_state=seed),
            ),
            {"mlpclassifier__hidden_layer_sizes": [(64,), (128, 64)]},
        ),
    }


def tune(
    model: Pipeline,
    grid: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Parameters of `grid` with the best macro F1 on the validation split."""
    best_score, best_params = -1.0, {}
    for params in ParameterGrid(grid):
        candidate = clone(model).set_params(**params).fit(x_train, y_train)
        score = f1_score(y_val, candidate.predict(x_val), average="macro", zero_division=0)
        if score > best_score:
            best_score, best_params = score, params
    return best_params
