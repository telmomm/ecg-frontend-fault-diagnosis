"""Reference tabular classifiers (experiment E3). Hyper-parameters are untuned defaults."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def classifiers(seed: int = 0) -> dict[str, Pipeline]:
    return {
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "random_forest": make_pipeline(
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
        ),
        "svm": make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", random_state=seed)),
        "gradient_boosting": make_pipeline(HistGradientBoostingClassifier(random_state=seed)),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed),
        ),
    }
