import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score


def score(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
) -> float:
    if y_true.ndim != 1 or y_pred.ndim > 1:
        raise ValueError("y_true and y_pred should be 1d array")

    return float(roc_auc_score(y_true=y_true.astype(np.int16), y_score=y_pred))


if __name__ == "__main__":
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    print(score(y_true, y_pred))
