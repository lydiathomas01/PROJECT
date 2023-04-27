from sklearn.utils import check_random_state
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class RotationForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_depth=None, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = sorted(set(y))

        # Initialize the base estimator
        self.base_estimator_ = DecisionTreeClassifier(max_depth=self.max_depth,
                                                       max_features=self.max_features,
                                                       random_state=self.random_state)

        # Generate random rotations
        rng = check_random_state(self.random_state)
        self.rotations_ = [rng.normal(size=(X.shape[1], X.shape[1])) for _ in range(self.n_estimators)]

        # Fit the rotated datasets
        self.estimators_ = []
        for i in range(self.n_estimators):
            X_rot = X @ self.rotations_[i]
            estimator = RandomForestClassifier(max_depth=self.max_depth,
                                                max_features=self.max_features,
                                                random_state=self.random_state)
            estimator.fit(X_rot, y)
            self.estimators_.append(estimator)

        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Predict class for X
        y_pred = []
        for i in range(self.n_estimators):
            X_rot = X @ self.rotations_[i]
            y_pred.append(self.estimators_[i].predict(X_rot))

        y_pred = np.array(y_pred).T
        y_pred = [np.bincount(y).argmax() for y in y_pred]

        return np.array(y_pred)
