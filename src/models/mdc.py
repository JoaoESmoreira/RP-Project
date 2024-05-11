
import numpy as np

class MinimumDistanceClassifierMahalanobis:
    def __init__(self):
        self.centroids = {}
        self.covariances = {}

    def fit(self, X_train, y_train):
        self.centroids = {c: X_train[y_train == c].mean() for c in np.unique(y_train)}
        self.covariances = {c: X_train[y_train == c].cov() for c in np.unique(y_train)}

    def predict(self, X_test):
        predictions = [
            min(
                self.centroids.keys(),
                key=lambda c: np.sqrt((x - self.centroids[c]) @ np.linalg.inv(self.covariances[c]) @ (x - self.centroids[c]))
            )
            for _, x in X_test.iterrows()
        ]
        return predictions
    
    def get_key(self):
        return self.centroids.keys()
import numpy as np

class NearestCentroidMahalanobis:
    def __init__(self):
        self.centroids = {}
        self.covariances = {}

    def fit(self, X_train, y_train):
        classes = np.unique(y_train)
        self.centroids = {c: X_train[y_train == c].mean() for c in classes}
        self.covariances = {c: X_train[y_train == c].cov() for c in classes}

    def predict(self, X_test):
        predictions = []
        for _, x in X_test.iterrows():
            dists = {}
            for c in self.centroids.keys():
                centroid = self.centroids[c]
                covariances = self.covariances[c]
                dists[c] = np.sqrt((x - centroid) @ covariances @ (x - centroid))
            predictions.append(min(dists, key=dists.get))
        return predictions
    