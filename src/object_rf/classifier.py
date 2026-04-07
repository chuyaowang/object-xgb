from sklearn.ensemble import RandomForestClassifier


class ObjectClassifier:
    def __init__(self, n_estimators=100, random_state=42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )

    def train(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
