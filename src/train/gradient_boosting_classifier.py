from sklearn.ensemble import GradientBoostingClassifier
from train.base_learner import BaseLearner

class GradientBoostingLearner(BaseLearner):
    def init(self, kwargs):
        self.model = GradientBoostingClassifier(kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)