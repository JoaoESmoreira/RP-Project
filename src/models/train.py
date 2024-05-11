
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix

class Train:
    def __init__(self, df_X, df_y, model, generations=30, **model_params) -> None:
        self.df_X = df_X
        self.df_y = df_y
        self.model = model
        self.generations = generations
        self.model_params = model_params
        self.model_fited = None

        self.error = {
            'accuracy_score': [],
            'sensitivity_score': [],
            'specificity_score': []
        }

    def fit(self):
        for i in range(self.generations):
            print(i)
            X_train, X_test, y_train, y_test = train_test_split(self.df_X, self.df_y, test_size=0.2, random_state=i, stratify=self.df_y)

            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

            scaler = StandardScaler()
            X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

            X_train.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)

            model = self.model(**self.model_params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
            self.error["accuracy_score"].append((tp + tn) / (tp + tn + fp + fn))
            self.error["sensitivity_score"].append(tp / (tp + fn))
            self.error["specificity_score"].append(tn / (tn + fp))
        self.error = pd.DataFrame(self.error, columns=self.error.keys())
    
class TrainSelectedFeatures(Train):
    def __init__(self, df_X, df_y, model, generations=30, **model_params) -> None:
        super().__init__(df_X, df_y, model, generations, **model_params)

    def fit(self):
        for i in range(self.generations):
            X_train, X_test, y_train, y_test = train_test_split(self.df_X, self.df_y, test_size=0.2, random_state=i, stratify=self.df_y)

            model = self.model(**self.model_params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
            self.error["accuracy_score"].append((tp + tn) / (tp + tn + fp + fn))
            self.error["sensitivity_score"].append(tp / (tp + fn))
            self.error["specificity_score"].append(tn / (tn + fp))
        self.error = pd.DataFrame(self.error, columns=self.error.keys())
