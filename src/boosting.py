
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(X, y):
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, stratify=y)
    
    pca = PCA(n_components=X_train.shape[1])
    pca.fit(X_train, y_train)
    X_train_pca = X_train.dot(pca.components_[range(8)].T)
    X_test_pca = X_test.dot(pca.components_[range(8)].T)
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    X_train_lda = pd.DataFrame(lda.transform(X_train))
    X_test_lda = pd.DataFrame(lda.transform(X_test))
    
    return X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, X_train_lda, X_test_lda

def train_models(X_train, y_train):
    mdc = NearestCentroid()
    mdc.fit(X_train, y_train)

    mdc_pca = NearestCentroid()
    mdc_pca.fit(X_train_pca, y_train)

    fisher = NearestCentroid()
    fisher.fit(X_train_lda, y_train)
    
    return mdc, mdc_pca, fisher

def evaluate_predictions(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return {'accuracy_score': accuracy, 'sensitivity_score': sensitivity, 'specificity_score': specificity}

INPUT_PATH = '../data/credit_card_clients.csv'
data = pd.read_csv(INPUT_PATH)
data.rename(columns={'default payment next month': 'target'}, inplace=True)
data.drop(["ID"], axis=1, inplace=True)

y = data['target'].copy()
X = data.copy()
X.drop(['target'], axis=1, inplace=True)

X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, X_train_lda, X_test_lda = preprocess_data(X, y)
mdc, mdc_pca, fisher = train_models(X_train, y_train)

predictions = pd.DataFrame({0: np.zeros(len(X_test)), 1: np.zeros(len(X_test))})

for model, X_test_data in [(mdc, X_test), (mdc_pca, X_test_pca), (fisher, X_test_lda)]:
    predict = model.predict(X_test_data)
    predictions.loc[predict == 0, 0] += 1
    predictions.loc[predict == 1, 1] += 1

final_predictions = pd.DataFrame(np.zeros(len(X_test)))
final_predictions[predictions[0] < predictions[1]] = 1
final_predictions = final_predictions.astype(int)

evaluation_results = evaluate_predictions(y_test, final_predictions)
print(evaluation_results)
