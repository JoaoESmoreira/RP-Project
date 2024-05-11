

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from models.mdc import NearestCentroidMahalanobis
from models.train import Train, TrainSelectedFeatures

#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from IPython.display import display
#
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#
#from scipy.stats import kstest, kruskal
#
#from sklearn.decomposition import PCA
#from sklearn.metrics import roc_curve, auc, confusion_matrix
#
#from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import LogisticRegression


INPUT_PATH = '../data/credit_card_clients.csv'
data = pd.read_csv(INPUT_PATH)
data.rename(columns={'default payment next month': 'target'}, inplace=True)
data.drop(["ID"], axis=1, inplace=True)

y = data['target'].copy()
X = data.copy()
X.drop(['target'], axis=1, inplace=True)


def main1():
    """
    Experiencia para todos os classificadores com os dados cruz sem qualquer estimação de parametros
    """
    models = [NearestCentroid, NearestCentroidMahalanobis, GaussianNB, KNeighborsClassifier, SVC, RandomForestClassifier]
    csvs = ['NearestCentroid', 'NearestCentroidMahalanobis', 'GaussianNB', 'KNeighborsClassifier', 'SVM', 'RandomForestClassifier']
    generations = [30, 30, 30, 30, 30, 30]

    for i in range(len(models)):
        print(csvs[i])
        trainer = Train(X, y, models[i], generations[i])
        trainer.fit()
        trainer.error.to_csv(f"./exp/{csvs[i]}.csv")

def main2():
    """
    Experiencia para MDC_euclidiana e Naive Bayses com a LDA
    """
    models = [NearestCentroid, GaussianNB]
    csvs = ['LDA_MDC_euc', 'LDA_Naive_Bayses']
    generations = [30, 30]

    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X, y)
    X_lda = pd.DataFrame(LDA.transform(X))

    for i in range(len(models)):
        print(csvs[i])
        trainer = Train(X_lda, y, models[i], generations[i])
        trainer.fit()
        trainer.error.to_csv(f"./exp/{csvs[i]}.csv")

def main3():
    """
    Experiencia para estimar o melhor K para o KNN com PCA
    """
    pca = PCA(n_components=8)
    pca.fit(X)
    X_pca = pd.DataFrame(pca.transform(X), columns=[str(i) for i in range(8)])

    error = {
        'k': [],
        'accuracy_score': [],
        'sensitivity_score': [],
        'specificity_score': [],
    }
    
    for i in range(1,30):
        knn_params = { 'n_neighbors': i }
        trainer = TrainSelectedFeatures(X_pca, y, KNeighborsClassifier, 10, **knn_params)
        trainer.fit()
        error['k'].append(i)
        error['accuracy_score'].append(trainer.error['accuracy_score'].mean())
        error['sensitivity_score'].append(trainer.error['sensitivity_score'].mean())
        error['specificity_score'].append(trainer.error['specificity_score'].mean())
    error = pd.DataFrame(error)
    error.to_csv(f"./exp/PCA_bestK.csv")


def main4():
    """
    Experiencia para estimar o melhor C e gamma para a SVM com PCA
    """
    C_values = [2**i for i in range(-5, 12)]
    gamma_values = [2**i for i in range(-30, 5)]

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    pca = PCA(n_components=8)
    pca.fit(data)
    X_pca = pd.DataFrame(pca.transform(X), columns=[str(i) for i in range(8)])

    acc = np.zeros([len(C_values), len(gamma_values)])
    spp = np.zeros([len(C_values), len(gamma_values)])
    sen = np.zeros([len(C_values), len(gamma_values)])

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.5)
    for i in range(len(C_values)):
        C = C_values[i]
        for j in range(len(gamma_values)):
            G = gamma_values[j]
            clf = SVC(C=C, gamma=G)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            print(i, j)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            acc[i][j] = (tp + tn) / (tp + tn + fp + fn)
            sen[i][j] = tp / (tp + fn)
            spp[i][j] = tn / (tn + fp)
        
    np.savetxt('./exp/PCA_svm_acc0.txt', acc)
    np.savetxt('./exp/PCA_svm_spp0.txt', spp)
    np.savetxt('./exp/PCA_svm_sen0.txt', sen)

def main5():
    """
    Experiencia para estimar o melhor C e gamma para a SVM com AUC
    """
    C_values = [2**i for i in range(-5, 5)]
    gamma_values = [2**i for i in range(-15, 5)]

    acc = np.zeros([len(C_values), len(gamma_values)])
    spp = np.zeros([len(C_values), len(gamma_values)])
    sen = np.zeros([len(C_values), len(gamma_values)])

    cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size=0.5)
    for i in range(len(C_values)):
        C = C_values[i]
        for j in range(len(gamma_values)):
            G = gamma_values[j]
            clf = SVC(C=C, gamma=G)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            print(i, j)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            acc[i][j] = (tp + tn) / (tp + tn + fp + fn)
            sen[i][j] = tp / (tp + fn)
            spp[i][j] = tn / (tn + fp)
        
    np.savetxt('./exp/AUC_svm_acc.txt', acc)
    np.savetxt('./exp/AUC_svm_spp.txt', spp)
    np.savetxt('./exp/AUC_svm_sen.txt', sen)

def main6():
    """
    Experiencia para estimar o melhor C e gamma para a SVM com KW
    """
    C_values = [2**i for i in range(-5, 12)]
    gamma_values = [2**i for i in range(-30, 5)]

    acc = np.zeros([len(C_values), len(gamma_values)])
    spp = np.zeros([len(C_values), len(gamma_values)])
    sen = np.zeros([len(C_values), len(gamma_values)])

    cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

    X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size=0.5)
    for i in range(len(C_values)):
        C = C_values[i]
        for j in range(len(gamma_values)):
            G = gamma_values[j]
            clf = SVC(C=C, gamma=G)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            print(i, j)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            acc[i][j] = (tp + tn) / (tp + tn + fp + fn)
            sen[i][j] = tp / (tp + fn)
            spp[i][j] = tn / (tn + fp)
        
    np.savetxt('./exp/KW_svm_acc0.txt', acc)
    np.savetxt('./exp/KW_svm_spp0.txt', spp)
    np.savetxt('./exp/KW_svm_sen0.txt', sen)


def main7():
    """
    Experiencia para todos os classificadores com PCA
    """
    models = [NearestCentroid, NearestCentroidMahalanobis, GaussianNB, KNeighborsClassifier, RandomForestClassifier]
    csvs = ['NearestCentroid', 'NearestCentroidMahalanobis', 'GaussianNB', 'KNeighborsClassifier', 'RandomForestClassifier']
    params = [{}, {}, {},  { 'n_neighbors': 1}, {}]
    generations = [30, 30, 30, 30, 30]

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    pca = PCA(n_components=8)
    pca.fit(data)
    X_pca = pd.DataFrame(pca.transform(X), columns=[str(i) for i in range(8)])

    for i in range(len(models)):
        print(csvs[i])
        trainer = TrainSelectedFeatures(X_pca, y, models[i], generations[i], **params[i])
        trainer.fit()
        trainer.error.to_csv(f"./new_exp/Classification_PCA_{csvs[i]}.csv")


def main8():
    """
    Experiencia para estimar o melhor K para o KNN com AUC
    """
    error = {
        'k': [],
        'accuracy_score': [],
        'sensitivity_score': [],
        'specificity_score': [],
    }
    
    cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    for i in range(1,40):
        knn_params = { 'n_neighbors': i }
        trainer = Train(X[cols], y, KNeighborsClassifier, 10, **knn_params)
        trainer.fit()
        error['k'].append(i)
        error['accuracy_score'].append(trainer.error['accuracy_score'].mean())
        error['sensitivity_score'].append(trainer.error['sensitivity_score'].mean())
        error['specificity_score'].append(trainer.error['specificity_score'].mean())
    error = pd.DataFrame(error)
    error.to_csv(f"./exp/AUC_bestK.csv")

def main9():
    """
    Experiencia para estimar o melhor K para o KNN com KW
    """
    error = {
        'k': [],
        'accuracy_score': [],
        'sensitivity_score': [],
        'specificity_score': [],
    }
    cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    for i in range(1,40):
        knn_params = { 'n_neighbors': i }
        trainer = Train(X[cols], y, KNeighborsClassifier, 10, **knn_params)
        trainer.fit()
        error['k'].append(i)
        error['accuracy_score'].append(trainer.error['accuracy_score'].mean())
        error['sensitivity_score'].append(trainer.error['sensitivity_score'].mean())
        error['specificity_score'].append(trainer.error['specificity_score'].mean())
        print(i)
    error = pd.DataFrame(error)
    error.to_csv(f"./exp/KW_bestK.csv")


def main10():
    """
    Experiencia para todos os classificadores com AUC
    """
    models = [NearestCentroid, NearestCentroidMahalanobis, GaussianNB, KNeighborsClassifier, RandomForestClassifier, SVC]
    csvs = ['NearestCentroid', 'NearestCentroidMahalanobis', 'GaussianNB', 'KNeighborsClassifier', 'RandomForestClassifier', 'SVM']
    params = [{}, {}, {},  { 'n_neighbors': 20}, {}, { 'C': 2, 'gamma':0.25}]
    generations = [30, 30, 30, 30, 30, 30]
    cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    for i in range(len(models)):
        print(csvs[i])
        trainer = Train(X[cols], y, models[i], generations[i], **params[i])
        trainer.fit()
        trainer.error.to_csv(f"./new_exp/Classification_AUC_{csvs[i]}.csv")

def main11():
    """
    Experiencia para todos os classificadores com KW
    """
    models = [NearestCentroid, NearestCentroidMahalanobis, GaussianNB, KNeighborsClassifier, RandomForestClassifier, SVC]
    csvs = ['NearestCentroid', 'NearestCentroidMahalanobis', 'GaussianNB', 'KNeighborsClassifier', 'RandomForestClassifier', 'SVM']
    params = [{}, {}, {},  { 'n_neighbors': 1}, {}, { 'C': 2048, 'gamma':2.9802322387695312e-08}]
    generations = [30, 30, 30, 30, 30, 30, 30, 30]

    cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

    for i in range(len(models)):
        print(csvs[i])
        trainer = Train(X[cols], y, models[i], generations[i], **params[i])
        trainer.fit()
        trainer.error.to_csv(f"./new_exp/Classification_KW_{csvs[i]}.csv")




def my_classification():
    """
    MDC euc raw
    Fisher LDA
    MDC euc PCA
    """
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, stratify=y)

    pca = PCA(n_components=X_train.shape[1])
    pca.fit(X_train, y_train)
    X_train_pca = X_train.dot(pca.components_[range(8)].T)
    X_test_pca = X_test.dot(pca.components_[range(8)].T)

    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, y_train)
    X_train_lda = pd.DataFrame(LDA.transform(X_train))
    X_test_lda = pd.DataFrame(LDA.transform(X_test))

    MDC = NearestCentroid()
    MDC.fit(X_train, y_train)

    MDC_PCA = NearestCentroid()
    MDC_PCA.fit(X_train_pca, y_train)

    FISHER = NearestCentroid()
    FISHER.fit(X_train_lda, y_train)

    predictions = pd.DataFrame({0:np.zeros(len(X_test)), 1:np.zeros(len(X_test))})

    predict = MDC.predict(X_test)
    predictions.loc[predict == 0, 0] += 1
    predictions.loc[predict == 1, 1] += 1

    predict = MDC_PCA.predict(X_test_pca)
    predictions.loc[predict == 0, 0] += 1
    predictions.loc[predict == 1, 1] += 1

    predict = FISHER.predict(X_test_lda)
    predictions.loc[predict == 0, 0] += 1
    predictions.loc[predict == 1, 1] += 1


    predict = pd.DataFrame( np.zeros(len(X_test)))
    predict[predictions[0] > predictions[1]] = 0
    predict[predictions[0] < predictions[1]] = 1
    predict = predict.astype(int)

    error = {
        'accuracy_score': [],
        'sensitivity_score': [],
        'specificity_score': []
    }
    tn, fp, fn, tp = confusion_matrix(y_test, predict).ravel()
    error["accuracy_score"].append((tp + tn) / (tp + tn + fp + fn))
    error["sensitivity_score"].append(tp / (tp + fn))
    error["specificity_score"].append(tn / (tn + fp))

    print(error)

if __name__ == "__main__":
    my_classification()
