---
Start: 2024-03-23 13:14
DeadLine: 2024-05-17
Status:
  - Active
Priority: 
tags:
  - PatternRecognition
Arquive: false
End: 
Author: João E. Moreira
---


# 📔 Default of Credit Card Clients Prediction
## Overview
This project aims to predict the likelihood of credit card clients defaulting on their payments using machine learning techniques. The dataset used for this project is sourced from Kaggle and is titled "Default of Credit Card Clients Dataset".

## Problem Statement
The problem at hand is a binary classification task, where the goal is to predict whether a credit card client will default on their payments (class 1) or not (class 0). This prediction is based on various demographic and credit-related features provided in the dataset.

## Dataset
The dataset contains various features including:

- Limit Balance
- Gender
- Education
- Marital Status
- Age
- Payment history for the past six months (Sep 2005 - Apr 2005)
- Bill statement amounts for the past six months
- Previous payment amounts for the past six months

The target variable is "default.payment.next.month", which indicates whether the client defaulted on their credit card payment in the following month.

## Approach
1. Data Exploration: Conduct exploratory data analysis (EDA) to understand the distribution of features, identify patterns, and visualize relationships between variables.
2. Data Preprocessing: Perform data cleaning, handle missing values, encode categorical variables, and scale numerical features.
3. Feature Selection and Reduction: Employ various techniques such as univariate feature selection, principal component analysis (PCA), and correlation analysis to select and reduce the dimensionality of features.
4. Model Building: Experiment with different classification algorithms including Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), k-Nearest Neighbors (k-NN) and Naive Bayes.
5. Model Evaluation: Evaluate the performance of each model using appropriate evaluation metrics such as accuracy, precision, recall and area under the ROC curve.

## Requirements
- Python 3.x
- Jupyter Notebook (for running the code)
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Repository Structure
- **data**: Contains the dataset file(s).
- **src**: 
	- **exp**, **new_exp**: experiments data.
	- **models**: MDC Mahalanobis model and generic train model
	- **resources**: images used in the report
	- Jupyter notebooks, train scripts and other support notebooks.
- **docs**: PDF Documents and Report
- **README.md**: Overview of the project and instructions for running the code.
- **requirements.txt**: List of required Python libraries with versions.

## Credits
**Dataset Source**: Default of Credit Card Clients Dataset on Kaggle.




# 📝 Next Steps

- [x] set up project ✅ 2024-03-23
	- [x] create repository ✅ 2024-03-23
	- [x] setup the folders ✅ 2024-03-23
	- [x] create env for packages ✅ 2024-03-23
		- [x] numpy ✅ 2024-03-23
		- [x] matplotlib ✅ 2024-03-23
		- [x] pandas ✅ 2024-03-23
	- [x] generate a requirements file ✅ 2024-03-23
- [x] read and explore dataset ✅ 2024-03-23
	- [x] convert *.xls to *.csv ✅ 2024-03-23
	- [x] what are the features ✅ 2024-03-23
	- [x] describe the features ✅ 2024-03-23
	- [x] size of dataset ✅ 2024-03-23
	- [x] filtering missing values ✅ 2024-03-23
	- [x] correlation matrix ✅ 2024-03-23
	- [x] boxplot ✅ 2024-04-02
- [x] dataset processing ✅ 2024-03-29
	- [x] Scaling data ✅ 2024-03-23

Delivery:
- [x] Data Processing ✅ 2024-03-29
	- [x] Scaling ✅ 2024-03-26
	- [x] K-S test ✅ 2024-03-29
	- [x] K-W test ✅ 2024-03-29
	- [x] PCA ✅ 2024-03-26
	- [x] LDA ✅ 2024-03-26
	- [x] Feature Selection ✅ 2024-05-10
- [x] Classifiers ✅ 2024-03-29
	- [x] MDC ✅ 2024-03-26
	- [x] Fisher LDA ✅ 2024-03-26
- [x] Report ✅ 2024-04-02
- [x] Preciso de ver o que o Kruskal-Wallis test permite fazer ✅ 2024-03-29


Report:
- [x] introdução ✅ 2024-04-14
- [x] dados ✅ 2024-05-10
	- [x] EDA ✅ 2024-04-14
		- [x] Distribuição dos dados ✅ 2024-04-14
		- [x] Distribuiçao das features ✅ 2024-04-14
		- [x] Correlaçao das features ✅ 2024-04-14
	- [x] Normalização dos dados ✅ 2024-05-10
	- [x] Divisão treino e teste ✅ 2024-05-10
- [x] Seleção e reduçao de features ✅ 2024-05-10
	- [x] KS test ✅ 2024-04-14
	- [x] kw test ✅ 2024-04-14
	- [x] PCA ✅ 2024-04-14
	- [x] LDA ✅ 2024-04-14
- [x] Classificaçao dos dados ✅ 2024-05-10
	- [x] MDC ✅ 2024-04-14
	- [x] fisher LDA ✅ 2024-04-15
	- [x] Naive bayes classifier ✅ 2024-04-15
	- [x] knn classifier ✅ 2024-04-15
	- [x] SVM ✅ 2024-04-15
	- [x] random florest ✅ 2024-05-10
- [x] resultados ✅ 2024-05-11

# 📋 Notes
- [x] Meta 1 📅 2024-04-2 ⏫ ✅ 2024-04-02
- [x] Meta 2 ⏫ 📅 2024-05-17 ✅ 2024-05-11

You can find the RP Theory in [[Pattern Recognition]].
PS: if you are reading this outside of the author's obsidian vault you will not be able to access the theory notes.
