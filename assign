from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

wine = datasets.load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

baseline_svm = SVC()
baseline_svm.fit(X_train, y_train)
y_pred_baseline = baseline_svm.predict(X_test)

print("Baseline SVM Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Precision:", precision_score(y_test, y_pred_baseline, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_baseline, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_baseline, average='weighted'))

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.1, 1, 'scale', 'auto']
}

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)

# Evaluate the model
print("\nGridSearchCV Performance:")
print("Best Hyperparameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_grid))
print("Precision:", precision_score(y_test, y_pred_grid, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_grid, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_grid, average='weighted'))

param_dist = {
    'C': uniform(0.1, 10),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 20))
}

# Randomized search with 20 iterations and 5-fold cross-validation
random_search = RandomizedSearchCV(estimator=SVC(), param_distributions=param_dist, n_iter=20, cv=5)
random_search.fit(X_train, y_train)

# Best model from random search
best_rand_model = random_search.best_estimator_
y_pred_rand = best_rand_model.predict(X_test)

# Evaluate the model
print("\nRandomizedSearchCV Performance:")
print("Best Hyperparameters:", random_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rand))
print("Precision:", precision_score(y_test, y_pred_rand, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_rand, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_rand, average='weighted'))

results = {
    "Model": ["Baseline SVM", "GridSearchCV", "RandomizedSearchCV"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_baseline),
        accuracy_score(y_test, y_pred_grid),
        accuracy_score(y_test, y_pred_rand)
    ],
    "Precision": [
        precision_score(y_test, y_pred_baseline, average='weighted'),
        precision_score(y_test, y_pred_grid, average='weighted'),
        precision_score(y_test, y_pred_rand, average='weighted')
    ],
    "Recall": [
        recall_score(y_test, y_pred_baseline, average='weighted'),
        recall_score(y_test, y_pred_grid, average='weighted'),
        recall_score(y_test, y_pred_rand, average='weighted')
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_baseline, average='weighted'),
        f1_score(y_test, y_pred_grid, average='weighted'),
        f1_score(y_test, y_pred_rand, average='weighted')
    ]
}

df = pd.DataFrame(results)
print("\nPerformance Comparison:\n")
print(df.round(3))
