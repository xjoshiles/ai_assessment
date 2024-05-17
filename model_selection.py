import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Data preprocessing:
dataset = pd.read_csv('Dataset of Diabetes.csv')
dataset = dataset.drop(['ID', 'No_Pation'], axis=1)
dataset['Gender'] = dataset['Gender'].map(lambda g: g.upper())
dataset['CLASS']  = dataset['CLASS'].map(lambda c: c.strip())
dataset.drop_duplicates(inplace=True)

# Splitting our dataset into features (X) and labels (Y):
X = dataset.drop(['CLASS'], axis=1)
Y = dataset['CLASS']

# Integer encoding categorical features:
X['Gender'] = pd.Categorical(dataset['Gender']).codes

# One-Hot encode output labels:
Y = pd.get_dummies(Y)

# Splitting the dataset into training and test sets:
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1, stratify=Y)

# Set up a pipeline to be passed to GridSearchCV:
pipe = Pipeline(steps=[
    ("scaler", MinMaxScaler()),
    ("model",  MLPClassifier(max_iter=8000, random_state=2))
])

# Set up a hyperparameter grid to be passed to GridSearchCV:
param_grid = {'model__hidden_layer_sizes': [(3, 3), (4, 4), (5, 5), (6, 6)],
              'model__activation': ['relu', 'logistic'],
              'model__solver': ['lbfgs', 'adam'],
              'model__alpha': [0.01, 0.001, 0.0001],
              'model__batch_size': [10, 25, 50, 100]
              }

# Set up the grid search:
search = GridSearchCV(estimator=pipe,
                      param_grid=param_grid,
                      cv=5,       # 5-fold cross-validation
                      scoring='accuracy',
                      n_jobs=-1,  # use all processors
                      refit=True)

# Fit and validate all models in the grid search using the training set:
search.fit(X_train, Y_train)

# Get best-found estimator (model) and its scaler:
scaler = search.best_estimator_.named_steps['scaler']
model  = search.best_estimator_.named_steps['model']

# Print the best found parameters:
print("The best-found parameters are:")
print(model)
