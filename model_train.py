import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Data preprocessing:
dataset = pd.read_csv('Dataset of Diabetes.csv')
dataset = dataset.drop(['ID', 'No_Pation'], axis=1)
dataset['Gender'] = dataset['Gender'].map(lambda g: g.upper())
dataset['CLASS']  = dataset['CLASS'].map(lambda c: c.strip())

# Splitting our dataset into features (X) and labels (Y):
X = dataset.drop(['CLASS'], axis=1)
Y = dataset['CLASS']

# Encoding categorical features:
X['Gender'] = pd.Categorical(dataset['Gender']).codes

# One-Hot encode output labels:
Y = pd.get_dummies(Y)

# Splitting the dataset into training and test sets:
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1, stratify=Y)

