import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

