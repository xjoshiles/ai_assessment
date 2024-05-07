import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data preprocessing
dataset = pd.read_csv('Dataset of Diabetes.csv')
dataset = dataset.drop(['ID', 'No_Pation'], axis=1)
dataset['Gender'] = dataset['Gender'].map(lambda g: g.upper())
dataset['CLASS']  = dataset['CLASS'].map(lambda c: c.strip())

