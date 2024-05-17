import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


def calculate_accuracy(confusion_matrix):
    """
    Calculate accuracy using a confusion matrix.
    
    Parameters
    ----------
        confusion_matrix: numpy.ndarray of shape (n_classes, n_classes).
    
    Returns
    -------
        float: Accuracy of the classification model.
    """
    true_predictions = 0

    # Calculate true_predictions by summing the main diagonal:
    for i in range(len(confusion_matrix)):
        true_predictions += confusion_matrix[i,i]

    # Calculate and return the accuracy by dividing the
    # true_predictions by the total number of predictions:
    return true_predictions / sum(sum(confusion_matrix))



def calculate_precision(confusion_matrix):
    """
    Calculate precision of each class using a confusion matrix.
    
    Parameters
    ----------
        confusion_matrix: numpy.ndarray of shape (n_classes, n_classes).
    
    Returns
    -------
        list[float]: precision values for each class.
    """
    precision_values = []

    # Calculate precision of each class:
    for i in range(len(confusion_matrix)):
        
        # Get the true positives for this class:
        tp = confusion_matrix[i, i]
        
        # Sum all positive predictions (tp + fp) for this class:
        pp = 0
        for row in confusion_matrix:
            pp += row[i]

        # Append the precision value of this class:
        if pp == 0:
            precision_values.append(0)
        else:
            precision_values.append(tp / pp)

    return precision_values


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

# Scale the data:
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Define our model (params found from model_selection.py):
mlp = MLPClassifier(hidden_layer_sizes=(5, 5),
                    activation='logistic',
                    solver='lbfgs',
                    alpha=0.01,
                    max_iter=8000,
                    random_state=2)

# Fit (train) the model with the training set:
mlp.fit(X_train, Y_train)

# Make our predictions on the test set:
Y_pred = mlp.predict(X_test)

# Create and display a confusion matrix:
cm = confusion_matrix(Y_test.values.argmax(axis=1), Y_pred.argmax(axis=1))
ConfusionMatrixDisplay(cm, display_labels=Y_test.columns).plot()

# Manually calculate accuracy and precision scores using confusion matrix:
print(f'Accuracy of model on test set:\n{calculate_accuracy(cm)}\n')
print(f'Precision of model on test set:\n{calculate_precision(cm)}')
print(f'labels: {list(Y_test.columns)}\n')

# Print scikit-learn's classification report for more detailed metrics:
print('Scikit-learn classification report:')
report = classification_report(Y_test.values.argmax(axis=1),
                               Y_pred.argmax(axis=1),
                               target_names=Y_test.columns,
                               digits=3,
                               zero_division=0)
print(report)
