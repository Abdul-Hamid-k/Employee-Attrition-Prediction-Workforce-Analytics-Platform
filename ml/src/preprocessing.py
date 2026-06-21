import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pathlib import Path
import sys

sys.path.append('..')

from src.config import DATA_RAW



def load_processed_data():

  """
    Load data
    - no nulls
    - no duplicates
    split data
    encode data
    return encoded data
  """

  df = pd.read_csv(DATA_RAW / 'HR-Employee-Attrition.csv')

  X = df.drop(['Attrition', 'Education', 'Gender', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'], axis=1)
  y = df['Attrition']

  # Train & Test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Test & Validate
  X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

  # Label encoding

  # Initialize LabelEncoder
  le = LabelEncoder()

  # Apply Label Encoding to categorical features in X_train, X_test, X_val
  for col in X_train.select_dtypes(include='object').columns:
      X_train[col] = le.fit_transform(X_train[col])
      X_test[col] = le.transform(X_test[col])
      X_val[col] = le.transform(X_val[col])

  # Apply Label Encoding to the target variable y_train, y_test, y_val
  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)
  y_val = le.transform(y_val)

  print("Label Encoding applied to categorical features in X and target variable y.")

  return X_train, X_test, X_val, y_train, y_test, y_val

