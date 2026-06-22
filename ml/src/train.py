import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
# from catboost import CatBoostClassifier

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform, randint

import joblib


from pathlib import Path
import sys

sys.path.append('..')

from src.preprocessing import load_processed_data

def attrition_classification_model():
  
# get preprocessed data
  X_train, X_test, X_val, y_train, y_test, y_val = load_processed_data()


  # Model
  gbc = GradientBoostingClassifier(random_state=42)
  gbc.fit(X_train, y_train)

  y_train_pred = gbc.predict(X_train)
  y_test_pred = gbc.predict(X_test)

  y_train_accuracy = accuracy_score(y_train, y_train_pred)
  y_test_accuracy = accuracy_score(y_test, y_test_pred)

  print(f"Training Accuracy: {y_train_accuracy}")
  print(f"Testing Accuracy: {y_test_accuracy}")

  # X_train.columns


  joblib.dump(gbc, open('../models/attrition_classification_model.pkl', 'wb'))


  return gbc

attrition_classification_model()