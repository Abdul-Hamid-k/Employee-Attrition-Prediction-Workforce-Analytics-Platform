from pathlib import Path
import joblib
import sys

sys.path.append('..')

from src.config import ROOT

import pandas as pd

"""
  Return 1 for Attrition and 0 for not
"""

def predict_attrition(test_data):
  model = joblib.load(f"{ROOT}/ml/models/attrition_classification_model.pkl")


  cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

  for col in cat_cols:
    label_encoder = joblib.load(f"{ROOT}/ml/models/{col}_encoder.pkl")
    test_data[col] = label_encoder.transform(test_data[col])

  pred = model.predict(test_data)

  return pred

import pandas as pd

columns = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'EducationField', 'EmployeeCount', 'EmployeeNumber',   'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
    'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany'
]

values = [
    25, 'Travel_Rarely', 480, 'Sales', 23, 'Other', 1000, 260, 2, 50, 2, 1, 'Sales Executive', 3, 'Single', 6000, 18000, 2, 'Y', 'Yes', 5, 2, 1, 80, 0, 4, 2, 3, 2
]

# Create a single-row DataFrame
df = pd.DataFrame([values], columns=columns)

# print(df)
print(predict_attrition(df))
