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
  pred = model.predict(test_data)

  return pred

import pandas as pd

columns = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'EducationField', 'EmployeeCount', 'EmployeeNumber',
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
    'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime',
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany'
]

values = [
    25, 1, 480, 1, 23, 1, 1, 260, 2, 50, 2, 1, 4, 3, 1,
    6000, 18000, 2, 1, 1, 5, 2, 1, 80, 0, 4, 2, 3, 2
]

# Create a single-row DataFrame
df = pd.DataFrame([values], columns=columns)

# print(df)
print(predict_attrition(df))
