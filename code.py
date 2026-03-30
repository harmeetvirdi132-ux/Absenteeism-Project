# Absenteeism Data Preprocessing Project
# Data Analyst Course Project

import pandas as pd
import numpy as np

# =========================
# Load Dataset
# =========================
df = pd.read_csv('Absenteeism-data.csv')

pd.options.display.max_columns = None
pd.options.display.max_rows = None

print(df.head())
print(df.info())

# =========================
# Drop ID Column
# =========================
df = df.drop(['ID'], axis=1)

# =========================
# Analyze Reason for Absence
# =========================
print(df['Reason for Absence'].min())
print(df['Reason for Absence'].max())
print(df['Reason for Absence'].unique())
print(len(df['Reason for Absence'].unique()))
print(sorted(df['Reason for Absence'].unique()))

# =========================
# Create Dummy Variables
# =========================
reason_columns = pd.get_dummies(df['Reason for Absence'])

# Check dummy variables
reason_columns['check'] = reason_columns.sum(axis=1)
print(reason_columns['check'].sum())
print(reason_columns['check'].unique())

# Drop check column
reason_columns = reason_columns.drop(['check'], axis=1)

# Remove multicollinearity
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)

# =========================
# Group Reasons into Categories
# =========================
reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

# =========================
# Drop Reason Column
# =========================
df = df.drop(['Reason for Absence'], axis=1)

# Add grouped reasons
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

# Rename columns
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                'Daily Work Load Average', 'Body Mass Index', 'Education',
                'Children', 'Pets', 'Absenteeism Time in Hours',
                'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']

df.columns = column_names

# =========================
# Reorder Columns
# =========================
column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
                          'Date', 'Transportation Expense', 'Distance to Work', 'Age',
                          'Daily Work Load Average', 'Body Mass Index',
                          'Education', 'Children', 'Pets',
                          'Absenteeism Time in Hours']

df = df[column_names_reordered]

# =========================
# Convert Date Column
# =========================
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# =========================
# Extract Month Value
# =========================
list_months = []

for i in range(len(df)):
    list_months.append(df['Date'][i].month)

df['Month Value'] = list_months

# =========================
# Extract Day of the Week
# =========================
def date_to_weekday(date_value):
    return date_value.weekday()

df['Day of the Week'] = df['Date'].apply(date_to_weekday)

# =========================
# Final Reorder Columns
# =========================
df = df.drop(['Date'], axis=1)

final_columns = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
                 'Month Value', 'Day of the Week',
                 'Transportation Expense', 'Distance to Work', 'Age',
                 'Daily Work Load Average', 'Body Mass Index',
                 'Education', 'Children', 'Pets',
                 'Absenteeism Time in Hours']

df = df[final_columns]

# =========================
# Education Mapping (Binary)
# =========================
print(df['Education'].unique())
print(df['Education'].value_counts())

df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

print(df['Education'].value_counts())

# =========================
# Final Clean Dataset
# =========================
df_cleaned = df.copy()

print(df_cleaned.head())

# Save Clean Data
df_cleaned.to_csv('Absenteeism_preprocessed.csv', index=False)

print("Preprocessing Complete. File Saved.")
