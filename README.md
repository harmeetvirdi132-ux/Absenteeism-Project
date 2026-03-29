# Absenteeism-Project
import pandas as pd
import numpy as np


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    # Drop ID column
    df = df.drop(['ID'], axis=1)

    # Create dummy variables for Reason for Absence
    reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)

    # Group reasons
    reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
    reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
    reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
    reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

    # Drop original column
    df = df.drop(['Reason for Absence'], axis=1)

    # Concatenate grouped reasons
    df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

    # Rename columns
    column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                    'Daily Work Load Average', 'Body Mass Index', 'Education',
                    'Children', 'Pets', 'Absenteeism Time in Hours',
                    'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']

    df.columns = column_names

    # Reorder columns
    column_order = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date',
                    'Transportation Expense', 'Distance to Work', 'Age',
                    'Daily Work Load Average', 'Body Mass Index', 'Education',
                    'Children', 'Pets', 'Absenteeism Time in Hours']

    df = df[column_order]

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract month and weekday
    df['Month Value'] = df['Date'].dt.month
    df['Day of the Week'] = df['Date'].dt.weekday

    # Drop Date column
    df = df.drop(['Date'], axis=1)

    # Modify Education column
    df['Education'] = df['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})

    return df


def save_data(df, path):
    df.to_csv(path, index=False)


def main():
    input_path = '../data/Absenteeism_data.csv'
    output_path = '../output/Absenteeism_preprocessed.csv'

    df = load_data(input_path)
    df_processed = preprocess_data(df)
    save_data(df_processed, output_path)

    print("Data preprocessing completed and file saved.")


if __name__ == "__main__":
    main()
