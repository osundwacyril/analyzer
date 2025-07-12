import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('Outreacher Role - Sheet1 (3).csv')

# Display the first 5 rows of the DataFrame
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Display information about the DataFrame
print(df.info())


# Print column names before dropping to inspect
print("Columns before dropping:", df.columns.tolist())

# Drop irrelevant and duplicate columns
# Only include columns in the list that actually exist in the DataFrame
columns_to_drop = [col for col in ['Unnamed: 7', 'Unnamed: 13', 'Resume.1', 'Status.1', 'final hiring status.1', 'Updates Log', 'Contact'] if col in df.columns]
df = df.drop(columns=columns_to_drop)

# Print column names after dropping to confirm
print("Columns after dropping:", df.columns.tolist())

# Remove empty rows
df = df.dropna(how='all')

# Handle missing values
# For 'Applicant' and 'Hired Candidate', fill missing values with 'Unknown'
if 'Applicant' in df.columns:
    df['Applicant'] = df['Applicant'].fillna('Unknown')
if 'Hired Candidate' in df.columns:
    df['Hired Candidate'] = df['Hired Candidate'].fillna('Unknown')

# For 'Country', fill missing values with 'Unknown'
if 'Country' in df.columns:
    df['Country'] = df['Country'].fillna('Unknown')

# For 'Valid Contact Number', fill missing values with 'Unknown'
if 'Valid Contact Number' in df.columns:
    df['Valid Contact Number'] = df['Valid Contact Number'].fillna('Unknown')
else:
    print("Warning: 'Valid Contact Number' column not found for missing value handling.")

# For 'Remote experience', 'Salary Acceptable (1)', 'Salary acceptance', fill with mode
for col in ['Remote experience', 'Salary Acceptable (1)', 'Salary acceptance']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        print(f"Warning: Column '{col}' not found, skipping mode imputation.")

# Clean up 'Applicant' column to extract only the name
if 'Applicant' in df.columns:
    df['Applicant'] = df['Applicant'].apply(lambda x: x.split('<')[0].strip() if '<' in str(x) else x)

# Clean up 'Hired Candidate' column to extract only the name
if 'Hired Candidate' in df.columns:
    df['Hired Candidate'] = df['Hired Candidate'].apply(lambda x: x.split('<')[0].strip() if '<' in str(x) else x)

# Data type conversion for 'Valid Contact Number' to string
if 'Valid Contact Number' in df.columns:
    df['Valid Contact Number'] = df['Valid Contact Number'].astype(str)
else:
    print("Warning: 'Valid Contact Number' column not found for data type conversion.")

# Categorical encoding
# Convert 'Remote experience', 'Salary Acceptable (1)', 'Salary acceptance' to binary (Yes/No to 1/0)
binary_cols = ['Remote experience', 'Salary Acceptable (1)', 'Salary acceptance']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
    else:
        print(f"Warning: Column '{col}' not found, skipping binary encoding.")

# Display the first 5 rows of the preprocessed DataFrame
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Display information about the preprocessed DataFrame
print(df.info())

# Save the preprocessed DataFrame to a new CSV file
df.to_csv('preprocessed_outreacher_role.csv', index=False)