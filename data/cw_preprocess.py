import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('Content Writer - Sheet1 (3).csv')

# Display the first few rows
print(df.head())

# Get information about the DataFrame, including data types and non-null values
print(df.info())

# Drop the columns that are completely empty
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 10', 'Unnamed: 13'])

# Clean column names by stripping whitespace and removing '\n'
df.columns = df.columns.str.strip().str.replace('\n', '')

# Rename columns to ensure consistency and readability
df = df.rename(columns={
    'Contact Number': 'Contact Number',
    'Please upload your resume here.': 'Resume Link',
    'Is the salary range quoted, acceptable to you?': 'Salary Acceptable Question',
    'Salary acceptabe': 'Salary Acceptable',
})

# Identify columns that are almost entirely empty (less than 10% non-null values) and seem to be duplicates/redundant
low_data_columns = [col for col in df.columns if df[col].count() < 0.1 * len(df) and col not in ['Applicant', 'Contact Number', 'Resume Link', 'Status', 'Updates Log', 'Final hiring status']]
df = df.drop(columns=low_data_columns)

# Impute missing values in the specified columns with 'Unknown'
for col in ['Applicant', 'Status', 'Updates Log', 'Final hiring status']:
    if col in df.columns: # Check if column still exists after dropping
        df[col] = df[col].fillna('Unknown')

# Display the first few rows after preprocessing
print("DataFrame after preprocessing (first 5 rows):")
print(df.head())

# Get information about the DataFrame after preprocessing
print("\nDataFrame info after preprocessing:")
print(df.info())

# Save the preprocessed data to a new CSV file
df.to_csv('preprocessed_content_writer_data.csv', index=False)