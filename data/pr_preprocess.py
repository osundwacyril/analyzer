import pandas as pd

# Load the dataset
df = pd.read_csv('Prospector Role - Sheet1 (3).csv')

# Drop columns with excessive missing values, safely checking if they exist
columns_to_drop = [
    'Unnamed: 11', 'Unnamed: 14', 'Hired candidate', 'contact',
    'resume', 'status', 'updated', 'final hiring status'
]
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
if existing_columns_to_drop:
    df = df.drop(columns=existing_columns_to_drop)
    print(f"\nDropped columns: {existing_columns_to_drop}")
else:
    print("\nNo specified columns to drop were found in the DataFrame.")

print("\nDataFrame info after dropping empty columns:")
print(df.info())

# Re-check for missing values after dropping columns
print("\nMissing values after dropping empty columns:")
print(df.isnull().sum())

# Handle missing values for remaining columns
# For 'Region', since it has a large number of missing values, and it's categorical,
# let's fill missing values with 'Unknown'.
if 'Region' in df.columns:
    df['Region'] = df['Region'].fillna('Unknown')
    print("\nMissing values in 'Region' imputed with 'Unknown'.")

# For other object type columns with missing values
object_cols_to_fill = [
    'Applicant', 'Valid Contact Number', 'CV', 'Hiring Status', 'Updates Log',
    'Final hiring status', 'Remote experience',
    'Is the salary range stated acceptable to you?', 'Salary acceptable'
]

for col in object_cols_to_fill:
    if col in df.columns:
        df[col] = df[col].fillna('Missing')
        print(f"Missing values in '{col}' imputed with 'Missing'.")

# Process 'Test score R1' column
if 'Test score R1' in df.columns:
    # Convert 'Yes' to 1, 'No' to 0, remove '%' and convert to numeric
    df['Test score R1_cleaned'] = df['Test score R1'].astype(str).str.replace('%', '', regex=False)
    df['Test score R1_cleaned'] = df['Test score R1_cleaned'].replace({'Yes': '1', 'No': '0'})
    df['Test score R1_cleaned'] = pd.to_numeric(df['Test score R1_cleaned'], errors='coerce')

    # Fill any remaining NaNs with the mean of the cleaned numeric column
    if df['Test score R1_cleaned'].isnull().any():
        mean_test_score = df['Test score R1_cleaned'].mean()
        df['Test score R1_cleaned'] = df['Test score R1_cleaned'].fillna(mean_test_score)
        print(f"\n'Test score R1' cleaned, converted to numeric, and missing/non-numeric values imputed with mean: {mean_test_score:.2f}.")
    else:
        print("\n'Test score R1' successfully cleaned and converted to numeric (no missing values after conversion).")

    df = df.drop(columns=['Test score R1']) # Drop the original column
    df = df.rename(columns={'Test score R1_cleaned': 'Test score R1'}) # Rename the new numeric column
    print("\nNew 'Test score R1' column info:")
    print(df['Test score R1'].info())

# Re-check missing values after all imputations
print("\nMissing values after all imputations:")
print(df.isnull().sum())

# Save the preprocessed data
output_file_name = 'preprocessed_prospector_data_cleaned.csv'
df.to_csv(output_file_name, index=False)
print(f"\nCleaned and preprocessed data saved to '{output_file_name}'")


