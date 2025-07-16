***Analyzer***

** Preprocessing**:
To preprocess your `Outreacher Role - Sheet1 (3).csv` file using the provided Python script `outreach_preprocess.py` located in the foder '/data' , follow these instructions:

### Prerequisites

  * **Python 3.x**: Make sure you have Python installed.
  * **pip**: Python's package installer, usually comes with Python.

### Step-by-Step Instructions

1.  **Save the Code**:

      * Save the provided code as `outreach_preprocess.py` in a directory of your choice.

2.  **Place Your Data File**:

      * Ensure your raw data CSV file, named `Outreacher Role - Sheet1 (3).csv`, is in the **same directory** as the `outreach_preprocess.py` script.

3.  **Install Dependencies**:

      * Open your terminal or command prompt.
      * Navigate to the directory where you saved `outreach_preprocess.py` and `Outreacher Role - Sheet1 (3).csv`.
      * Run the following command to install the necessary Python library:
        ```bash
        pip install pandas numpy
        ```

4.  **Run the Preprocessing Script**:

      * In the same terminal or command prompt, execute the script:
        ```bash
        python outreach_preprocess.py
        ```

### What the Script Does

The `outreach_preprocess.py` script performs the following data cleaning and preprocessing steps:

1.  **Loads Data**: Reads the `Outreacher Role - Sheet1 (3).csv` file into a pandas DataFrame.
2.  **Initial Inspection**: Prints the first 5 rows and a summary of the DataFrame's information (data types, non-null counts).
3.  **Column Dropping**:
      * Identifies and drops columns that are typically irrelevant or duplicate, specifically: `'Unnamed: 7'`, `'Unnamed: 13'`, `'Resume.1'`, `'Status.1'`, `'final hiring status.1'`, `'Updates Log'`, and `'Contact'`. It intelligently checks if these columns exist before attempting to drop them, printing warnings for missing columns.
4.  **Row Removal**: Deletes any rows that are completely empty (all `NaN` values).
5.  **Missing Value Handling**:
      * Fills missing values in the `'Applicant'`, `'Hired Candidate'`, `'Country'`, and `'Valid Contact Number'` columns with the string 'Unknown'.
      * Fills missing values in `'Remote experience'`, `'Salary Acceptable (1)'`, and `'Salary acceptance'` columns with their respective mode (most frequent value). Warnings are printed if these columns are not found.
6.  **Data Cleaning - Name Extraction**:
      * For the `'Applicant'` and `'Hired Candidate'` columns, it cleans the entries by extracting only the name part if the entry contains an email format (e.g., "Name [email@example.com](mailto:email@example.com)" becomes "Name").
7.  **Data Type Conversion**:
      * Converts the `'Valid Contact Number'` column to string type. A warning is printed if the column is not found.
8.  **Categorical Encoding**:
      * Transforms the `'Remote experience'`, `'Salary Acceptable (1)'`, and `'Salary acceptance'` columns into binary (1 or 0) where 'Yes' becomes 1 and anything else becomes 0. Warnings are printed if these columns are not found.
9.  **Final Inspection**: Prints the first 5 rows and information about the preprocessed DataFrame again to show the changes.
10. **Save Preprocessed Data**: Saves the cleaned DataFrame to a new CSV file named `preprocessed_outreacher_role.csv` in the same directory as the script, without including the DataFrame index.

After running the script, a new CSV file named `preprocessed_outreacher_role.csv` will be created in your working directory, containing the cleaned and transformed data.

### Prerequisites

Before you begin, ensure you have the following installed:

  * **Python 3.x**: The code is written in Python.
  * **pip**: Python's package installer, usually comes with Python.

### Step-by-Step Installation and Execution

1.  **Save the Code**: Save the provided code as `outreacher.py` in a directory of your choice.

2.  **Install Dependencies**: Open your terminal or command prompt, navigate to the directory where you saved `outreacher.py`, and run the following command to install all necessary Python libraries:

    ```bash
    pip install pandas pdfminer.six spacy scikit-learn matplotlib seaborn gdown
    ```

3.  **Download SpaCy Model**: The code uses SpaCy for natural language processing. You need to download the English small model. Run this command in your terminal:

    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **Prepare Your Data**:

      * **CSV File**: You need a CSV file containing resume URLs and applicant names. This file should have at least two columns: `'Resume'` (for Google Drive URLs of the resumes) and `'Applicant'` (for the applicant's name). Optionally, it can also include a `'final hiring status'` column for merging and analysis.

      * **File Path**: By default, the code expects this CSV file to be located at `../cleaned/preprocessed_outreacher_role.csv` relative to where you run `outreacher.py`.

          * **Create `cleaned` directory**: Create a directory named `cleaned` one level up from where `outreacher.py` is located.
          * **Place CSV**: Place your CSV file (e.g., `preprocessed_outreacher_role.csv`) inside this `cleaned` directory.

        *Example Directory Structure:*

        ```
        analyzer/files/
        ├── cleaned/
        │   └── preprocessed_outreacher_role.csv
        └── outreacher_role/
            └── outreacher.py
        ```

      * **Adjust CSV Path (if needed)**: If your CSV file is located elsewhere, you'll need to modify the `csv_path` in the `if __name__ == '__main__':` block of `outreacher.py`:

        ```python
        # In outreacher.py, find this line and modify if necessary:
        processor = ResumeProcessorWithClustering(
            csv_path='path/to/your/preprocessed_outreacher_role.csv', # <--- Modify this line
            pdf_dir='out/downloaded_resumes',
            txt_dir='out/extracted_resumes'
        )
        ```

5.  **Create Output Directories**: The script is configured to save downloaded PDFs, extracted texts, and results in specific output directories. It will automatically create these if they don't exist:

      * `out/downloaded_resumes`: For downloaded PDF files.**(ALREADY UPLOADED WITH THE PROJECT)**.
      * `out/extracted_resumes`: For text extracted from PDFs.**(ALREADY UPLOADED WITH THE PROJECT)**.
      * `outreach/`: For clustered data CSV, cluster summaries, and plots.

    You can manually create the `out` and `outreach` directories, or the script will handle it.

6.  **Run the Script**: Execute the `outreacher.py` script from your terminal:

    ```bash
    python outreacher.py
    ```

### What the Script Does When You Run It

The script will perform the following actions:

1.  **Initialization**: Sets up logging and checks for the SpaCy model.
2.  **Resume Processing (`ResumeProcessor`)**:
      * Reads the specified CSV file.
      * For each resume URL in the CSV:
          * Extracts the Google Drive file ID.
          * Downloads the PDF resume to `out/downloaded_resumes`.
          * Extracts text content from the PDF and saves it to `out/extracted_resumes`.
          * Extracts features like common skills, education highlights, and years of experience from the text.
      * Saves these extracted features to `extracted_features.csv`.
3.  **Clustering and Analysis (`ResumeProcessorWithClustering`)**:
      * Loads the extracted text data and merges it with the original CSV data (including 'final hiring status' if available).
      * Performs TF-IDF vectorization on the resume texts.
      * Applies K-Means clustering to group similar resumes (default 5 clusters).
      * Analyzes each cluster, providing:
          * Total resumes in the cluster.
          * Count and percentage of hired candidates.
          * Average years of experience for all candidates and for hired candidates.
          * Top 5 common skills among hired candidates in that cluster.
          * Top 10 overall keywords for the cluster.
4.  **Output Generation**:
      * **Clustered Data CSV**: Exports the full DataFrame, including cluster assignments and all extracted features, to a CSV file named `outreach/clustered_resumes_YYYYMMDD_HHMMSS.csv`.
      * **Cluster Summary Text File**: Generates a detailed summary of each cluster's characteristics and saves it to `outreach/cluster_summary_YYYYMMDD_HHMMSS.txt`.
      * **Cluster Plots**: Creates two 2D scatter plots visualizing the clusters:
          * One using Principal Component Analysis (PCA), saved as `outreach/resume_clusters_pca_YYYYMMDD_HHMMSS.png`.
          * One using t-distributed Stochastic Neighbor Embedding (t-SNE), saved as `outreach/resume_clusters_tsne_YYYYMMDD_HHMMSS.png`.
          * Points in the plots are colored by cluster and marked differently based on their 'final hiring status' (hired/not hired).

Logs will be printed to the console indicating the progress and any warnings or errors.
