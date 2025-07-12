import os
import re
import pandas as pd
import gdown
from pdfminer.high_level import extract_text
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    exit()


class ResumeProcessor:
    """
    A class to process resumes, extract features, and prepare data for clustering.
    """
    COMMON_SKILLS = [
        "python", "java", "sql", "excel", "powerpoint", "communication",
        "project management", "machine learning", "deep learning", "leadership",
        "aws", "azure", "google cloud", "data analysis", "crm", "seo", "marketing",
        "data science", "r", "tableau", "power bi", "statistics", "nlp", "computer vision",
        "scikit-learn", "tensorflow", "pytorch", "agile", "scrum", "git", "linux",
        "docker", "kubernetes", "frontend", "backend", "full-stack", "javascript",
        "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
        "spring", "c++", "c#", "devops", "cloud computing", "cybersecurity",
        "network security", "risk management", "business analysis", "financial modeling",
        "microsoft office", "customer service", "sales", "marketing strategy",
        "social media", "content creation", "public speaking", "teamwork",
        "problem solving", "critical thinking", "adaptability", "creativity",
        # Specific skills for Content Writer
        "content writing", "copywriting", "editing", "proofreading", "seo writing",
        "blogging", "article writing", "content marketing", "grammar", "storytelling",
        "digital marketing", "social media management", "web content", "creative writing",
        "journalism", "technical writing", "research", "keyword research",
        "content strategy", "brand voice", "cms", "wordpress", "joomla", "drupal",
        "email marketing", "newsletter", "press release", "white paper", "case study"
    ]

    def __init__(self, csv_path, pdf_dir='downloaded_resumes', txt_dir='extracted_resumes'):
        """
        Initializes the ResumeProcessor.

        Args:
            csv_path (str): Path to the CSV file containing resume URLs and applicant names.
            pdf_dir (str): Directory to store downloaded PDF resumes.
            txt_dir (str): Directory to store extracted text from resumes.
        """
        self.csv_path = csv_path
        self.pdf_dir = pdf_dir
        self.txt_dir = txt_dir
        self.extracted_features = []
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.txt_dir, exist_ok=True)
        logging.info(f"Initialized ResumeProcessor with CSV: {self.csv_path}")

    def _extract_file_id(self, url):
        """
        Extracts the Google Drive file ID from a URL.
        """
        match = re.search(r'(?:id=|file/d/)([a-zA-Z0-9_-]+)', url)
        return match.group(1) if match else None

    def _download_pdf(self, file_id, output_path):
        """
        Downloads a PDF from Google Drive using its file ID.
        """
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, output_path, quiet=True, fuzzy=True)
            logging.info(f"Successfully downloaded {os.path.basename(output_path)}")
            return True
        except Exception as e:
            logging.error(f"Download failed for file ID {file_id}: {e}")
            return False

    def _extract_pdf_text(self, pdf_path):
        """
        Extracts text from a PDF file.
        """
        try:
            text = extract_text(pdf_path)
            logging.info(f"Successfully extracted text from {os.path.basename(pdf_path)}")
            return text
        except Exception as e:
            logging.error(f"Text extraction failed for {os.path.basename(pdf_path)}: {e}")
            return None

    def _extract_skills(self, text):
        """
        Extracts common skills from the resume text.
        """
        text_lower = text.lower()
        found_skills = [skill for skill in self.COMMON_SKILLS if skill in text_lower]
        return found_skills

    def _extract_education(self, text):
        """
        Extracts education highlights from the resume text.
        """
        education_keywords = [
            "bachelor", "master", "phd", "mba", "b.sc", "m.sc", "bs", "ms", "university",
            "college", "degree", "diploma", "associate", "doctorate"
        ]
        doc = nlp(text)
        matches = []
        for sent in doc.sents:
            if any(kw in sent.text.lower() for kw in education_keywords):
                matches.append(sent.text.strip())
        return matches

    def _extract_experience_years(self, text):
        """
        Extracts years of experience from the resume text using various patterns.
        """
        patterns = [
            r"(\d+)\s*years?\s+of\s+experience",
            r"experience\s+of\s+(\d+)\s*years?",
            r"(\d+)\+?\s*years?\s*(?:working|in\s+\w+)?\s*(?:experience)?",
            r"(\d+)\s*yrs?\s*(?:exp)?",
            r"(\d+)\s*-\s*(\d+)\s*years?\s+experience"
        ]
        found = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if '-' in match.group(0):
                    parts = re.findall(r'\d+', match.group(0))
                    if len(parts) == 2:
                        found.append(int(parts[1]))
                else:
                    digit_match = re.search(r'\d+', match.group(0))
                    if digit_match:
                        found.append(int(digit_match.group(0)))
        return max(found) if found else 0

    def process_resumes(self):
        """
        Processes each resume listed in the CSV, downloads PDFs, extracts text,
        and extracts features.
        """
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            logging.error(f"CSV file not found at: {self.csv_path}")
            return

        if 'Resume' not in df.columns or 'Applicant' not in df.columns:
            logging.error("CSV must contain 'Resume' (URL) and 'Applicant' columns.")
            return

        for idx, row in df.iterrows():
            resume_url = str(row['Resume'])
            applicant_name = re.sub(r'[\\/:*?"<>|]', '_', str(row['Applicant']).strip())

            if not resume_url or resume_url.lower() == 'unknown' or pd.isna(resume_url):
                logging.info(f"Skipping row {idx}: Invalid or unknown resume URL.")
                continue

            file_id = self._extract_file_id(resume_url)
            if not file_id:
                logging.warning(f"Skipping row {idx}: Could not extract file ID from URL: {resume_url}")
                continue

            pdf_name = f"{applicant_name}_{file_id}.pdf"
            txt_name = f"{applicant_name}_{file_id}.txt"
            pdf_path = os.path.join(self.pdf_dir, pdf_name)
            txt_path = os.path.join(self.txt_dir, txt_name)

            # Download PDF
            if not os.path.exists(pdf_path):
                logging.info(f"[{idx}] Downloading {pdf_name}...")
                if not self._download_pdf(file_id, pdf_path):
                    continue
            else:
                logging.info(f"[{idx}] PDF already downloaded: {pdf_name}")

            # Extract text
            text = None
            if not os.path.exists(txt_path):
                logging.info(f"[{idx}] Extracting text from {pdf_name}...")
                text = self._extract_pdf_text(pdf_path)
                if text:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                else:
                    continue
            else:
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    logging.info(f"[{idx}] Text already extracted: {txt_name}")
                except Exception as e:
                    logging.error(f"Could not read existing text file {txt_path}: {e}")
                    continue

            if text:
                # Extract features
                skills = self._extract_skills(text)
                education = self._extract_education(text)
                years_experience = self._extract_experience_years(text)

                self.extracted_features.append({
                    "Applicant": applicant_name,
                    "Skills": ", ".join(skills) if skills else "N/A",
                    "Education Highlights": "; ".join(education[:3]) if education else "N/A",
                    "Years of Experience": years_experience,
                    "Resume_Text_File": txt_name
                })
            else:
                logging.warning(f"Skipping feature extraction for {applicant_name} due to no text found.")

        logging.info("Resume processing complete.")
        self.save_results()

    def save_results(self, out_file='extracted_features.csv'):
        """
        Saves the extracted features to a CSV file.
        """
        if self.extracted_features:
            df = pd.DataFrame(self.extracted_features)
            df.to_csv(out_file, index=False)
            logging.info(f"Extracted features saved to: {out_file}")
        else:
            logging.warning("No features extracted to save.")


class ResumeProcessorWithClustering(ResumeProcessor):
    """
    Extends ResumeProcessor to include K-Means clustering and visualization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_full = None
        self.vectorizer = None
        self.X = None

    def process_resumes(self):
        """
        Orchestrates resume processing, then performs clustering and analysis.
        """
        super().process_resumes()

        if not self.extracted_features:
            logging.warning("No extracted features available for clustering.")
            return

        # Prepare data for clustering
        text_data = []
        meta_data = []

        for feat in self.extracted_features:
            txt_name = feat["Resume_Text_File"]
            txt_path = os.path.join(self.txt_dir, txt_name)
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    text_data.append(text)
                    meta_data.append(feat)
                except Exception as e:
                    logging.error(f"Could not read text file {txt_path} for clustering: {e}")
            else:
                logging.warning(f"Text file {txt_path} not found for clustering. Skipping.")

        if not text_data:
            logging.error("No resume texts found for clustering after extraction.")
            return

        self.df_full = pd.DataFrame(meta_data)
        self.df_full['resume_text'] = text_data

        # Merge hiring status from the original CSV
        try:
            original_df = pd.read_csv(self.csv_path)
            original_df['Applicant'] = original_df['Applicant'].astype(str)
            self.df_full = pd.merge(self.df_full, original_df[['Applicant', 'final hiring status']], on='Applicant', how='left')
            logging.info("Merged 'final hiring status' from original CSV.")
        except Exception as e:
            logging.warning(f"Could not merge 'final hiring status' (CSV might be missing column or file): {e}")
            self.df_full['final hiring status'] = "Unknown"

        self._cluster_and_analyze()

    def _cluster_and_analyze(self, n_clusters=5):
        """
        Performs K-Means clustering on resume texts and analyzes the clusters.
        """
        logging.info(f"Performing K-Means clustering with {n_clusters} clusters...")

        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))
        self.X = self.vectorizer.fit_transform(self.df_full['resume_text'])
        logging.info(f"TF-IDF matrix created with {self.X.shape[1]} features.")

        # K-Means Clustering
        try:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.df_full['cluster'] = model.fit_predict(self.X)
            logging.info(f"Clustering complete. {n_clusters} groups formed.")
        except Exception as e:
            logging.error(f"K-Means clustering failed: {e}")
            return

        # Cluster Analysis
        for cluster_id in sorted(self.df_full['cluster'].unique()):
            cluster_df = self.df_full[self.df_full['cluster'] == cluster_id]
            success_df = cluster_df[cluster_df['final hiring status'].astype(str).str.lower().str.contains("hired", na=False)]
            fail_df = cluster_df[~cluster_df['final hiring status'].astype(str).str.lower().str.contains("hired", na=False)]

            total_resumes = len(cluster_df)
            hired_count = len(success_df)
            hired_percentage = (100 * hired_count / total_resumes) if total_resumes > 0 else 0

            avg_exp_all = cluster_df['Years of Experience'].astype(int).mean()
            avg_exp_hired = success_df['Years of Experience'].astype(int).mean() if hired_count > 0 else 0

            logging.info(f"\n--- Cluster {cluster_id} ---")
            logging.info(f"Total Resumes: {total_resumes}")
            logging.info(f"Hired: {hired_count} ({hired_percentage:.1f}%)")
            logging.info(f"Avg. Experience (All): {avg_exp_all:.2f} years")
            logging.info(f"Avg. Experience (Hired): {avg_exp_hired:.2f} years")

            # Top Skills (Hired Candidates)
            if not success_df.empty:
                all_hired_skills = [skill.strip() for skills_str in success_df['Skills'].dropna() for skill in skills_str.split(',')]
                top_hired_skills = Counter(all_hired_skills).most_common(5)
                logging.info(f"Top Skills (Hired): {top_hired_skills}")
            else:
                logging.info("Top Skills (Hired): None (no hired candidates in this cluster)")

            # Top Keywords (Overall Cluster)
            if not cluster_df.empty:
                top_keywords = self._top_keywords(cluster_df['resume_text'].tolist(), self.vectorizer, self.X[cluster_df.index])
                logging.info(f"Top Keywords (Cluster {cluster_id}): {top_keywords}")
            else:
                logging.info(f"Top Keywords (Cluster {cluster_id}): None (empty cluster)")

    def _top_keywords(self, texts, vectorizer, X_subset):
        """
        Identifies the top keywords for a given set of texts within a cluster.
        """
        if X_subset.shape[0] == 0:
            return []
        terms = vectorizer.get_feature_names_out()
        cluster_tfidf = X_subset.mean(axis=0).A1
        top_indices = cluster_tfidf.argsort()[::-1][:10]
        return [terms[i] for i in top_indices]

    def export_clusters_to_csv(self, filename="clustered_resumes.csv"):
        """
        Exports the full DataFrame with cluster assignments to a CSV file.
        """
        if self.df_full is not None:
            self.df_full.to_csv(filename, index=False)
            logging.info(f"Cluster data exported to {filename}")
        else:
            logging.warning("No clustered data to export.")

    def generate_cluster_summary(self, filename="cluster_summary.txt"):
        """
        Generates a summary of each cluster and saves it to a text file.
        """
        if self.df_full is None:
            logging.warning("No data to summarize.")
            return

        with open(filename, "w", encoding='utf-8') as f:
            f.write("Resume Clustering Summary\n")
            f.write("=" * 30 + "\n\n")
            for cluster_id in sorted(self.df_full['cluster'].unique()):
                cluster_df = self.df_full[self.df_full['cluster'] == cluster_id]
                success_df = cluster_df[cluster_df['final hiring status'].astype(str).str.lower().str.contains("hired", na=False)]

                total_resumes = len(cluster_df)
                hired_count = len(success_df)
                hired_percentage = (100 * hired_count / total_resumes) if total_resumes > 0 else 0
                avg_exp_all = cluster_df['Years of Experience'].astype(int).mean()
                avg_exp_hired = success_df['Years of Experience'].astype(int).mean() if hired_count > 0 else 0

                f.write(f"\n--- Cluster {cluster_id} ---\n")
                f.write(f"Total Resumes: {total_resumes}\n")
                f.write(f"Hired: {hired_count} ({hired_percentage:.1f}%)\n")
                f.write(f"Average Experience (All): {avg_exp_all:.2f} years\n")
                f.write(f"Average Experience (Hired): {avg_exp_hired:.2f} years\n")

                if not success_df.empty:
                    all_hired_skills = [skill.strip() for skills_str in success_df['Skills'].dropna() for skill in skills_str.split(',')]
                    top_hired_skills = Counter(all_hired_skills).most_common(5)
                    f.write(f"Top Skills (Hired): {top_hired_skills}\n")
                else:
                    f.write("Top Skills (Hired): None (no hired candidates in this cluster)\n")

                if not cluster_df.empty:
                    top_words = self._top_keywords(cluster_df['resume_text'].tolist(), self.vectorizer,
                                                   self.X[cluster_df.index])
                    f.write(f"Top Keywords (Overall Cluster): {', '.join(top_words)}\n")
                else:
                    f.write("Top Keywords (Overall Cluster): None (empty cluster)\n")

        logging.info(f"Cluster summary saved to {filename}")

    def plot_clusters(self, method='pca', filename="cluster_plot.png"):
        """
        Generates and saves a 2D scatter plot of the clusters using PCA or t-SNE.
        Points are colored by cluster and styled by hiring status.
        """
        if self.df_full is None or self.X is None:
            logging.warning("No clustered data or TF-IDF matrix to visualize.")
            return

        logging.info(f"Generating cluster plot using {method.upper()}...")

        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'tsne':
            n_samples = self.X.shape[0]
            perplexity_val = min(30, max(5, n_samples - 1)) if n_samples > 1 else 1
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_jobs=-1)
        else:
            logging.error("Unsupported method. Use 'pca' or 'tsne'.")
            return

        try:
            reduced = reducer.fit_transform(self.X.toarray())
            self.df_full['x'] = reduced[:, 0]
            self.df_full['y'] = reduced[:, 1]

            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                x='x', y='y',
                hue='cluster',
                palette='tab10',
                style=self.df_full['final hiring status'].astype(str).str.contains("hired", case=False, na=False),
                markers={True: 'o', False: 'X'},
                s=100,
                alpha=0.7,
                data=self.df_full
            )
            plt.title(f"Resume Clustering ({method.upper()})", fontsize=16)
            plt.xlabel(f"{method.upper()} Component 1", fontsize=12)
            plt.ylabel(f"{method.upper()} Component 2", fontsize=12)
            plt.legend(title='Cluster / Hired Status', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(filename, dpi=300)
            logging.info(f"Plot saved to {filename}")
        except Exception as e:
            logging.error(f"Error generating {method.upper()} plot: {e}")


# --- Main Execution ---
if __name__ == '__main__':
     # Ensure the 'cleaned' directory exists or adjust the path
    if not os.path.exists('files/cleaned'): # Changed to 'files/cleaned' to match typical structure
        logging.warning("Directory 'cleaned' not found. Please ensure 'preprocessed_content_writer_role.csv' is in the correct path.")
        # You might want to create it or handle this case based on your project structure

    # Initialize and run the processor with clustering capabilities for content writer role
    processor = ResumeProcessorWithClustering(
        csv_path='../cleaned/preprocessed_content_writer_data.csv',
        pdf_dir='content_resumes/downloaded_resumes',
        txt_dir='content_resumes/extracted_resumes'
    )
    processor.process_resumes() # This now triggers the clustering and analysis

    # Export results and generate summaries/plots
    os.makedirs('content_out', exist_ok=True) # Ensure output directory exists
    processor.export_clusters_to_csv("content_out/clustered_resumes.csv")
    processor.generate_cluster_summary("content_out/cluster_summary.txt")
    processor.plot_clusters(method='pca', filename="content_out/resume_clusters_pca.png")
    processor.plot_clusters(method='tsne', filename="content_out/resume_clusters_tsne.png")