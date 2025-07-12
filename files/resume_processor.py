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
        "problem solving", "critical thinking", "adaptability", "creativity"
    ]

    def __init__(self, csv_path, pdf_dir='downloaded_resumes', txt_dir='extracted_resumes'):
        self.csv_path = csv_path
        self.pdf_dir = pdf_dir
        self.txt_dir = txt_dir
        self.extracted_features = []
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.txt_dir, exist_ok=True)
        logging.info(f"Initialized ResumeProcessor with CSV: {self.csv_path}")

    def _extract_file_id(self, url):
        match = re.search(r'(?:id=|file/d/)([a-zA-Z0-9_-]+)', url)
        return match.group(1) if match else None

    def _download_pdf(self, file_id, output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, output_path, quiet=True, fuzzy=True)
            logging.info(f"Successfully downloaded {os.path.basename(output_path)}")
            return True
        except Exception as e:
            logging.error(f"Download failed for file ID {file_id}: {e}")
            return False

    def _extract_pdf_text(self, pdf_path):
        try:
            text = extract_text(pdf_path)
            logging.info(f"Successfully extracted text from {os.path.basename(pdf_path)}")
            return text
        except Exception as e:
            logging.error(f"Text extraction failed for {os.path.basename(pdf_path)}: {e}")
            return None

    def _extract_skills(self, text):
        text_lower = text.lower()
        found_skills = [skill for skill in self.COMMON_SKILLS if re.search(rf'\\b{re.escape(skill)}\\b', text_lower)]
        return found_skills

    def process_resumes(self):
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

            if not os.path.exists(pdf_path):
                logging.info(f"[{idx}] Downloading {pdf_name}...")
                if not self._download_pdf(file_id, pdf_path):
                    continue
            else:
                logging.info(f"[{idx}] PDF already downloaded: {pdf_name}")

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
                skills = self._extract_skills(text)
                self.extracted_features.append({
                    "Applicant": applicant_name,
                    "Skills": ", ".join(skills) if skills else "N/A",
                    "Resume_Text_File": txt_name,
                    "Text": text
                })
            else:
                logging.warning(f"Skipping feature extraction for {applicant_name} due to no text found.")

        logging.info("Resume processing complete.")
        self.save_results()
        self._cluster_resumes()

    def save_results(self, out_file='extracted_features.csv'):
        if self.extracted_features:
            df = pd.DataFrame(self.extracted_features)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            df.to_csv(out_file, index=False)
            logging.info(f"Extracted features saved to: {out_file}")
        else:
            logging.warning("No features extracted to save.")

    def _cluster_resumes(self, n_clusters=5):
        df = pd.DataFrame(self.extracted_features)
        if df.empty:
            logging.warning("No data available for clustering.")
            return

        tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
        X = tfidf.fit_transform(df['Text'])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X.toarray())
        df['x_pca'] = reduced[:, 0]
        df['y_pca'] = reduced[:, 1]

        tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(df) - 1)), random_state=42)
        tsne_result = tsne.fit_transform(X.toarray())
        df['x_tsne'] = tsne_result[:, 0]
        df['y_tsne'] = tsne_result[:, 1]

        os.makedirs('outreach', exist_ok=True)
        df.to_csv('outreach/clustered_resumes.csv', index=False)

        with open("outreach/cluster_summary.txt", "w", encoding='utf-8') as f:
            for cluster_id in sorted(df['Cluster'].unique()):
                cluster_df = df[df['Cluster'] == cluster_id]
                skills = [skill.strip() for skills in cluster_df['Skills'] if skills != "N/A" for skill in skills.split(",")]
                top_skills = Counter(skills).most_common(5)
                f.write(f"--- Cluster {cluster_id} ---\n")
                f.write(f"Total resumes: {len(cluster_df)}\n")
                f.write(f"Top Skills: {top_skills}\n\n")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='x_pca', y='y_pca', hue='Cluster', palette='tab10')
        plt.title('Resume Clusters (PCA)')
        plt.savefig('outreach/resume_clusters_pca.png')
        plt.clf()

        sns.scatterplot(data=df, x='x_tsne', y='y_tsne', hue='Cluster', palette='tab10')
        plt.title('Resume Clusters (t-SNE)')
        plt.savefig('outreach/resume_clusters_tsne.png')
        plt.clf()

        logging.info("Clustering and visualizations complete.")

if __name__ == '__main__':
    os.makedirs("outreach", exist_ok=True)
    if not os.path.exists('cleaned'):
        logging.warning("Directory 'cleaned' not found. Please ensure CSV is in the correct path.")

    processor = ResumeProcessor(
        csv_path='cleaned/preprocessed_outreacher_role.csv',
        pdf_dir='out/downloaded_resumes',
        txt_dir='out/extracted_resumes'
    )
    processor.process_resumes()
    processor.save_results('outreach/extracted_features.csv')
