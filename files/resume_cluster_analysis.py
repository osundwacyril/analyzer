import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

from files.resume_processor import ResumeProcessor

class ResumeProcessorWithClustering(ResumeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_full = None  # includes all metadata + clustering

    def process_resumes(self):
        super().process_resumes()

        # Load original data
        df = pd.read_csv(self.csv_path)
        text_data = []
        meta_data = []

        for feat in self.extracted_features:
            applicant = feat["Applicant"]
            txt_files = [f for f in os.listdir(self.txt_dir) if f.startswith(applicant) and f.endswith('.txt')]
            if not txt_files:
                continue
            with open(os.path.join(self.txt_dir, txt_files[0]), 'r', encoding='utf-8') as f:
                text = f.read()
            text_data.append(text)
            meta_data.append(feat)

        if not text_data:
            print("❌ No resume texts found for clustering.")
            return

        self.df_full = pd.DataFrame(meta_data)
        self.df_full['resume_text'] = text_data

        # Merge hiring status if available
        original_df = pd.read_csv(self.csv_path)
        original_df['Applicant'] = original_df['Applicant'].astype(str)
        self.df_full = pd.merge(self.df_full, original_df[['Applicant', 'final hiring status']], on='Applicant', how='left')

        self._cluster_and_analyze()

    def _cluster_and_analyze(self, n_clusters=5):
        print("\n🔍 Performing K-Means clustering...")

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(self.df_full['resume_text'])
        self.vectorizer = vectorizer
        self.X = X


        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df_full['cluster'] = model.fit_predict(X)

        print(f"✅ Clustered into {n_clusters} groups.\n")

        for cluster_id in sorted(self.df_full['cluster'].unique()):
            cluster_df = self.df_full[self.df_full['cluster'] == cluster_id]
            success_df = cluster_df[cluster_df['final hiring status'].str.lower().str.contains("hired", na=False)]
            fail_df = cluster_df[~cluster_df['final hiring status'].str.lower().str.contains("hired", na=False)]

            print(f"\n📁 Cluster {cluster_id}:")
            print(f" - Total Resumes: {len(cluster_df)}")
            print(f" - Hired: {len(success_df)} ({100 * len(success_df)/len(cluster_df):.1f}%)")
            print(f" - Avg. Experience (All): {cluster_df['Years of Experience'].astype(int).mean():.2f} years")
            print(f" - Avg. Experience (Hired): {success_df['Years of Experience'].astype(int).mean():.2f} years")
            print(" - Top Skills (Hired):")
            print("   ", Counter(", ".join(success_df['Skills']).split(', ')).most_common(5))

            print(" - Top Keywords:")
            top_words = self._top_keywords(cluster_df['resume_text'].tolist(), vectorizer, X[cluster_df.index])
            print("   ", top_words)

    def _top_keywords(self, texts, vectorizer, X_subset):
        terms = vectorizer.get_feature_names_out()
        cluster_tfidf = X_subset.mean(axis=0).A1
        top_indices = cluster_tfidf.argsort()[::-1][:5]
        return [terms[i] for i in top_indices]

    def export_clusters_to_csv(self, filename="clustered_resumes.csv"):
        if self.df_full is not None:
            self.df_full.to_csv(filename, index=False)
            print(f"\n📁 Cluster data exported to {filename}")
        else:
            print("❌ No clustered data to export.")

    def generate_cluster_summary(self, filename="cluster_summary.txt"):
        if self.df_full is None:
            print("❌ No data to summarize.")
            return

        with open(filename, "w", encoding='utf-8') as f:
            for cluster_id in sorted(self.df_full['cluster'].unique()):
                cluster_df = self.df_full[self.df_full['cluster'] == cluster_id]
                success_df = cluster_df[cluster_df['final hiring status'].str.lower().str.contains("hired", na=False)]

                f.write(f"\n--- Cluster {cluster_id} ---\n")
                f.write(f"Total: {len(cluster_df)}\n")
                f.write(f"Hired: {len(success_df)} ({100 * len(success_df) / len(cluster_df):.1f}%)\n")
                f.write(f"Avg Experience: {cluster_df['Years of Experience'].astype(int).mean():.2f} yrs\n")
                f.write(
                    "Top Skills: " + str(Counter(", ".join(success_df['Skills']).split(', ')).most_common(5)) + "\n")
                top_words = self._top_keywords(cluster_df['resume_text'].tolist(), self.vectorizer,
                                               self.X[cluster_df.index])
                f.write("Top Keywords: " + ", ".join(top_words) + "\n")

        print(f"\n📄 Cluster summary saved to {filename}")

    def plot_clusters(self, method='pca', filename="cluster_plot.png"):
        if self.df_full is None:
            print("❌ No data to visualize.")
            return

        print(f"\n📊 Generating cluster plot using {method.upper()}...")

        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            print("❌ Unsupported method. Use 'pca' or 'tsne'.")
            return

        reduced = reducer.fit_transform(self.X.toarray())
        self.df_full['x'] = reduced[:, 0]
        self.df_full['y'] = reduced[:, 1]

        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x='x', y='y',
            hue='cluster',
            palette='tab10',
            style=self.df_full['final hiring status'].str.contains("hired", case=False, na=False),
            data=self.df_full
        )
        plt.title(f"Resume Clustering ({method.upper()})")
        plt.legend(title='Cluster / Hired')
        plt.savefig(filename)
        plt.show()
        print(f"📍 Plot saved to {filename}")
