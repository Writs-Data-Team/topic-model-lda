#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

def load_data(parquet_path):
    """Load parquet file, corpus, and dictionary"""
    print("Loading data...")
    df = pd.read_parquet(parquet_path)
    
    dictionary = Dictionary.load("data/dictionary.gensim")
    with open("data/corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    
    # Align data
    min_len = min(len(df), len(corpus))
    df = df.iloc[:min_len]
    corpus = corpus[:min_len]
    
    print(f"Processing {min_len} documents")
    return df, corpus

def get_topic_columns(df):
    """Find topic assignment columns"""
    topic_cols = {}
    for col in df.columns:
        for k in [14, 17, 20, 24]:
            if str(k) in col and 'topic' in col.lower():
                topic_cols[k] = col
                break
    return topic_cols

def get_topic_matrix(model, corpus):
    """Get document-topic matrix for all documents"""
    print(f"Converting {len(corpus)} documents to topic vectors...")
    
    doc_topic_matrix = np.zeros((len(corpus), model.num_topics))
    
    for i, doc in enumerate(corpus):
        doc_topics = model.get_document_topics(doc, minimum_probability=0.0)
        for topic_id, prob in doc_topics:
            doc_topic_matrix[i, topic_id] = prob
        
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{len(corpus)} documents")
    
    return doc_topic_matrix

def calculate_metrics(topic_matrix, labels):
    """Calculate core clustering metrics"""
    print("Calculating clustering metrics...")
    
    # Filter valid data
    valid_mask = ~np.isnan(topic_matrix).any(axis=1) & (labels >= 0)
    X = topic_matrix[valid_mask]
    y = labels[valid_mask]
    
    print(f"Valid documents: {len(X)}")
    
    if len(np.unique(y)) <= 1:
        return {'error': 'insufficient_clusters'}
    
    metrics = {}
    
    try:
        metrics['silhouette'] = silhouette_score(X, y)
    except Exception as e:
        print(f"Silhouette error: {e}")
        metrics['silhouette'] = None
    
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(X, y)
    except Exception as e:
        print(f"Davies-Bouldin error: {e}")
        metrics['davies_bouldin'] = None
    
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, y)
    except Exception as e:
        print(f"Calinski-Harabasz error: {e}")
        metrics['calinski_harabasz'] = None
    
    return metrics

def main(parquet_path):
    """Main function - calculate metrics for all k values on full dataset"""
    
    # Load data
    df, corpus = load_data(parquet_path)
    
    # Find topic columns
    topic_cols = get_topic_columns(df)
    if not topic_cols:
        print("No topic columns found. Available columns:")
        print(df.columns.tolist())
        return None
    
    print(f"Found topic columns: {topic_cols}")
    
    # Process each k value
    results = []
    
    for k, col_name in topic_cols.items():
        print(f"\n{'='*40}")
        print(f"Processing k={k}")
        print(f"{'='*40}")
        
        try:
            # Load model
            model = LdaMulticore.load(f"data/lda_model_k{k}.gensim")
            
            # Get topic matrix for all documents
            topic_matrix = get_topic_matrix(model, corpus)
            
            # Get labels
            labels = df[col_name].values
            
            # Calculate metrics
            metrics = calculate_metrics(topic_matrix, labels)
            metrics['k'] = k
            metrics['n_docs'] = len(df)
            for metric, value in metrics.items():
                if metric not in ['k', 'n_docs', 'error'] and value is not None:
                    print(f"{metric}: {value:.4f}")
            
            results.append(metrics)
            
        except Exception as e:
            print(f"Error with k={k}: {e}")
            results.append({'k': k, 'error': str(e)})
    
    # Final results
    if results:
        results_df = pd.DataFrame(results)
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        display_cols = ['k', 'silhouette', 'davies_bouldin', 'calinski_harabasz']
        valid_results = results_df[~results_df.get('error', pd.Series()).notna()]
        if not valid_results.empty:
            print(valid_results[display_cols].round(4).to_string(index=False))
        else:
            print("No valid results generated")
        results_df.to_csv('clustering_metrics_full.csv', index=False)
        print(f"\nAll results saved to 'clustering_metrics_full.csv'")
        
        return results_df
    else:
        print("No results generated!")
        return None

if __name__ == "__main__":
    PARQUET_PATH = "/media/f1_drive/topic-model-lda/topic_model/data/petitions_with_topics_allk.parquet"
    results = main(PARQUET_PATH)
