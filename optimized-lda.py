#!/usr/bin/env python
# coding: utf-8

# Final topic model for SCOTUS petitions 
import pickle
import pyLDAvis
import pyLDAvis.gensim_models
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm
from collections import Counter
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel
import matplotlib.pyplot as plt
import multiprocessing

nltk.download('stopwords', quiet=True)
base_stop = set(stopwords.words('english'))

# --- LEGAL BLAH BLAH BLAH ---
legal_stop = {
    # Parties and legal actors
    "court", "judge", "justice", "plaintiff", "defendant", "appellant", "appellee",
    "respondent", "petitioner", "prosecutor", "defense", "counsel", "attorney", "attorneys",
    "lawyer", "firm", "litigant", "party", "parties", "represent", "representation",

    # Legal process and documents
    "petition", "brief", "writ", "writs", "opinion", "decision", "dissent", "concurrence",
    "judgment", "order", "motion", "motions", "plea", "pleaded", "pleading", "filing",
    "filed", "record", "docket", "summary", "memorandum", "transcript", "statement", "testimony",
    "exhibit", "appendix", "page", "pages", "volume", "footnote", "note", "citation", "citations",

    # Court actions and outcomes
    "certiorari", "remand", "dismiss", "affirm", "reverse", "vacate", "rehearing", "injunction",
    "sentence", "sentencing", "verdict", "arraignment", "indictment", "conviction", "acquittal",
    "retrial", "mistrial", "stay", "reversal", "hearing", "trial", "review", "jurisdiction",

    # Legal theory, reasoning, and doctrines
    "holding", "dicta", "precedent", "jurisprudence", "standard", "burden", "test", "mens",
    "rea", "actus", "reus", "prima", "facie", "res", "judicata", "stare", "decisis", "ratio",
    "obiter", "legal", "illegal", "unlawful", "liability", "negligence", "tort", "contract",
    "damages", "injury", "harm", "equity", "remedy", "estoppel", "waiver", "doctrine",

    # Legal language and Latin phrases
    "habeas", "corpus", "ex", "parte", "en", "banc", "voir", "dire", "nolo", "contendere",
    "amicus", "curiae", "per", "curiam", "de", "facto", "de", "jure", "inter", "alia", "ipso", "facto",
    "ultra", "vires", "sine", "qua", "non", "pro", "se", "in", "re", "sub", "judice",

    # Legal categories and law types
    "civil", "criminal", "statutory", "common", "constitutional", "federal", "state", "administrative",
    "tax", "property", "procedural", "substantive", "corporate", "securities", "antitrust", "employment",

    # Institutions and courts
    "supreme", "circuit", "district", "appeals", "tribunal", "agency", "commission", "board",
    "department", "bureau", "office", "government", "federal", "municipal", "state", "official", "officials",

    # Legal structure terms
    "act", "acts", "statute", "statutes", "clause", "clauses", "section", "sections", "subsection", "article",
    "regulation", "regulations", "rule", "rules", "code", "ordinance", "title", "chapter", "paragraph",

    # Time and reference noise
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "year", "date", "number", "volume", "page", "pages",

    # Corporate/legal entity suffixes
    "inc", "corp", "llc", "ltd", "company", "co", "group", "pllc", "plc", "pc", "partners", "lp", "llp",

    # Legal publication and citation shorthand
    "us", "l", "ed", "sup", "ct", "fed", "f", "2d", "3d", "cir", "so", "cal", "ny", "mass",
    "ill", "tex", "wash", "ala", "nc", "ga", "fl", "ariz", "ohio", "mich", "tenn", "mo", "kan",

    # Miscellaneous stop words in legal discourse
    "herein", "thereof", "hereto", "therein", "hereby", "whereas", "wherein", "thereby",
    "hereafter", "thereafter", "thereunder", "hereunder", "american", "subject", "matter",

    # Redundancies / common SCOTUS metadata
    "et", "al", "vs", "v", "united", "states", "government", "state", "people", "county", "city",
    "district", "school", "board", "department", "commission", "office",

    # Cleaning artifacts / OCR or text error tokens
    "cee", "cece", "een", "supp", "title", "cir", "inc", "fed", "corp",

    #miscellaneous stop words after runs
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", 
    "cause","allege", "must", "york", "tell", "get", "word", "supra", "see", "call",
     "must", "apply", "decide", "determine", "result", "effect",
    "know", "tell", "get", "offer", "ask", "call",
    "plan", "system", "mean", "language", "word", "report",
    "york", "texas", "california", "washington", "america", "john",
    "work", "officer", "agent", "whether", "within", "action", "relief", "set", "even", "new", "respect", "show",
    "fully", "since", "sec", "committee", "house", "cong", "cong_sess", "app",
    "hawaiian", "palmyra", "wilkinson", "bent", "king", "trademark", "ineffective assistance",
    "victim", "interview", "user", "yes", "okay", "talk", "designate_publication_yet_unpublished", "lien"
}
stop_words = base_stop | legal_stop

nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
nlp.max_length = 10_000_000  # Increased for high memory system

def clean_boilerplate(text):
    text = re.sub(r"No\.?\s*\d{1,6}(?:[-–]\d{1,6})?", "", text)
    text = re.sub(r"U\.?S\.?\s*\d+", "", text)
    text = re.sub(r"(?si)^.*?(JOINT\s+PETITION\s+FOR\s+A\s+WRIT\s+OF\s+CERTIORARI)", "", text)
    return text

def pre_tokenize(text, min_len=3):
    if len(text) > 5_000_000:  # Increased limit for high memory
        text = text[:5_000_000]
    t = clean_boilerplate(text)
    t = re.sub(r"[^A-Za-z]", " ", t).lower()
    toks = simple_preprocess(t, deacc=True, min_len=min_len)
    return [w for w in toks if w not in stop_words]

PARQUET_PATH = "/media/f1_drive/CorrelationCode/all_years_petitions_text_noIFP.parquet"
TEXT_COL     = "text"

print("Loading all documents into memory...")
df_original = pd.read_parquet(PARQUET_PATH)
df_original = df_original.dropna(subset=[TEXT_COL]).reset_index(drop=True)
print(f"Loaded {len(df_original)} documents with non-null text")

# tracking original indices
df_original['original_index'] = df_original.index
texts = df_original[TEXT_COL].astype(str).tolist()

print("Pre-tokenizing (removing stopwords, cleaning)...")
pre_tokenized = [pre_tokenize(text) for text in tqdm(texts)]

# 1: remove docs with < 5 tokens after pre-tokenization
print("Filtering after pre-tokenization...")
valid_indices_1 = [i for i, doc in enumerate(pre_tokenized) if len(doc) >= 5]
pre_tokenized = [pre_tokenized[i] for i in valid_indices_1]
texts = [texts[i] for i in valid_indices_1]
df_working = df_original.iloc[valid_indices_1].copy().reset_index(drop=True)
print(f"Retained {len(df_working)} documents after pre-tokenization filtering")

print("Lemmatizing with spaCy using multiprocessing.Pool...")
docs_for_spacy = [" ".join(doc) for doc in pre_tokenized]

def init_worker():
    """Initialize spaCy model in each worker process"""
    global worker_nlp
    import spacy
    worker_nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
    worker_nlp.max_length = 10_000_000  # Match the increased limit

def lemmatize_doc_worker(doc):
    """Worker function that uses the process-local spaCy model"""
    spacy_doc = worker_nlp(doc)
    return [tok.lemma_ for tok in spacy_doc if tok.lemma_ not in stop_words]

#new optimized multiprocessing approach
with multiprocessing.Pool(
    processes=multiprocessing.cpu_count(),  
    initializer=init_worker
) as pool:
    documents = list(tqdm(
        pool.imap(lemmatize_doc_worker, docs_for_spacy, chunksize=200),  
        total=len(docs_for_spacy),
        desc="Lemmatizing"
    ))

#2: remove docs with < 5 tokens after lemmatization
print("Filtering after lemmatization...")
valid_indices_2 = [i for i, doc in enumerate(documents) if len(doc) >= 5]
documents = [documents[i] for i in valid_indices_2]
texts = [texts[i] for i in valid_indices_2]
df_working = df_working.iloc[valid_indices_2].copy().reset_index(drop=True)
print(f"Retained {len(df_working)} documents after lemmatization filtering")

# Build bigrams and trigrams 
print("Building bigrams and trigrams...")
bigram  = Phrases(documents, min_count=50, threshold=10)
bigram_mod = Phraser(bigram)
trigram = Phrases(bigram[documents], min_count=25, threshold=10)
trigram_mod = Phraser(trigram)
documents = [trigram_mod[bigram_mod[doc]] for doc in documents]

# 3: remove docs with < 5 tokens after n-gram processing
print("Filtering after n-gram processing...")
valid_indices_3 = [i for i, doc in enumerate(documents) if len(doc) >= 5]
documents = [documents[i] for i in valid_indices_3]
texts = [texts[i] for i in valid_indices_3]
df_final = df_working.iloc[valid_indices_3].copy().reset_index(drop=True)
print(f"Final dataset: {len(df_final)} documents")

# alignment check
print(f"Documents list length: {len(documents)}")
print(f"Texts list length: {len(texts)}")
print(f"DataFrame length: {len(df_final)}")
assert len(documents) == len(texts) == len(df_final), "Alignment error!"

#dictionary and corpus
print("Building dictionary and corpus...")
dictionary = Dictionary(documents)
dictionary.filter_extremes(no_below=20, no_above=0.3)
corpus = [dictionary.doc2bow(doc) for doc in documents]
print(f"Dictionary size after filtering: {len(dictionary)} tokens")

# save dictionary and corpus
dictionary.save("data/dictionary.gensim")
print("Saved dictionary to data/dictionary.gensim")
with open("data/corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)
print("Saved corpus to data/corpus.pkl")


topic_nums = [14, 15, 16, 17, 18, 19, 20]
passes     = 20 
eval_every = 1
workers    = multiprocessing.cpu_count()  

lda_models = {}
topic_assignments = {}

def get_dominant_topic(lda_model, corpus):
    """Get dominant topic for each document"""
    dominant_topics = []
    topic_probs_list = []
    
    for bow in corpus:
        topic_probs = lda_model.get_document_topics(bow)
        if topic_probs:
            dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
            max_prob = max(topic_probs, key=lambda x: x[1])[1]
        else:
            dominant_topic = -1
            max_prob = 0.0
            
        dominant_topics.append(dominant_topic)
        topic_probs_list.append(max_prob)
    
    return dominant_topics, topic_probs_list

for k in topic_nums:
    print(f"\nTraining LdaMulticore with k={k} topics…")
    lda = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        passes=passes,
        eval_every=eval_every,
        workers=workers,
        random_state=42,
        alpha='asymmetric',
        eta='auto'
    )
    lda_models[k] = lda

    # Save model
    lda.save(f"data/lda_model_k{k}.gensim")
    print(f"Saved LDA model to data/lda_model_k{k}.gensim")

    # Save top words
    with open(f"data/top_words_k{k}.txt", "w", encoding="utf-8") as f:
        for idx, topic in lda.show_topics(num_topics=k, num_words=10, formatted=False):
            line = f"Topic {idx}: {', '.join([w for w,_ in topic])}\n"
            print(line.strip())
            f.write(line)
    print(f"Saved top words per topic to data/top_words_k{k}.txt")

    #pyLDAvis 
    try:
        vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(vis, f"data/pyldavis_k{k}.html")
        print(f"Saved pyLDAvis visualization to data/pyldavis_k{k}.html")
    except Exception as e:
        print(f"pyLDAvis failed for k={k}: {e}")

    #topic assignment probabilities
    topics, probs = get_dominant_topic(lda, corpus)
    topic_assignments[k] = topics
    
    #topic assignments to main df (for comparison across k values)
    df_final[f'topic_k{k}'] = topics
    df_final[f'topic_k{k}_prob'] = probs
    
    # Create and save individual k DataFrame
    base_columns = [col for col in df_final.columns if not col.startswith('topic_')]
    df_individual = df_final[base_columns].copy()
    df_individual['topic'] = topics
    df_individual['topic_prob'] = probs
    
    # Save individual parquet
    df_individual.to_parquet(f'data/petitions_k{k}.parquet', index=False)
    print(f"Saved individual parquet for k={k} to data/petitions_k{k}.parquet")

    #some examples for verification
    print(f"Sample topic assignments for k={k}:")
    for i in range(min(5, len(df_final))):
        print(f"  Doc {i} (orig_idx {df_final.iloc[i]['original_index']}): Topic {topics[i]} (prob: {probs[i]:.3f})")

#verify
print(f"\nFinal verification:")
print(f"Main DataFrame shape: {df_final.shape}")
print(f"Number of unique original indices: {df_final['original_index'].nunique()}")
print(f"Topic assignment columns: {[col for col in df_final.columns if col.startswith('topic_')]}")

#parquet save (main df with all k topic assignments)
df_final.to_parquet(f'data/petitions_with_topics_allk.parquet', index=False)
print(f"Saved main parquet with all topic assignments to data/petitions_with_topics_allk.parquet")

#mapping indices to csv
index_mapping = df_final[['original_index']].copy()
index_mapping['final_index'] = index_mapping.index
index_mapping.to_csv('data/index_mapping.csv', index=False)
print("Saved index mapping to data/index_mapping.csv")

print("\nOUTPUT FILES:")
print("Model files:")
for k in topic_nums:
    print(f"  - data/lda_model_k{k}.gensim")
    print(f"  - data/petitions_k{k}.parquet")
    print(f"  - data/top_words_k{k}.txt")
    print(f"  - data/pyldavis_k{k}.html")
print("  - data/petitions_with_topics_allk.parquet")
print("  - data/dictionary.gensim")
print("  - data/corpus.pkl")
print("  - data/index_mapping.csv")

print("\nen of pipeline")