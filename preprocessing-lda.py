#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Final topic model for SCOTUS petitions 
import pickle
import pyLDAvis
import pyLDAvis.gensim_models
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
import concurrent.futures
from tqdm import tqdm
from collections import Counter
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed

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
    "cee", "cece", "een", "supp", "title", "cir", "inc", "fed", "corp"

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
nlp.max_length = 5_000_000

def clean_boilerplate(text):
    text = re.sub(r"No\.?\s*\d{1,6}(?:[-–]\d{1,6})?", "", text)
    text = re.sub(r"U\.?S\.?\s*\d+", "", text)
    text = re.sub(r"(?si)^.*?(JOINT\s+PETITION\s+FOR\s+A\s+WRIT\s+OF\s+CERTIORARI)", "", text)
    return text

def pre_tokenize(text, min_len=3):
    if len(text) > 2_000_000:
        text = text[:2_000_000]
    t = clean_boilerplate(text)
    t = re.sub(r"[^A-Za-z]", " ", t).lower()
    toks = simple_preprocess(t, deacc=True, min_len=min_len)
    return [w for w in toks if w not in stop_words]

PARQUET_PATH = "data/raw_petitions_text.parquet"
TEXT_COL     = "text"

print("Loading all documents into memory...")
df = pd.read_parquet(PARQUET_PATH)
texts = df[TEXT_COL].dropna().astype(str).tolist()

print("Pre-tokenizing (removing stopwords, cleaning)...")
pre_tokenized = [pre_tokenize(text) for text in tqdm(texts)]

nonempty = [i for i, doc in enumerate(pre_tokenized) if len(doc) >= 5]
pre_tokenized = [pre_tokenized[i] for i in nonempty]
texts = [texts[i] for i in nonempty]
df = df.iloc[nonempty].copy()

print("Lemmatizing with spaCy in parallel using joblib...")
docs_for_spacy = [" ".join(doc) for doc in pre_tokenized]

def lemmatize_doc(doc):
    spacy_doc = nlp(doc)
    return [tok.lemma_ for tok in spacy_doc if tok.lemma_ not in stop_words]

documents = Parallel(n_jobs=-1)(
    delayed(lemmatize_doc)(doc) for doc in tqdm(docs_for_spacy, desc="Lemmatizing")
)

# Remove empty docs after lemmatization
nonempty = [i for i, doc in enumerate(documents) if len(doc) >= 5]
documents = [documents[i] for i in nonempty]
texts = [texts[i] for i in nonempty]
df = df.iloc[nonempty].copy()
print(f"Collected {len(documents)} documents after spaCy lemmatization")

# Build bigrams and trigrams 
bigram  = Phrases(documents, min_count=50, threshold=10)
bigram_mod = Phraser(bigram)
trigram = Phrases(bigram[documents], min_count=25, threshold=10)
trigram_mod = Phraser(trigram)
documents = [trigram_mod[bigram_mod[doc]] for doc in documents]

# Remove empty docs after n-gram step
nonempty = [i for i, doc in enumerate(documents) if len(doc) >= 5]
documents = [documents[i] for i in nonempty]
texts = [texts[i] for i in nonempty]
df = df.iloc[nonempty].copy()

# Rebuild dictionary and corpus AFTER stopword filtering

dictionary = Dictionary(documents)
dictionary.filter_extremes(no_below=20, no_above=0.3)
corpus     = [dictionary.doc2bow(doc) for doc in documents]
print(f"Dictionary size after filtering: {len(dictionary)} tokens")
dictionary.save("data/dictionary.gensim")
print("Saved dictionary to data/dictionary.gensim")
with open("data/corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)
print("Saved corpus to data/corpus.pkl")