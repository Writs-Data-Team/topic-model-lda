#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#final topic model for SCOTUS petitions with topic assignment and distribution graph
import pyLDAvis
import pyLDAvis.gensim_models
import re
import pyarrow.parquet as pq
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
import pyarrow as pa

nltk.download('stopwords', quiet=True)
base_stop = set(stopwords.words('english'))

#mega jargon list
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
    "hawaiian", "palmyra", "wilkinson", "bent", "king"

}

stop_words = base_stop | legal_stop
nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])

def clean_boilerplate(text):
    text = re.sub(r"No\\.\\s*\\d{1,6}(?:[-–]\\d{1,6})?", "", text)
    text = re.sub(r"U\\.?S\\.?\\s*\\d+", "", text)
    text = re.sub(r"(?si)^.*?(JOINT\\s+PETITION\\s+FOR\\s+A\\s+WRIT\\s+OF\\s+CERTIORARI)", "", text)
    return text

def preprocess(text, min_len=3):
    t = clean_boilerplate(text)
    t = re.sub(r"[^A-Za-z]", " ", t).lower()
    toks = simple_preprocess(t, deacc=True, min_len=min_len)
    toks = [w for w in toks if w not in stop_words]
    doc = nlp(" ".join(toks))
    return [tok.lemma_ for tok in doc if tok.lemma_ not in stop_words]

def stream_parquet_docs(path, column='text', batch_size=10000, min_words=100):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        for txt in df[column].dropna().astype(str):
            buf, count = [], 0
            for w in txt.split():
                buf.append(w); count += 1
                if count >= min_words:
                    yield " ".join(buf)
                    buf, count = [], 0
            if buf:
                yield " ".join(buf)

PARQUET_PATH = "data/raw_petitions_text.parquet"
TEXT_COL     = "text"
documents, text_data = [], []

pf = pq.ParquetFile(PARQUET_PATH)
total_rows = sum(batch.num_rows for batch in pf.iter_batches())

for chunk in tqdm(stream_parquet_docs(PARQUET_PATH, column=TEXT_COL), total=total_rows):
    toks = preprocess(chunk)
    if len(toks) >= 5:
        documents.append(toks)
        text_data.append(chunk)

print(f"Collected {len(documents)} documents after initial preprocessing")

freq = Counter(token for doc in documents for token in doc)
top_n = {tok for tok, cnt in freq.most_common(50)}
print("Top‐50 tokens to drop:", top_n)
stop_words |= top_n
documents = [preprocess(" ".join(doc)) for doc in documents]
documents = [doc for doc in documents if len(doc) >= 5]
print(f"{len(documents)} docs remain after filtering top‐freq tokens")

bigram  = Phrases(documents, min_count=50, threshold=10)
bigram_mod = Phraser(bigram)
trigram = Phrases(bigram[documents], min_count=25, threshold=10)
trigram_mod = Phraser(trigram)
documents = [trigram_mod[bigram_mod[doc]] for doc in documents]

dictionary = Dictionary(documents)
dictionary.filter_extremes(no_below=20, no_above=0.3)
corpus = [dictionary.doc2bow(doc) for doc in documents]
print(f"Dictionary size after filtering: {len(dictionary)} tokens")

topic_nums = [14,17,20]
passes = 15
eval_every = 1
workers = max(1, multiprocessing.cpu_count() - 1)
coherence_scores = {}

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
    cm = CoherenceModel(
        model=lda,
        texts=documents,
        dictionary=dictionary,
        coherence='c_v'
    )
    score = cm.get_coherence()
    coherence_scores[k] = score
    print(f"Coherence (c_v) = {score:.4f}")

plt.figure(figsize=(6,4))
plt.plot(list(coherence_scores.keys()), list(coherence_scores.values()), marker='o')
plt.title("Coherence Score vs Number of Topics")
plt.xlabel("Topics")
plt.ylabel("Coherence (c_v)")
plt.grid(True)
plt.show()

gen_k = 14
print(f"\nrequired: k={gen_k} topics. Top words per topic:")
gen_lda = LdaMulticore(
    corpus=corpus,
    id2word=dictionary,
    num_topics=gen_k,
    passes=passes,
    eval_every=eval_every,
    workers=workers,
    random_state=42,
    alpha='asymmetric',
    eta='auto'
)
for idx, topic in gen_lda.show_topics(num_topics=gen_k, num_words=10, formatted=False):
    print(f"Topic {idx}: {', '.join([w for w,_ in topic])}")

topic_assignments = []
for doc_bow in corpus:
    topic_probs = gen_lda.get_document_topics(doc_bow)
    dominant_topic = max(topic_probs, key=lambda x: x[1])[0] if topic_probs else -1
    topic_assignments.append(dominant_topic)

df_out = pd.DataFrame({"text": text_data, "topic": topic_assignments})
output_path = "data/clustered_petitions.parquet"
table = pa.Table.from_pandas(df_out)
pq.write_table(table, output_path)
print(f"\nSaved topic assignments to {output_path}")

topic_counts = Counter(topic_assignments)
plt.figure(figsize=(8,6))
plt.bar(topic_counts.keys(), topic_counts.values())
plt.xlabel("Topic Number")
plt.ylabel("Number of Documents")
plt.title("Distribution of Documents Across Topics")
plt.grid(axis='y')
plt.show()

