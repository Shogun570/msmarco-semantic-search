#!/usr/bin/env python3
"""
MS MARCO Dense Retrieval Baseline - 33/110pts
SentenceTransformer + FAISS on real MS MARCO data

Run: pip install -r requirements.txt && python 01_baseline.py
"""

import os
import ir_datasets
import torch
import numpy as np
import pickle
from collections import defaultdict
from itertools import islice
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Windows UTF-8 fix
os.environ['PYTHONUTF8'] = '1'

def load_data():
    """Load real MS MARCO passage data (UTF-8 safe)"""
    print("🔄 Loading MS MARCO passage data...")
    
    # Use DEV subset first (guaranteed clean)
    dataset = ir_datasets.load("msmarco-passage/dev")
    
    # Load corpus (20K docs for speed)
    corpus = {}
    for doc in tqdm(islice(dataset.docs_iter(), 20000), total=20000, desc="Docs"):
        corpus[doc.doc_id] = doc.text
    
    # Load queries
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    
    # Load qrels
    train_qrels = defaultdict(set)
    dev_qrels = defaultdict(set)
    
    # Dev qrels
    for qrel in dataset.qrels_iter():
        dev_qrels[qrel.query_id]
