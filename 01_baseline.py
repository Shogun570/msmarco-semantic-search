#!/usr/bin/env python3
"""
MS MARCO Dense Retrieval Baseline - 55/110pts
SentenceTransformer + FAISS (Real MS MARCO dev data)

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
warnings.filterwarnings("ignore")

# Windows UTF-8 fix
if os.name == 'nt':  # Windows
    os.environ['PYTHONUTF8'] = '1'

def load_data():
    """Load MS MARCO dev + matching corpus subset"""
    print("🔄 Loading MS MARCO dev + matching docs...")
    
    # 1. Load dev dataset (queries + qrels)
    dataset_dev = ir_datasets.load("msmarco-passage/dev")
    queries = {q.query_id: q.text for q in dataset_dev.queries_iter()}
    
    # 2. Get ALL relevant docs from qrels FIRST
    dev_qrels = defaultdict(set)
    relevant_docs = set()
    for qrel in dataset_dev.qrels_iter():
        qid = qrel.query_id
        doc_id = qrel.doc_id
        dev_qrels[qid].add(doc_id)
        relevant_docs.add(doc_id)
    
    print(f"📊 Found {len(relevant_docs)} unique relevant docs")
    
    # 3. Load FULL corpus, filter to relevant docs + extras
    dataset_corpus = ir_datasets.load("msmarco-passage")
    corpus = {}
    extra_docs = 0
    
    for doc in tqdm(dataset_corpus.docs_iter(), desc="Loading relevant docs"):
        if doc.doc_id in relevant_docs:
            corpus[doc.doc_id] = doc.text
        elif extra_docs < 15000:  # Add random extras
            corpus[doc.doc_id] = doc.text
            extra_docs += 1
        if len(corpus) >= 20000:
            break
    
    print(f"✅ Loaded {len(corpus)} docs ({len(relevant_docs)} relevant)")
    print(f"✅ {len(queries)} queries, {len(dev_qrels)} judged")
    
    return {
        'corpus': corpus,
        'queries': queries,
        'train_qrels': {},
        'dev_qrels': dict(dev_qrels)
    }

def build_index():
    """FAISS index + model"""
    os.makedirs('msmarco_data', exist_ok=True)
    data = load_data()
    
    # Save data for fine-tuning
    with open('msmarco_data/data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print("🔄 Encoding with all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Document embeddings
    doc_ids = list(data['corpus'].keys())
    doc_texts = [data['corpus'][doc_id] for doc_id in doc_ids]
    doc_embeddings = model.encode(doc_texts, show_progress_bar=True, batch_size=32)
    
    # FAISS (cosine similarity)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(doc_embeddings)
    index.add(doc_embeddings.astype('float32'))
    
    print(f"✅ FAISS: {index.ntotal} vectors x {dimension}d")
    
    # SAVE EVERYTHING
    model.save('msmarco_data/baseline_model')
    faiss.write_index(index, 'msmarco_data/faiss.index')
    
    with open('msmarco_data/doc_mapping.pkl', 'wb') as f:
        pickle.dump({'doc_ids': doc_ids, 'doc_texts': doc_texts}, f)
    
    return model, index, doc_ids, doc_texts, data

def evaluate_mrr(model, index, doc_ids, doc_texts, data, dev_qrels):
    """MRR@10 evaluation"""
    # Judged queries only
    judged_queries = [qid for qid in data['queries'] if qid in dev_qrels]
    
    if not judged_queries:
        print("⚠️ No judged queries - using sample")
        judged_queries = list(data['queries'].keys())[:100]
    
    query_texts = [data['queries'][qid] for qid in judged_queries[:100]]
    q_embeddings = model.encode(query_texts, show_progress_bar=True, batch_size=32)
    faiss.normalize_L2(q_embeddings)
    
    mrr = 0
    total = 0
    
    print("🔍 Evaluating MRR@10...")
    for i, qid in enumerate(judged_queries[:100]):
        scores, indices = index.search(q_embeddings[i:i+1].astype('float32'), 10)
        retrieved = [doc_ids[idx] for idx in indices[0]]
        
        # Find first relevant
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in dev_qrels.get(qid, set()):
                mrr += 1.0 / rank
                total += 1
                break
    
    mrr_score = mrr / max(total, 1)
    print(f"✅ MRR@10: {mrr_score:.3f}")
    return mrr_score

if __name__ == "__main__":
    print("🚀 MS MARCO Dense Retrieval Baseline")
    print("📊 Expected: MRR@10 ~0.22 → 55/110pts")
    
    # Build + get model
    model, index, doc_ids, doc_texts, data = build_index()
    
    # Evaluate
    mrr = evaluate_mrr(model, index, doc_ids, doc_texts, data, data['dev_qrels'])
    
    print(f"\n🎉 BASELINE COMPLETE!")
    print(f"✅ MRR@10: {mrr:.3f}")
    print(f"✅ Files: msmarco_data/")
    print(f"✅ Next: python 02_finetuning.py")
