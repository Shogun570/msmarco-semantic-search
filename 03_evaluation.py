#!/usr/bin/env python3
"""
Real MS MARCO Evaluation - Complete Metrics Suite
Baseline vs Fine-tuned with ALL standard IR metrics
"""

import os
import pickle
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import psutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import statistics

@dataclass
class Metrics:
    mrr: float = 0.0
    ndcg1: float = 0.0
    ndcg5: float = 0.0
    ndcg10: float = 0.0
    ndcg20: float = 0.0
    recall1: float = 0.0
    recall5: float = 0.0
    recall10: float = 0.0
    recall100: float = 0.0
    precision1: float = 0.0
    precision5: float = 0.0
    precision10: float = 0.0

def load_models():
    """Load all MS MARCO data and models"""
    data = pickle.load(open('msmarco_data/data.pkl', 'rb'))
    doc_mapping = pickle.load(open('msmarco_data/doc_mapping.pkl', 'rb'))
    
    baseline_model = SentenceTransformer('msmarco_data/baseline_model')
    finetuned_model = SentenceTransformer('msmarco_data/finetuned_model')
    index = faiss.read_index('msmarco_data/faiss.index')
    
    return data, doc_mapping, baseline_model, finetuned_model, index

def compute_metrics(qrels: Dict[str, set], run: Dict[str, Dict[str, float]], 
                   k_values: List[int] = [1, 5, 10, 20, 100]) -> Metrics:
    """Complete IR metrics suite for MS MARCO"""
    total_queries = len([q for q in run if q in qrels])
    mrr_total, ndcg_total = 0.0, defaultdict(float)
    recall_total, prec_total = defaultdict(float), defaultdict(float)
    count_queries = 0
    
    for qid, ranking in run.items():
        if qid not in qrels:
            continue
            
        relevant_docs = qrels[qid]
        ranked_docs = list(ranking.keys())
        scores = list(ranking.values())
        
        # MRR@10
        for rank, doc_id in enumerate(ranked_docs[:10], 1):
            if doc_id in relevant_docs:
                mrr_total += 1.0 / rank
                break
        
        # NDCG@k
        max_relevant = min(len(relevant_docs), max(k_values))
        ideal_dcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, max_relevant + 1))
        
        for k in k_values:
            dcg = 0.0
            for rank in range(min(k, len(ranked_docs))):
                doc_id = ranked_docs[rank]
                rel = 1.0 if doc_id in relevant_docs else 0.0
                dcg += rel / np.log2(rank + 2)
            ndcg_total[k] += dcg / max(ideal_dcg, 1e-8)
        
        # Recall@k
        for k in k_values:
            hits = len(set(ranked_docs[:k]) & relevant_docs)
            total_relevant = len(relevant_docs)
            recall_total[k] += hits / max(total_relevant, 1)
        
        # Precision@k
        for k in k_values[:3]:  # P@1,5,10 only
            relevant_in_topk = len(set(ranked_docs[:k]) & relevant_docs)
            prec_total[k] += relevant_in_topk / k
        
        count_queries += 1
    
    metrics = Metrics(
        mrr=mrr_total / max(count_queries, 1),
        ndcg1=ndcg_total[1] / max(count_queries, 1),
        ndcg5=ndcg_total[5] / max(count_queries, 1),
        ndcg10=ndcg_total[10] / max(count_queries, 1),
        ndcg20=ndcg_total[20] / max(count_queries, 1),
        recall1=recall_total[1] / max(count_queries, 1),
        recall5=recall_total[5] / max(count_queries, 1),
        recall10=recall_total[10] / max(count_queries, 1),
        recall100=recall_total[100] / max(count_queries, 1),
        precision1=prec_total[1] / max(count_queries, 1),
        precision5=prec_total[5] / max(count_queries, 1),
        precision10=prec_total[10] / max(count_queries, 1)
    )
    return metrics

def benchmark_latency(model, index, test_queries, doc_ids, num_runs=1000):
    """Measure query latency percentiles"""
    latencies = []
    
    for _ in range(min(num_runs, len(test_queries))):
        start_time = time.perf_counter()
        
        qid, q_text = test_queries[_ % len(test_queries)]
        q_emb = model.encode([q_text])
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb.astype('float32'), 10)
        
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # ms
    
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies))]
    p99 = latencies[int(0.99 * len(latencies))]
    
    qps = len(latencies) / sum(latencies) * 1000  # queries per second
    
    return p50, p95, p99, qps

def print_results_table(baseline_metrics, finetuned_metrics, baseline_latency, finetuned_latency):
    """Professional results table"""
    print("\n" + "="*80)
    print("COMPLETE MS MARCO EVALUATION RESULTS")
    print("="*80)
    
    print("\nRETRIEVAL QUALITY METRICS")
    print("-" * 50)
    print(f"{'Metric':<12} {'Baseline':<10} {'Fine-tuned':<12} {'Δ (%)':<8}")
    print("-" * 50)
    print(f"{'MRR@10':<12} {baseline_metrics.mrr:<10.4f} {finetuned_metrics.mrr:<12.4f} "
          f"{((finetuned_metrics.mrr/baseline_metrics.mrr-1)*100):+6.1f}%")
    print(f"{'NDCG@1':<12} {baseline_metrics.ndcg1:<10.4f} {finetuned_metrics.ndcg1:<12.4f} "
          f"{((finetuned_metrics.ndcg1/baseline_metrics.ndcg1-1)*100):+6.1f}%")
    print(f"{'NDCG@5':<12} {baseline_metrics.ndcg5:<10.4f} {finetuned_metrics.ndcg5:<12.4f} "
          f"{((finetuned_metrics.ndcg5/baseline_metrics.ndcg5-1)*100):+6.1f}%")
    print(f"{'NDCG@10':<12} {baseline_metrics.ndcg10:<10.4f} {finetuned_metrics.ndcg10:<12.4f} "
          f"{((finetuned_metrics.ndcg10/baseline_metrics.ndcg10-1)*100):+6.1f}%")
    print(f"{'NDCG@20':<12} {baseline_metrics.ndcg20:<10.4f} {finetuned_metrics.ndcg20:<12.4f} "
          f"{((finetuned_metrics.ndcg20/baseline_metrics.ndcg20-1)*100):+6.1f}%")
    print(f"{'R@1':<12} {baseline_metrics.recall1:<10.4f} {finetuned_metrics.recall1:<12.4f} "
          f"{((finetuned_metrics.recall1/baseline_metrics.recall1-1)*100):+6.1f}%")
    print(f"{'R@5':<12} {baseline_metrics.recall5:<10.4f} {finetuned_metrics.recall5:<12.4f} "
          f"{((finetuned_metrics.recall5/baseline_metrics.recall5-1)*100):+6.1f}%")
    print(f"{'R@10':<12} {baseline_metrics.recall10:<10.4f} {finetuned_metrics.recall10:<12.4f} "
          f"{((finetuned_metrics.recall10/baseline_metrics.recall10-1)*100):+6.1f}%")
    print(f"{'R@100':<12} {baseline_metrics.recall100:<10.4f} {finetuned_metrics.recall100:<12.4f} "
          f"{((finetuned_metrics.recall100/baseline_metrics.recall100-1)*100):+6.1f}%")
    print(f"{'P@1':<12} {baseline_metrics.precision1:<10.4f} {finetuned_metrics.precision1:<12.4f} "
          f"{((finetuned_metrics.precision1/baseline_metrics.precision1-1)*100):+6.1f}%")
    print(f"{'P@5':<12} {baseline_metrics.precision5:<10.4f} {finetuned_metrics.precision5:<12.4f} "
          f"{((finetuned_metrics.precision5/baseline_metrics.precision5-1)*100):+6.1f}%")
    print(f"{'P@10':<12} {baseline_metrics.precision10:<10.4f} {finetuned_metrics.precision10:<12.4f} "
          f"{((finetuned_metrics.precision10/baseline_metrics.precision10-1)*100):+6.1f}%")
    
    print("\nEFFICIENCY METRICS")
    print("-" * 50)
    print(f"{'Metric':<20} {'Baseline':<12} {'Fine-tuned':<12}")
    print("-" * 50)
    print(f"{'Latency p50 (ms)':<20} {baseline_latency[0]:<12.1f} {finetuned_latency[0]:<12.1f}")
    print(f"{'Latency p95 (ms)':<20} {baseline_latency[1]:<12.1f} {finetuned_latency[1]:<12.1f}")
    print(f"{'Latency p99 (ms)':<20} {baseline_latency[2]:<12.1f} {finetuned_latency[2]:<12.1f}")
    print(f"{'QPS':<20} {baseline_latency[3]:<12.1f} {finetuned_latency[3]:<12.1f}")
    
    # Memory & Storage
    mem = psutil.Process().memory_info().rss / 1e6
    index_size = os.path.getsize('msmarco_data/faiss.index') / 1e6
    print(f"\n{'Peak RAM':<20} {mem:<12.1f} MB")
    print(f"{'Index Size':<20} {index_size:<12.1f} MB")
    print("="*80)

if __name__ == "__main__":
    print("Real MS MARCO Complete Evaluation Suite")
    print("Metrics: MRR, NDCG@1-20, Recall@1-100, Precision@1-10")
    
    # Load data
    data, doc_mapping, baseline_model, finetuned_model, index = load_models()
    doc_ids = doc_mapping['doc_ids']
    test_queries = list(data['queries'].items())[:1000]  # 1000 queries for latency
    qrels = data['dev_qrels']
    
    # Quality evaluation (first 50 queries for speed)
    eval_queries = test_queries[:50]
    
    print("Evaluating baseline quality...")
    baseline_run = {}
    for qid, q_text in tqdm(eval_queries, desc="Baseline"):
        q_emb = baseline_model.encode([q_text])
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb.astype('float32'), 100)  # R@100 needs top-100
        baseline_run[qid] = {doc_ids[idx]: float(score) 
                           for idx, score in zip(indices[0], scores[0])}
    
    print("Evaluating fine-tuned quality...")
    finetuned_run = {}
    for qid, q_text in tqdm(eval_queries, desc="Fine-tuned"):
        q_emb = finetuned_model.encode([q_text])
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb.astype('float32'), 100)
        finetuned_run[qid] = {doc_ids[idx]: float(score) 
                            for idx, score in zip(indices[0], scores[0])}
    
    # Compute all metrics
    baseline_metrics = compute_metrics(qrels, baseline_run)
    finetuned_metrics = compute_metrics(qrels, finetuned_run)
    
    # Latency benchmarking (1000 queries)
    print("Benchmarking latency (1000 queries)...")
    baseline_latency = benchmark_latency(baseline_model, index, test_queries, doc_ids)
    finetuned_latency = benchmark_latency(finetuned_model, index, test_queries, doc_ids)
    
    # Results
    print_results_table(baseline_metrics, finetuned_metrics, 
                       baseline_latency, finetuned_latency)
    
    sys.exit(0)
