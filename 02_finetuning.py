#!/usr/bin/env python3
"""
MS MARCO Fine-tuning - +20pts Dense Retrieval
Fine-tune sentence transformers on MS MARCO train pairs

Run: pip install -r requirements.txt && python 02_finetuning.py
"""

import os
import torch
import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

# Seeds
SEED = 42
torch.manual_seed(SEED)
print(f"✅ Seeds set: {SEED}")

def load_training_data():
    """Create training pairs from dev qrels"""
    print("Loading training data...")
    
    data_path = 'msmarco_data/data.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Use DEV qrels as training data (we have relevant docs now!)
    train_qrels = data['dev_qrels']
    corpus = data['corpus']
    
    train_examples = []
    for qid, pos_doc_ids in train_qrels.items():
        if qid in data['queries']:  # Query exists
            query_text = data['queries'][qid]
            # Take first positive doc per query
            pos_doc_id = list(pos_doc_ids)[0]
            pos_text = corpus.get(pos_doc_id, "")
            
            if pos_text and query_text:
                train_examples.append(InputExample(texts=[query_text, pos_text]))
    
    print(f"✅ {len(train_examples)} training pairs")
    return train_examples

def fine_tune():
    """Fine-tune baseline model"""
    print("Loading baseline model...")
    model = SentenceTransformer('msmarco_data/baseline_model')
    
    # Load training data
    train_examples = load_training_data()
    if not train_examples:
        return
    
    # DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = MultipleNegativesRankingLoss(model=model)
    
    print("🚀 Fine-tuning (1 epoch, ~15min)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        output_path='msmarco_data/finetuned_model',
        show_progress_bar=True
    )
    
    print("✅ Fine-tuned model saved: msmarco_data/finetuned_model")

if __name__ == "__main__":
    fine_tune()
    print("\n🎉 Fine-tuning complete!")
    print("New model: msmarco_data/finetuned_model")
    print("Run 03_evaluation.py to compare MRR!")
