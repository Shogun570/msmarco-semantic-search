# MS MARCO Semantic Search

**Dense Retrieval Implementation**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]() [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)]()


## *Quick Start (Windows)*

### **1. Fix Windows UTF-8 (Admin CMD - ONCE)**
```cmd
setx PYTHONUTF8 1
```
### **2. Installation and running**
```bash
# 1. Clone
git clone https://github.com/shogun570/msmarco-semantic-search.git
cd msmarco-semantic-search

# 2. Install (2min)
pip install -r requirements.txt

# 3. Run (5min first time)
python 01_baseline.py
python 02_finetuning.py
python 03_evaluation.py
python 04_iteration.py
```

### Failure/Edge cases
03_evaluation often does not reflect the same MRR as reported by 01_baseline. It also sometimes reports ∆=0 for one or more parameters after fine tuning. Due to time constraints for this project I couldn't figure out why.

Nevertheless I quickly learnt the basis for these models learn, MRR scores, FAISS indexing, BM25 retrieval and much more!

Made with ❤️ in BPHC
