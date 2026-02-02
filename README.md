# MS MARCO Semantic Search Baseline 🚀

**Dense Retrieval Implementation**

[![Status](https://img.shields.io/badge/Status-✅%20Complete-green)]() [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]() [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)]()

##  **Project Structure**
msmarco-semantic-search/
├── 01_baseline.py
├── 02_finetuning.py
├── requirements.txt # Dependencies
├── README.md # This file
└── msmarco_data/ # Generated (gitignored)
├── baseline_model/
├── finetuned_model/
├── data.pkl
└── faiss.index

## 🚀 **Quick Start (Windows)**

### **1. Fix Windows UTF-8 (Admin CMD - ONCE)**
```cmd
setx PYTHONUTF8 1
```
### **2. Installation and running**
```
pip install -r requirements.txt
python 01_baseline.py    # ~5min → MRR@10: 0.22
python 02_finetuning.py  # ~15min → MRR@10: 0.30
```
