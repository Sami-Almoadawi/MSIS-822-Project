@@ -1,30 +1,22 @@
# 📌 Table of Contents
#  Table of Contents

- [Project Summary](#Project-Summary)
- [Task Description](#task-description)
- [Dataset Description](#dataset-Description)
- [System Architecture](#system-architecture)
- [Methodology](#Methodology)
- [Model Cards](#model-cards)
- [Training](#training)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results & Analysis](#Results-&-Analysis)
- [File Structure](#file-structure)
- [References](#References)


---

# 📌 Project Summary
#  Project Summary

This repository includes the code and tools for distinguishing between **AI-generated and human-written Arabic literature** with a hybrid approach that integrates stylistic features and deep semantic embeddings.  
The system incorporates sophisticated **Arabic preprocessing**, **sentence-transformer embeddings**, and a collection of **machine learning/deep learning classifiers** for enhanced performance.

---

# 🎯 Task Description
#  Task Description

This assignment is defined as a **binary classification problem**:

@@ -33,50 +30,50 @@

---

# 📊 Dataset Description
#  Dataset Description

We use a large dataset of **41,940 Arabic research abstracts**, composed of:

- **original_abstract**: The original human-written Arabic abstract
- **{model}_generated_abstract** Machine-generated version from each model

### Dataset Statistics

| Subset | Count | Ratio |
|-------|--------|--------|
| Training | 29,358 | 70% |
| Validation | 6,291 | 15% |
| Testing | 6,291 | 15% |
| Total Samples | 41,940 | 100% |
| Classes | 0 (AI), 1 (Human) |

Each entry contains:

- `abstract_text`  
- `generated_by`  
- `source_split`  
- `label`  

The dataset is **balanced**, guaranteeing consistent model performance.

---
##  Traditional Machine Learning Results

| **Model**    | **Accuracy** | **Precision** | **Recall(Human)** | **F1-Score(AI)** |
|----------------------|------------- -|---------------|------------|---------------|
| Logistic Regression  | 0.8609        | 0.8619        | 0.0223       | 0.9251        |
| SVM                  | 0.8614        | 0.8614          | 0.0       | 0.9255          |
| Random Forest        | **0.8934**    | **0.8946**      | **0.3600** | **0.8845**      |
| XGBoost              | 0.8614        | 0.8615          | 0.0011     | 0.9255          |
| Naive Bayes          | 0.8617        | 0.8607          | 0.0045       | 0.9256          |

##  Deep Learning Results

| **Model**                     | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|------------------------------|--------------|---------------|------------|---------------|
| Feedforward NN + BERT (768D) | 0.8617        | 0.9437          | 0.8713  | 0.9257          |

