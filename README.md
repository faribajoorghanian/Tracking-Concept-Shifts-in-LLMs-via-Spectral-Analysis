# Tracking-Concept-Shifts-in-LLMs-via-Spectral-Analysis

This repository contains a **Mechanistic Interpretability** research notebook that explores how Large Language Models (LLMs) adapt to new domains without catastrophic forgetting. By applying Singular Value Decomposition (SVD) and Logit Lens techniques, we decode the latent semantic representations of GPT-2 before and after LoRA fine-tuning.

## 📌 Project Overview
Instead of treating the LLM as a black box, this project mathematically analyzes the internal weight updates ($\Delta W$) and hidden state representations to answer: *Where and how does a model store new stylistic concepts while preserving its foundational knowledge?*

**All experiments, code, visualizations, and analyses are contained within the main Jupyter Notebook.**

### 📓 What's Inside the Notebook:
- **Phase 1: Domain Adaptation:** Parameter-Efficient Fine-Tuning (PEFT/LoRA) implementation on a journalistic dataset to induce a behavioral shift.
- **Phase 2: Zero-Shot Spectral Analysis:** Applying SVD on the `c_proj` MLP weights (Layer 11) to extract foundational semantic concepts.
- **Phase 3: Weight Update Distribution:** Statistical analysis of $\Delta W$ to empirically validate the low-rank bottleneck and absence of catastrophic forgetting.
- **Phase 4: Semantic Emergence via Logit Lens:** Projecting hidden states from all transformer layers into the vocabulary space to track real-time syntactic-to-semantic transitions.

## 🚀 How to Explore
Simply click on the **`.ipynb`** file in this repository to view the complete interactive report. The notebook includes all code cells, output logs, and generated charts (Weight Distribution and Logit Lens layer tracking) saved directly within the file.

## 💻 Tech Stack
- **Frameworks:** PyTorch, Hugging Face `transformers`, `peft`, `trl`, `datasets`
- **Analysis:** Singular Value Decomposition (`torch.svd`), Logit Lens
- **Visualization:** Matplotlib, Seaborn
