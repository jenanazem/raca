# RACA Legal LLM — Fine-tuning + SAE Pipeline

## Project structure

```
raca_llm_project/
├── phase1_data_prep.py       # Data cleaning, synthetic Q&A via OpenRouter, dataset export
├── phase2_model_setup.py     # GPU check, model download, project config
├── phase3_finetune.py        # QLoRA fine-tuning (CPT stage → SFT stage)
├── phase4_sae_training.py    # Sparse Autoencoder training on fine-tuned model
├── phase5_hal_reduction.py   # Hallucination reduction via activation steering + evaluation
├── requirements.txt          # All Python dependencies
│
├── data/
│   ├── raw_pdfs/             ← PUT YOUR 50 PDF FILES HERE
│   ├── raca_laws_tab1.csv    ← PUT YOUR CSV FILES HERE (raca_laws_tab*.csv)
│   └── processed/            (auto-created by Phase 1)
│
├── checkpoints/              (auto-created by Phase 3 & 4)
└── results/                  (auto-created by Phase 5)
```

## Where to place your data

1. CSV files (e.g. raca_laws_tab1.csv, raca_laws_tab2.csv, ...):
   → Place directly in raca_llm_project/
   → Phase 1 finds them via glob pattern raca_laws_tab*.csv

2. PDF files (your 50 source documents):
   → Place in raca_llm_project/data/raw_pdfs/
   → These are your source of truth for RAG in Phase 5

## Setup

```bash
# 1. Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# 2. Set your OpenRouter API key
export OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx

# 3. Run phases in order
python phase1_data_prep.py --input_glob "raca_laws_tab*.csv" --output_dir ./data/processed
python phase2_model_setup.py --skip_model_load
python phase3_finetune.py --config ./data/processed/project_config.json --data_dir ./data/processed --output_dir ./checkpoints/ft_raca
python phase4_sae_training.py --config ./data/processed/project_config.json --model_path ./checkpoints/ft_raca/merged --data_dir ./data/processed --output_dir ./checkpoints/sae_raca
python phase5_hal_reduction.py --config ./data/processed/project_config.json --model_path ./checkpoints/ft_raca/merged --sae_path ./checkpoints/sae_raca/sae_best.pt --feature_file ./checkpoints/sae_raca/feature_interpretations.json
```

## Manual step between Phase 4 and Phase 5

After Phase 4 finishes, open:
  checkpoints/sae_raca/feature_interpretations.json

For each feature entry, review the top_activating_docs snippets and fill in:
  "label": "what concept does this feature encode?"
  "is_hallucination_feature": true / false

This is required before running Phase 5.

## Hardware requirements

- Minimum: 1× GPU with 12GB VRAM (e.g. RTX 3080, T4)
- Recommended: 1× A100 40GB (Google Colab Pro+, Lambda Labs, RunPod)
- Phase 3 training time: ~2–4 hours on A100 for 2000 examples
- Phase 4 collection time: ~1–2 hours on A100 for 50 documents

## Estimated OpenRouter API cost (Phase 1)

- Model: qwen/qwen-2.5-72b-instruct
- 50 docs × 20 questions = 1000 Q&A pairs
- + paraphrase augmentation
- Estimated cost: $2–5 USD total
