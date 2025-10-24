# Week 2 — AI Models: Hands-on Tutorial Notebooks

This repo contains 8 short, hands-on Jupyter notebooks (20–30 minutes each) covering core ML/DL models with practical guidance on assumptions, cautions, and diagnostics.

Stack and constraints:
- Frameworks: PyTorch (+ torchvision) and Hugging Face (transformers, datasets)
- Classic ML: scikit-learn + statsmodels
- Environment: CPU-only
- Connectivity: Internet available for dataset/model downloads
- Datasets: Standard public datasets (sklearn, torchvision, HF Datasets)
- Exercises: Each notebook includes exercises with instructor solution cells hidden/collapsed by default

Contents
1) 01_linear_regression.ipynb
   - Synthetic/California housing regression, assumptions checks
   - Residual/QQ plots, homoscedasticity tests, VIF
2) 02_kmeans_clustering.ipynb
   - Standardization, elbow method, silhouette score, cluster inspection
3) 03_decision_trees.ipynb
   - Train/test split, max_depth/pruning, class imbalance considerations
4) 04_random_forest.ipynb
   - Feature importance, OOB score (optional), memory/runtime tips
5) 05_xgboost.ipynb
   - Basic training with early stopping and sensible defaults, tuning guide
6) 06_neural_networks_ann.ipynb
   - PyTorch MLP on tabular data, scaling, overfitting controls
7) 07_cnn_resnet.ipynb
   - Simple CNN on MNIST/Fashion-MNIST (CPU-friendly), normalization, augmentation
8) 08_transformers_text.ipynb
   - Tokenization, max length, sentiment classification via HF pipelines (inference-first)

Quickstart
1) Create and activate a Python environment (recommended Python 3.9+)
   - macOS/Linux:
     python3 -m venv .venv
     source .venv/bin/activate
   - Windows (PowerShell):
     py -m venv .venv
     .venv\Scripts\Activate.ps1

2) Install requirements (CPU-only)
   pip install --upgrade pip
   pip install -r requirements.txt

3) Launch Jupyter
   jupyter lab
   # or
   jupyter notebook

4) Open the notebooks in the notebooks/ folder in order.

Notes on CPU-only and runtime
- All examples are constrained for CPU: small subsets, few epochs, and/or inference-first for deep models.
- CNN uses MNIST/Fashion-MNIST; training epochs are short.
- Transformers notebook uses Hugging Face pipelines for inference to avoid heavy training.
- If runtime is tight, run only the mandatory cells; stretch sections are clearly marked optional.

Instructor solutions
- Exercises include solution cells that are hidden/collapsed by default.
- How to reveal:
  - Jupyter Notebook: Click the collapse/expand indicator on the cell or toggle cell metadata in View.
  - VS Code Notebooks: Click the chevron (▸) on the cell to expand. Look for cells tagged “solution”.
- Solutions are marked with the “solution” tag and have hidden source via metadata. They are intended for instructors.

Data and downloads
- Notebooks will download standard datasets/models on first run:
  - sklearn datasets (e.g., make_regression, iris, breast_cancer, california_housing)
  - torchvision datasets (MNIST/Fashion-MNIST)
  - Hugging Face pipeline models (e.g., distilbert-base-uncased-finetuned-sst-2-english)
- These are cached locally by their respective libraries.

Structure
- notebooks/
  - 01_linear_regression.ipynb
  - 02_kmeans_clustering.ipynb
  - 03_decision_trees.ipynb
  - 04_random_forest.ipynb
  - 05_xgboost.ipynb
  - 06_neural_networks_ann.ipynb
  - 07_cnn_resnet.ipynb
  - 08_transformers_text.ipynb
- requirements.txt
- README.md

Troubleshooting
- If torch/torchvision install is slow: ensure you’re on latest pip. CPU wheels will be installed by default.
- If a dataset/model download fails: verify internet connection and re-run the cell.
- If plotting windows don’t appear: ensure inline plotting is enabled (%matplotlib inline is set in notebooks).

License
- Educational use. Datasets and pre-trained models are subject to their original licenses.
