<!--- # languagemodels
LLM, NLP, ML projects. --->

## [Probing Occupational Gender Bias in GPT-2 Large Using Surprisal](https://github.com/destynee/LanguageModels/blob/main/LLM_Evaluation_ExperimentalDesign.ipynb)
This project investigates the sensitivity of `gpt2-large` language model to gender bias within occupational contexts through implementing an experimental design.

### Research Question 
Is a LLM more "surprised" (assigns lower probability) when encountering stereotype-incongruent pronoun-occupation pairs (e.g., "he" and "nurse") compared to congruent pairs (e.g., "she" and "nurse")?

### Methodology
**Model**: `gpt2-large` (Using `Transformers` & `PyTorch`  
**Stimuli**: Designed minimal pairs of sentences varying only the gendered pronoun preceding stereotyped occupations (e.g., `nurse`, `plumber`, `firefighter`).  
**Metric**: Calculated Surprisal `(-log P(word | context))` for the target occupation word using functions `next_seq_prob` and `surprisal`.  
**Analysis**: Compared surprisal values between `'Expected'` (stereotype-congruent) and `'Anomalous'` (stereotype-incongruent) conditions. The results are visualized using `Pandas`, `Matplotlib`, and `Seaborn`.

### Key Findings
The model generally exhibits higher surprisal for stereotype-incongruent pairings, suggesting it reflects biases learned from training data.  
Notable exceptions were observed for certain occupations (e.g., `'plumber'`, `'manager'`), prompting further discussion.

### Technologies
- [Python](https://docs.python.org/3/)
- Jupyter Notebook
- [`PyTorch`](https://github.com/pytorch/pytorch)
- [`Transformers`](https://github.com/huggingface/transformers)
- [`Pandas`](https://github.com/pandas-dev/pandas)
- [`NumPy`](https://github.com/numpy/numpy)
- [`Matplotlib`](https://github.com/matplotlib/matplotlib)
- [`Seaborn`](https://github.com/mwaskom/seaborn)
