# Confidence Diagram of Nonparametric Ranking for Uncertainty Assessment in Large Language Model Evaluation

Code, data and results accompanying the manuscript *Confidence Diagram of
Nonparametric Ranking for Uncertainty Assessment in Large Language Model
Evaluation*. The repository contains

* the implementation of the proposed methodology: the kernel-smoothed
  regularized MLE of the contextual Bradley–Terry–Luce scores, the Gaussian
  multiplier bootstrap for the simultaneous confidence band, the pairwise and
  top-*K* tests, and the step-down construction of the confidence diagram with
  its transitive reduction;
* the synthetic experiments of Section 5.1 and the comparison with parametric
  contextual BTL baselines of Section 5.2 (Table 1, Figure 6);
* the MMLU application of Section 5.3 (Figures 7–8) and the embedding-sensitivity
  study of Appendix F, together with the prompts, model responses, GPT-4
  pairwise judgments and estimation outputs used in the paper.

Everything needed to regenerate the tables and figures from the included data
is in this repository; the only steps that cannot be rerun offline are the
calls to external models (OpenAI embeddings, GPT-4 judging, and response
generation with the open-weight LLMs), whose outputs are included.

## Repository layout

```
synthetic/            Section 5.1 — synthetic experiments
  simu1.py            data generation + estimator + bootstrap, first design  θ_i(x) = exp(k·i·Σx)
  simu2.py            same for the second design                             θ_i(x) = i·exp(Σx) + i
  plot1.ipynb         MSE study (Figure 2) and confidence-band coverage (Figure 3)
  plot2.ipynb         pairwise tests / step-down screening -> dominance matrices (Figures 4–5)
  plot3.R             draws the Hasse (confidence) diagram and the rank heatmap from the dominance matrices
  figures/            the figures as they appear in the paper

baselines/            Section 5.2 — comparison with parametric contextual BTL baselines
  baseline_common.py  vectorized implementation of the proposed estimator and bootstrap
                      (fit_theta_at, gmb_ratio_draws, band_halfwidth, pairwise_halfwidths, run_proposed)
                      and the three synthetic designs (settings 1–3)
  baseline_parametric.py   linear and quadratic contextual BTL models with Gaussian-multiplier bands
  run_baseline_comparison.py   driver producing Table 1
  plot_baseline_comparison.py  Figure 6
  results/            the 50-replication results behind Table 1 (.npz, LaTeX rows, run log)
  figures/            Figure 6

real_data/            Section 5.3 and Appendices E–F — MMLU application
  collect_prompt.py   step 1: MMLU questions -> prompts -> text-embedding-3-small embeddings
  run_llama1.py, run_llama2.py, run_alpaca.py (+ .sh SLURM templates)
                      step 2: responses of the open-weight models (GPT-3.5 / GPT-4o mini via the OpenAI API)
  compare_llms.ipynb  step 3: GPT-4 pairwise judgments on an Erdős–Rényi comparison graph (p = 0.5)
  estimation.py       step 4: kernel-smoothed MLE at the training prompts, estimates and bootstrap
                      draws at the held-out prompts (h = 0.3, d = 16)
  plot.ipynb          step 5: win-rate heatmaps (Figure 7)
  sensitivity_estimation.py   step 6: pairwise dominance tests under the original PCA projection and
                      under an alternative orthonormal projection (Figure 8 and Appendix F)
  render_hasse.py     step 7: draws the four-panel confidence diagrams from the dominance matrices
  data/prompt/        questions and 1536-dim embeddings, training (100 per subject) and held-out sets
  data/response/      raw responses of the five models on the training prompts
  data/eval/          GPT-4 pairwise judgments (one JSON per subject)
  data/out/           estimation outputs of estimation.py (h = 0.3) and the original dominance matrices
  results/sensitivity/  dominance matrices and estimates for Figure 8 / Appendix F
  figures/            Figures 7–8 and the Appendix F figure
```

## Requirements

Python 3.9+ with the packages in `requirements.txt`
(`pip install -r requirements.txt`). The core estimation code only needs
`numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib` and `tqdm`;
`openai`, `datasets`, `transformers` and `torch` are needed only to regenerate
the LLM responses and judgments. `synthetic/plot3.R` needs R with `ggplot2`,
`reshape2`, `dplyr`, `viridis`, `akima`, `RColorBrewer` and `magrittr`.

## The method in code

The estimator and inference procedures are implemented twice: once in the
scripts used for the original experiments (`synthetic/simu1.py`,
`synthetic/simu2.py`, `real_data/estimation.py`; functions `f_theta_h2`
for the regularized kernel MLE and `generate_W` for the multiplier-bootstrap
statistic), and once in vectorized form in `baselines/baseline_common.py`,
which is used by the baseline comparison and by the real-data sensitivity
study. The two agree numerically; the vectorized version is the one to start
from for new applications:

| Paper | Function |
|---|---|
| Regularized kernel MLE, eq. (3.2) | `baseline_common.fit_theta_at` |
| Bootstrap statistic *W*, eqs. (3.3)–(3.6) | `baseline_common.gmb_ratio_draws` |
| Confidence band, Algorithm 1 | `baseline_common.band_halfwidth`, `run_proposed` |
| Pairwise test statistics *T_ij*, *W_ij* | `baseline_common.pairwise_halfwidths`; `sensitivity_estimation.run_subject` |
| Confidence diagram, Algorithm 2 (step-down + transitive reduction) | `synthetic/plot2.ipynb`; `render_hasse.transitive_reduction` |

## Reproducing the results

### Section 5.1 — synthetic experiments (Figures 2–5)

`simu1.py` and `simu2.py` are command-line programs; for example

```
cd synthetic
python simu1.py --n 20 --L_val 100 --p 0.5 --dim 3 --h 0.1 --k 0.01 --w_iter 100 --iter_num 50 --cpu 8
python simu2.py --n 20 --L_val 100 --p 0.2 --dim 3 --h 0.23 --k 1.0  --w_iter 100 --iter_num 50 --cpu 8
```

run one configuration and save the estimates, ground truth and bootstrap
draws. The notebooks reproduce the figures: `plot1.ipynb` loops over the
configurations of Figure 2 (`p = 0.5` with varying `n`, `L`; `n = 20` with
varying `L`, `p`) and draws the coverage plots of Figure 3 (`x_3 = 0.4`,
`α = 0.1`); `plot2.ipynb` computes the pairwise statistics, the step-down
quantiles and the dominance matrices for the confidence diagram
(`n = 20`, `L ∈ {50, 100}`, `p = 0.2`, `α = 0.1`, 50 replications) and writes
them to `results/`; `plot3.R` draws Figure 4 and the rank heatmap of Figure 5
from those files. Paths inside the notebooks are relative to `synthetic/`.

### Section 5.2 — baseline comparison (Table 1, Figure 6)

```
cd baselines
python run_baseline_comparison.py --setting all --reps 50 --n 5 --L_val 1000 --p 0.5 --dim 3 \
    --h 0.25 --lambda_val 1e-5 --w_iter 200 --pp 400 --alpha 0.1 \
    --grid_num 125 --grid_lo 0.3 --grid_hi 0.7 --scale 0.1 --amp 2.0 --seed 42 --cpu 8 --out results
python plot_baseline_comparison.py --out figures/baseline_comparison.pdf
```

The first command writes one `.npz` and one `_table.tex` per setting (the
rows of Table 1) and prints the summary that is also stored in
`results/run_log.txt`; it takes a few minutes on a laptop. The second draws
Figure 6.

### Section 5.3 — MMLU application (Figures 7–8, Appendices E–F)

All commands are run from `real_data/`. Steps 1–3 call external models and are
included for completeness; their outputs are in `data/`.

1. `collect_prompt.py` — downloads the MMLU test split (`cais/mmlu`) for a
   subject, forms the prompts (question + answer choices) and embeds them with
   `text-embedding-3-small` (`OPENAI_API_KEY` must be set). Outputs
   `data/prompt/questions_*.csv` and `data/prompt/embedded_questions_*.csv`.
2. `run_llama1.py`, `run_llama2.py`, `run_alpaca.py` — generate responses with
   LLaMA-7B, Llama-2-7b-chat and Alpaca-7B (set the checkpoint paths at the top
   of each file; the `.sh` files are SLURM templates). GPT-3.5 and GPT-4o mini
   responses were obtained through the OpenAI API. Outputs
   `data/response/<subject>_<model>_num=100.json`.
3. `compare_llms.ipynb` — draws the comparison pairs (edge probability 0.5)
   and asks GPT-4 to judge each pair on the 100 training prompts. Outputs
   `data/eval/<subject>_num=100.json`.
4. `python estimation.py <k>` with `k = 0, 1, 2, 3` for anatomy, clinical
   knowledge, medical genetics, college biology — per-subject PCA to 16
   dimensions, regularized kernel MLE (`h = 0.3`, `λ = 1e-5`) at the training
   prompts and at the held-out prompts, and 100 multiplier-bootstrap draws.
   Outputs `data/out/<subject>_h0.3_dim16_tr100_te*.npz`.
5. `plot.ipynb` — win-rate heatmaps of Figure 7 from the estimates.
6. `python sensitivity_estimation.py --sub all --projection both` — pairwise
   dominance tests (`α = 0.05`, 200 bootstrap draws) under the original PCA
   projection and under a random orthonormal projection of the leading
   principal subspace (Appendix F). Outputs the dominance matrices
   `results/sensitivity/Imax_<subject>_<pca|rotated>_h0.3_dim16.txt` and the
   corresponding `.npz` files. The script uses an exact SVD
   (`svd_solver="full"`) so that the result is deterministic.
7. `python render_hasse.py --mode pca --out figures/hasse_pca_rerun.pdf` and
   `python render_hasse.py --mode rotated --out figures/embedding_sensitivity.pdf`
   — the confidence diagrams of Figure 8 and Appendix F (transitive reduction
   of the dominance relation, drawn as a Hasse diagram).

### Data

* `data/prompt/*.csv`: MMLU questions with answer choices and their
  1536-dimensional `text-embedding-3-small` embeddings; `questions_*` /
  `embedded_questions_*` are the 100 training prompts per subject and
  `*_test_*` the held-out evaluation prompts (50 per subject; 16 for medical
  genetics).
* `data/response/*.json`: responses of GPT-3.5, GPT-4o mini, LLaMA-1, LLaMA-2
  and Alpaca to the training prompts.
* `data/eval/*.json`: for each compared pair, the 100 GPT-4 judgments
  (`GPT4_output`, 1 if the first model's answer is preferred).
* `data/out/Imax_*`: dominance matrices of the original analysis; rows/columns
  are ordered `gpt3, alpaca, gpt4o, llama1, llama2`.

MMLU is distributed under the MIT license by its authors
(Hendrycks et al., 2021).
