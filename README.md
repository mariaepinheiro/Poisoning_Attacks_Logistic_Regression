# Poisoning Attacks on Logistic Regression

Code and results for the paper *Poisoning Attacks on Logistic Regression*
by C. M. Costa, M. E. Pinheiro, and M. Schmidt.

The poisoning attack is a bilevel problem.  We compare two single-level
reformulations, both solved with the spatial branch-and-bound solver SCIP:

- **PA_G** — the KKT reformulation (P2);
- **CAR** — the convex-analytic reformulation (P3), whose only
  nonconvexities are bounded bilinear terms.

Each is run with two sets of bounds: only the bound of Proposition 3
(**PA_G₀**, **CAR₀**) or all bounds of Section 3.4 (**PA_G**, **CAR**).
Within each pair, both methods receive exactly the same bounds.

## Repository structure

```
src/           model: data, bounds, formulations
experiments/   scripts that run the experiments
analysis/      scripts that produce the tables and figures of the paper
results/       raw results used in the paper (CSV)
figures/       figures of the paper (PDF, and PNG for this page)
tables/        tables of the paper (LaTeX)
data/          downloaded data (Spambase)
```

| File | Content |
|---|---|
| `src/datasets.jl`, `src/spambase_loader.jl` | data loading and preprocessing (Section 4.1) |
| `src/common.jl` | train/test split, budget `q`, poisoning labels, metrics |
| `src/models_global_regularization.jl` | solver setup, logistic regression, helpers |
| `src/bounds.jl` | bounds of Section 3.4 (`:prop3` = Prop. 3 only, `:full` = all) |
| `src/models_bounded.jl` | formulations (P2) and (P3) with given bounds |
| `experiments/quick_test.jl` | smoke test (≈ 5 min) |
| `experiments/run_phase1.jl` | Phase 1: LR and PA_L (Ipopt, 5 starts, 5 partitions) |
| `experiments/run_phase2.jl` | Phase 2: PA_G₀, CAR₀, CAR (SCIP, 3600 s) |
| `experiments/run_phase2_pagfull.jl` | Phase 2: PA_G with all bounds |
| `analysis/tables_paper.jl` | tables of Section 4 |
| `analysis/figures_phase1.jl`, `analysis/figures_phase2.jl` | figures of Section 4 |

## Requirements

- Julia ≥ 1.12
- SCIP and Ipopt are installed automatically by SCIP.jl and Ipopt.jl.
- Gurobi is **not** required.

From the repository root, install the exact package versions used in the
paper (`Project.toml`, `Manifest.toml`):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Reproducing the paper

The raw results are included in `results/`, so the tables and figures can be
regenerated without rerunning the experiments:

```bash
julia --project=. analysis/tables_paper.jl      # → tables/
julia --project=. analysis/figures_phase1.jl    # → figures/ac_grid, mcc_grid
julia --project=. analysis/figures_phase2.jl    # → figures/gap_grid, ratio_grid
```

To rerun the experiments:

```bash
julia --project=. experiments/quick_test.jl          # check the installation
julia --project=. experiments/run_phase1.jl          # several hours
julia --project=. experiments/run_phase2.jl          # ≈ 20 h with 5 workers
julia --project=. experiments/run_phase2_pagfull.jl  # ≈ 7 h with 5 workers
```

The run scripts save each finished instance as soon as it completes and skip
finished instances when restarted, so long runs can be interrupted and
resumed.  Running times depend on the machine; with a time limit, the results
of the global solver may differ slightly between machines.

## Results

### Phase 1: attack effectiveness

![Accuracy under attack](figures/ac_grid.pdf)
![MCC under attack](figures/mcc_grid.pdf)

### Phase 2: global optimization

![Dual-bound ratios](figures/ratio_grid.pdf)

## Citation

```bibtex
@article{CostaPinheiroSchmidt2026,
  title  = {Poisoning Attacks on Logistic Regression},
  author = {Costa, Carina Moreira and Pinheiro, Maria Eduarda and Schmidt, Martin},
  year   = {2026}
}
```

## License

To be decided.
