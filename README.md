# Replication package

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE) [![License: CC BY 4.0](https://img.shields.io/badge/paper%20%26%20data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23050144.svg)](https://doi.org/10.5281/zenodo.23050144)

**Superhuman Scores, Sub-Human Adaptation: A Multi-Dimensional, Growth-Aware Comparison of Frontier AI Models and Human Intelligence**
Shreyash Gondane, Independent Researcher (2026). Data snapshot 27 September 2026, re-verified 29 September 2026.

## Reproduce everything

```bash
pip install -r requirements.txt
bash run_all.sh
```

Tested with Python 3.11, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.9. All random draws use fixed seeds, so results match the paper exactly.

## Contents

| Path | What it is |
|---|---|
| `data/evidence.csv` | Provenance ledger: every figure used, its value, evidence level, source URL and check date (28 rows) |
| `code/calc.py` | Composite indices (arithmetic and geometric), static/adaptive z-scores, adaptability-adjusted IQ, success-landscape shares, horizon projections |
| `code/mc.py` | Monte Carlo: 200,000 Dirichlet weightings x score noise; sensitivity coefficients; growth-parameter simulation |
| `code/figs.py` | Figures 1-9 |
| `code/figs_mc.py` | Figures 10-11 (run after `mc.py`) |
| `paper/` | LaTeX source (compile with XeLaTeX) and the PDF |
| `interactive/index.html` | Interactive companion page (open in a browser; loads Chart.js and Three.js from cdnjs) |

## Where the inputs come from

Model scores are hard-coded at the top of each script and match Table 1 of the paper. Each value's source and evidence level is in `data/evidence.csv`. The nine-dimension profile scores and the human-master parameters are author estimates, as stated in the paper; change them in `calc.py` / `mc.py` to test your own assumptions.

## Map from paper to code

| Paper element | Script |
|---|---|
| Table 3 (composites), Table 4 (IQ*), Table 5 (projections) | `calc.py` |
| Section 3.7 and 4.6, Table 6 (Monte Carlo) | `mc.py` |
| Figures 1-9 | `figs.py` |
| Figures 10-11 | `figs_mc.py` |

## AI assistance

Data compilation, code, figures and drafting were done with the assistance of Anthropic's Claude; the author directed and reviewed the work. See the paper's disclosure.

## Citation

Gondane, S. (2026). *Superhuman Scores, Sub-Human Adaptation* [Preprint and replication package]. Zenodo. https://doi.org/10.5281/zenodo.23050144
