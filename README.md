# GKP Superdense Coding

Reference implementation accompanying the paper:

> **Enhanced Superdense Coding with Finite-Energy GKP Codes over a Lossy Bosonic Channel**

This repository contains the analytical implementation, numerical benchmarks, figure-generation scripts, and independent validation routines used in the manuscript.

---

## Repository Structure

```
.
├── capacity_sweep.py     # Main analytical implementation
├── figures.py            # Generates all figures in the paper
├── validation.py         # Independent Monte Carlo and circuit validation
├── requirements.txt
└── README.md
```

---

## Features

- Analytical evaluation of square-lattice GKP superdense coding capacity
- Finite-energy GKP noise model
- Pure-loss bosonic channel simulation
- Entanglement-assisted and Holevo capacity benchmarks
- Threshold analysis over squeezing and channel transmissivity
- Independent Monte Carlo verification
- Stage-by-stage physical circuit validation

---

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib

Install dependencies with

```bash
pip install -r requirements.txt
```

---

## Reproducing the Results

### Capacity calculations

```bash
python capacity_sweep.py
```

Generates

- Capacity tables
- Threshold analysis
- Benchmark comparisons

---

### Figures

```bash
python figures.py
```

Generates all figures appearing in the manuscript.

---

### Validation

```bash
python validation.py
```

Runs two independent validation procedures:

1. Monte Carlo verification of the analytical decoding probabilities.
2. Stage-by-stage circuit simulation validating the effective noise model.

---

## Methodology

The analytical model evaluates the achievable superdense coding capacity of finite-energy GKP states transmitted through a pure-loss bosonic channel.

The implementation includes comparisons against

- Entanglement-Assisted Classical Capacity
- Holevo Capacity

to provide performance benchmarks.

Independent validation scripts verify both the analytical decoding probabilities and the effective circuit-level noise derivation.

---

## Citation

If you use this code in your research, please cite the accompanying paper.

```bibtex
@article{YOUR_CITATION,
  title   = {Enhanced Superdense Coding with Finite-Energy GKP Codes over a Lossy Bosonic Channel},
  author  = {Krishna Gupta},
  journal = {Under Review},
  year    = {2026}
}
```

(Update this entry after publication.)

---

## License

This project is released under the MIT License.

---

## Contact

For questions or issues regarding the implementation, please open a GitHub issue.
