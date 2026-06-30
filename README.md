# GKP Superdense Coding over Lossy Bosonic Channels

Numerical simulations accompanying the paper:

**"Performance of Square and Hexagonal GKP Superdense Coding over Lossy Bosonic Channels"**

This repository reproduces the analytical capacity calculations, numerical parameter sweeps, validation tests, and figures presented in the paper.

---

## Repository Structure

```
.
├── paper/
│   └── paper.pdf
│
├── src/
│   ├── sweep.py          # Capacity calculations and parameter sweep
│   └── validation.py     # Monte Carlo and circuit-level validation
│
├── figures/
│   └── figures.py        # Reproduces all figures in the paper
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Features

- Square-lattice GKP superdense coding capacity
- Hexagonal-lattice GKP superdense coding capacity
- Gaussian entanglement-assisted (EA) capacity benchmark
- Holevo capacity benchmark
- Capacity sweeps over:
  - Squeezing: 7–20 dB
  - Channel transmissivity:
    - η = 0.70
    - η = 0.80
    - η = 0.90
    - η = 0.95
    - η = 0.99
- Threshold analysis
- Monte Carlo verification
- Stage-by-stage circuit simulation validation

---

## Requirements

Python 3.10 or newer.

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Reproducing the Results

Run the analytical sweep

```bash
python src/sweep.py
```

Run the validation tests

```bash
python src/validation.py
```

Generate all figures

```bash
python figures/figures.py
```

---

## Validation

The analytical expressions are independently verified using:

- Monte Carlo decoding simulations
- Stage-by-stage circuit simulations of the complete GKP superdense coding protocol

The validation script compares simulated results with the analytical predictions across the parameter range considered in the paper.

---

## Citation

If you use this code in your research, please cite the accompanying paper.

```
Citation information will be added after publication.
```

---

## License

This project is released under the MIT License.
