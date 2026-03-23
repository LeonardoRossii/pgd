# PGD

Projected Gradient Descent (PGD) for 2D Constrained Optimization.

**PhD Course Project** — *Proximal Methods in Numerical Optimization*  
University of Trento, A.Y. 2025/2026  
Instructor: PhD Andrea De Marchi

## Overview

Implements Projected Gradient Descent (PGD) with backtracking line search to solve:

$$\min_{z \in C} f(z)$$

Tested on the **Himmelblau** and **Three Hump Camel** functions subject to a **disk constraint**.

## Installation

```bash
conda env create -f environment.yaml
conda activate proxopt
```

## Usage

Run the script:

```bash
cd src
python main.py
```

Or explore the notebook:

```bash
jupyter notebook notebooks/proximal_optimization_assignment.ipynb
```

## Project Structure

```
pgd/
├── notebooks/
│   └── proximal_optimization_assignment.ipynb
├── src/
│   ├── functions.py      # Test functions definition
│   ├── constraints.py    # Constraint sets with projection operators
│   ├── solver.py         # PGD solver with backtracking line search
│   ├── plot.py           # Visualization utilities
│   └── main.py           # Entry point
└── README.md
```

## License

MIT
