# Similarity Search for Dense Multidimensional Vectors

Assignment 3 for the course **Complex Data Management (MYE041/PLE081)** — MSc, Spring Semester 2025-26.

## Overview

Implementation and evaluation of three similarity search methods on a dataset of 10,000 ten-dimensional vectors, using the **Euclidean (L2) distance**:

- **Naive Linear Scan** — brute-force baseline
- **Pivot-based** — uses triangle inequality for pruning
- **iDistance** — maps objects to a 1D sorted array for efficient range search

Both **Range Queries** and **kNN Queries** are supported.

## File Structure

```
├── similarity_search.py   # Core implementation (Parts 1–4)
├── plots.py               # Experiments & plots (Part 5)
└── README.md
```

## Requirements

- Python 3.x (standard library only — no external packages needed)
- `matplotlib` (only for `plots.py`)

## Usage

### similarity_search.py

Runs both Range and kNN queries for a given number of pivots, epsilon, and k:

```bash
python similarity_search.py
```

You can adjust `numpivots`, `epsilon`, and `k` directly in the `main` block.

The program expects two input files in the same directory:
- `data10K10.txt` — dataset (10,000 objects, 10 dimensions, space-separated)
- `queries10.txt` — query objects (same format)

### plots.py

Runs the full experimental evaluation and saves a `plots.png` file with 4 diagrams:

```bash
python plots.py
```

## Implementation Details

### Part 1 — Data Loading & Pivot Selection

Pivots are selected greedily to maximize spread:
- Pivot 0: farthest object from seed (object 0)
- Pivot k: object with maximum sum of distances to all previous pivots

A precomputed `distances[i][j]` matrix stores the distance from every object `i` to every pivot `j`.

### Part 2 — iDistance Index

Each object `o` is mapped to a 1D value:
```
v(o) = maxd * i + dist(o, p_i)
```
where `p_i` is the nearest pivot to `o` and `maxd = max{ maxd(p_i) }`.
The resulting array of `(iDist, oid)` pairs is sorted for binary search.

### Part 3 — Range Queries (q, ε)

| Method | Strategy |
|---|---|
| Naive | Compute dist(q,o) for all objects |
| Pivot-based | Prune o if \|dist(p,o) - dist(p,q)\| > ε for any pivot |
| iDistance | Binary search on 1D array; prune entire partitions |

### Part 4 — kNN Queries (q, k)

| Method | Strategy |
|---|---|
| Naive | Max-heap of size k over all objects |
| Pivot-based | Same as Naive with dynamic pivot-based pruning |
| iDistance | Start from nearest pivot's partition; use current k-th distance as ε for pruning other partitions |

## Results (numpivots=10)

### Range Queries (ε=0.2)

| Method | Avg Distance Computations | Total Time (s) |
|---|---|---|
| Naive | 10000 | ~3.65 |
| Pivot-based | 16.71 | ~0.61 |
| iDistance | 2074.24 | ~0.95 |

### kNN Queries (k=5)

| Method | Avg Distance Computations | Total Time (s) |
|---|---|---|
| Naive | 10000 | ~3.83 |
| Pivot-based | 3156.76 | ~2.76 |
| iDistance | 6128.03 | ~3.06 |
