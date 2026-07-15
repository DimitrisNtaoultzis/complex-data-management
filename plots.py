#Dimitris Ntaoultzis 5311
#Pavlos Mpasoukeas 5296 


import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from similarity_search import (load_data, compute_pivots, compute_idistance,
                                range_naive, range_pivot, range_idistance,
                                knn_naive, knn_pivot, knn_idistance)

# ==================== ΠΑΡΑΜΕΤΡΟΙ ====================

NUMPIVOTS = 10
EPSILONS = [0.1, 0.2, 0.4, 0.8]
KS = [1, 5, 10, 50, 100]

# ==================== ΦΟΡΤΩΣΗ & ΠΡΟΕΤΟΙΜΑΣΙΑ ====================

print("Φόρτωση δεδομένων")
data = load_data("data10K10.txt")
queries = load_data("queries10.txt")
nq = len(queries)

print(f"Υπολογισμός {NUMPIVOTS} pivots")
pivot_ids, distances = compute_pivots(data, NUMPIVOTS)
print(f"pivots: {pivot_ids}")

print("Κατασκευή iDistance index")
idistance_array, maxd_per_pivot, maxd, nearest_pivot = compute_idistance(
    data, pivot_ids, distances)

# ==================== ΠΕΙΡΑΜΑΤΑ RANGE QUERIES ====================

print("\n--- Πειράματα Range Queries ---")

range_time_naive = []
range_time_pivot = []
range_time_idist = []
range_comp_naive = []
range_comp_pivot = []
range_comp_idist = []

for epsilon in EPSILONS:
    print(f"  ε = {epsilon}")

    # Naive
    tt, tc = 0, 0
    for q in queries:
        t0 = time.time()
        res, comps = range_naive(data, q, epsilon)
        tt += time.time() - t0
        tc += comps
    range_time_naive.append(tt)
    range_comp_naive.append(tc / nq)

    # Pivot-based
    tt, tc = 0, 0
    for q in queries:
        t0 = time.time()
        res, comps = range_pivot(data, q, epsilon, pivot_ids, distances)
        tt += time.time() - t0
        tc += comps
    range_time_pivot.append(tt)
    range_comp_pivot.append(tc / nq)

    # iDistance
    tt, tc = 0, 0
    for q in queries:
        t0 = time.time()
        res, comps = range_idistance(data, q, epsilon, pivot_ids, distances,
                                     idistance_array, maxd_per_pivot, maxd, nearest_pivot)
        tt += time.time() - t0
        tc += comps
    range_time_idist.append(tt)
    range_comp_idist.append(tc / nq)

# ==================== ΠΕΙΡΑΜΑΤΑ KNN QUERIES ====================

print("\n--- Πειράματα kNN Queries ---")

knn_time_naive = []
knn_time_pivot = []
knn_time_idist = []
knn_comp_naive = []
knn_comp_pivot = []
knn_comp_idist = []

for k in KS:
    print(f"  k = {k}")

    # Naive
    tt, tc = 0, 0
    for q in queries:
        t0 = time.time()
        res, comps = knn_naive(data, q, k)
        tt += time.time() - t0
        tc += comps
    knn_time_naive.append(tt)
    knn_comp_naive.append(tc / nq)

    # Pivot-based
    tt, tc = 0, 0
    for q in queries:
        t0 = time.time()
        res, comps = knn_pivot(data, q, k, pivot_ids, distances)
        tt += time.time() - t0
        tc += comps
    knn_time_pivot.append(tt)
    knn_comp_pivot.append(tc / nq)

    # iDistance
    tt, tc = 0, 0
    for q in queries:
        t0 = time.time()
        res, comps = knn_idistance(data, q, k, pivot_ids, distances,
                                   idistance_array, maxd_per_pivot, maxd, nearest_pivot)
        tt += time.time() - t0
        tc += comps
    knn_time_idist.append(tt)
    knn_comp_idist.append(tc / nq)

# ==================== PLOTS ====================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Evaluation of Similarity Search Methods', fontsize=14)

# Plot 1: Range - Total Time
ax = axes[0, 0]
ax.plot(EPSILONS, range_time_naive, marker='o', label='Naive')
ax.plot(EPSILONS, range_time_pivot, marker='s', label='Pivot-based')
ax.plot(EPSILONS, range_time_idist, marker='^', label='iDistance')
ax.set_xlabel('ε')
ax.set_ylabel('Total Time (s)')
ax.set_title('Range Queries - Total Time')
ax.legend()
ax.grid(True)

# Plot 2: Range - Avg Distance Computations
ax = axes[0, 1]
ax.plot(EPSILONS, range_comp_naive, marker='o', label='Naive')
ax.plot(EPSILONS, range_comp_pivot, marker='s', label='Pivot-based')
ax.plot(EPSILONS, range_comp_idist, marker='^', label='iDistance')
ax.set_xlabel('ε')
ax.set_ylabel('Avg Distance Computations')
ax.set_title('Range Queries - Avg Distance Computations')
ax.legend()
ax.grid(True)

# Plot 3: kNN - Total Time
ax = axes[1, 0]
ax.plot(KS, knn_time_naive, marker='o', label='Naive')
ax.plot(KS, knn_time_pivot, marker='s', label='Pivot-based')
ax.plot(KS, knn_time_idist, marker='^', label='iDistance')
ax.set_xlabel('k')
ax.set_ylabel('Total Time (s)')
ax.set_title('kNN Queries - Total Time')
ax.legend()
ax.grid(True)

# Plot 4: kNN - Avg Distance Computations
ax = axes[1, 1]
ax.plot(KS, knn_comp_naive, marker='o', label='Naive')
ax.plot(KS, knn_comp_pivot, marker='s', label='Pivot-based')
ax.plot(KS, knn_comp_idist, marker='^', label='iDistance')
ax.set_xlabel('k')
ax.set_ylabel('Avg Distance Computations')
ax.set_title('kNN Queries - Avg Distance Computations')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('plots.png', dpi=150)
print("\nΤα plots αποθηκεύτηκαν στο plots.png")