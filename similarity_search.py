#Dimitris Ntaoultzis 5311
#Pavlos Mpasoukeas 5296 

import math
import time
import heapq
import sys

# ==================== ΕΥΚΛΕΙΔΕΙΑ ΑΠΟΣΤΑΣΗ ====================

def euclidean_distance(a, b):
    s = 0.0
    for i in range(len(a)):
        s += (a[i] - b[i]) ** 2
    return math.sqrt(s)

# ==================== ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ ====================

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                coords = list(map(float, line.split()))
                data.append(coords)
    return data

# ==================== ΥΠΟΛΟΓΙΣΜΟΣ PIVOTS ====================

def compute_pivots(data, numpivots):
    n = len(data)

    # 2D array: distances[i][j] = dist(object_i, pivot_j)
    distances = [[0.0] * numpivots for _ in range(n)]

    pivot_ids = []

    # Seed = αντικείμενο 0
    seed = data[0]

    # Πρώτο pivot: πιο μακριά από το seed
    max_dist = -1
    first_pivot = -1
    for i in range(n):
        d = euclidean_distance(seed, data[i])
        if d > max_dist:
            max_dist = d
            first_pivot = i
    pivot_ids.append(first_pivot)

    for i in range(n):
        distances[i][0] = euclidean_distance(data[i], data[first_pivot])

    # Υπόλοιπα pivots
    for p in range(1, numpivots):
        max_sum = -1
        next_pivot = -1
        for i in range(n):
            s = sum(distances[i][j] for j in range(p))
            if s > max_sum:
                max_sum = s
                next_pivot = i
        pivot_ids.append(next_pivot)

        for i in range(n):
            distances[i][p] = euclidean_distance(data[i], data[next_pivot])

    return pivot_ids, distances

# ==================== ΥΠΟΛΟΓΙΣΜΟΣ IDISTANCE INDEX ====================

def compute_idistance(data, pivot_ids, distances):
    n = len(data)
    numpivots = len(pivot_ids)

    # Για κάθε αντικείμενο, βρες το κοντινότερο pivot
    nearest_pivot = [0] * n
    for i in range(n):
        min_dist = float('inf')
        for j in range(numpivots):
            if distances[i][j] < min_dist:
                min_dist = distances[i][j]
                nearest_pivot[i] = j

    # Για κάθε pivot, υπολόγισε το maxd_i
    maxd_per_pivot = [0.0] * numpivots
    for i in range(n):
        j = nearest_pivot[i]
        if distances[i][j] > maxd_per_pivot[j]:
            maxd_per_pivot[j] = distances[i][j]

    # maxd = max από όλα τα maxd_i
    maxd = max(maxd_per_pivot)

    # Υπολόγισε iDistance για κάθε αντικείμενο
    idistance_array = []
    for i in range(n):
        j = nearest_pivot[i]
        idist = maxd * j + distances[i][j]
        idistance_array.append((idist, i))

    # Ταξινόμηση βάσει iDistance
    idistance_array.sort()

    return idistance_array, maxd_per_pivot, maxd, nearest_pivot


# ==================== ΜΕΡΟΣ 3: RANGE QUERIES ====================

def range_naive(data, query, epsilon):
    result = []
    dist_comps = 0
    for i in range(len(data)):
        d = euclidean_distance(query, data[i])
        dist_comps += 1
        if d <= epsilon:
            result.append(i)
    return result, dist_comps

def range_pivot(data, query, epsilon, pivot_ids, distances):
    result = []
    dist_comps = 0
    numpivots = len(pivot_ids)

    # Υπολόγισε dist(q, p) για κάθε pivot μία φορά
    dist_q_pivot = []
    for j in range(numpivots):
        dist_q_pivot.append(euclidean_distance(query, data[pivot_ids[j]]))
        dist_comps += 1

    for i in range(len(data)):
        pruned = False
        for j in range(numpivots):
            if abs(distances[i][j] - dist_q_pivot[j]) > epsilon:
                pruned = True
                break
        if not pruned:
            d = euclidean_distance(query, data[i])
            dist_comps += 1
            if d <= epsilon:
                result.append(i)

    return result, dist_comps

def range_idistance(data, query, epsilon, pivot_ids, distances,
                    idistance_array, maxd_per_pivot, maxd, nearest_pivot):
    import bisect
    result = []
    dist_comps = 0
    numpivots = len(pivot_ids)

    # Υπολόγισε dist(q, p_i) για κάθε pivot
    dist_q_pivot = []
    for j in range(numpivots):
        dist_q_pivot.append(euclidean_distance(query, data[pivot_ids[j]]))
        dist_comps += 1

    # Φτιάξε λίστα μόνο με τις iDistance τιμές για binary search
    idist_values = [x[0] for x in idistance_array]

    for j in range(numpivots):
        # Prune ολόκληρο το partition αν είναι πολύ μακριά
        if dist_q_pivot[j] - maxd_per_pivot[j] > epsilon:
            continue

        # Lower και upper bound στον 1D άξονα
        lower = maxd * j + dist_q_pivot[j] - epsilon
        upper = maxd * j + dist_q_pivot[j] + epsilon

        # Binary search για την πρώτη εγγραφή >= lower
        start = bisect.bisect_left(idist_values, lower)

        # Scan από start μέχρι upper
        k = start
        while k < len(idistance_array) and idistance_array[k][0] <= upper:
            oid = idistance_array[k][1]
            # Μόνο αντικείμενα που ανήκουν στο partition j
            if nearest_pivot[oid] == j:
                d = euclidean_distance(query, data[oid])
                dist_comps += 1
                if d <= epsilon:
                    result.append(oid)
            k += 1

    return result, dist_comps


# ==================== ΜΕΡΟΣ 4: KNN QUERIES ====================

def knn_naive(data, query, k):
    # Max-heap: κρατάμε (-dist, oid) ώστε η μεγαλύτερη απόσταση να είναι στην κορυφή
    heap = []
    dist_comps = 0
    for i in range(len(data)):
        d = euclidean_distance(query, data[i])
        dist_comps += 1
        if len(heap) < k:
            heapq.heappush(heap, (-d, i))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, i))
    result = [(-dist, oid) for dist, oid in heap]
    return result, dist_comps

def knn_pivot(data, query, k, pivot_ids, distances):
    heap = []
    dist_comps = 0
    numpivots = len(pivot_ids)

    # Υπολόγισε dist(q, p) για κάθε pivot μία φορά
    dist_q_pivot = []
    for j in range(numpivots):
        dist_q_pivot.append(euclidean_distance(query, data[pivot_ids[j]]))
        dist_comps += 1

    for i in range(len(data)):
        # Αν το heap δεν έχει k στοιχεία ακόμα δεν κάνουμε pruning
        if len(heap) < k:
            d = euclidean_distance(query, data[i])
            dist_comps += 1
            heapq.heappush(heap, (-d, i))
        else:
            epsilon = -heap[0][0]  # μέγιστη απόσταση στο heap = τρέχον ε
            pruned = False
            for j in range(numpivots):
                if abs(distances[i][j] - dist_q_pivot[j]) > epsilon:
                    pruned = True
                    break
            if not pruned:
                d = euclidean_distance(query, data[i])
                dist_comps += 1
                if d < epsilon:
                    heapq.heapreplace(heap, (-d, i))

    result = [(-dist, oid) for dist, oid in heap]
    return result, dist_comps

def knn_idistance(data, query, k, pivot_ids, distances,
                  idistance_array, maxd_per_pivot, maxd, nearest_pivot):
    import bisect
    dist_comps = 0
    numpivots = len(pivot_ids)

    # Υπολόγισε dist(q, p_i) για κάθε pivot
    dist_q_pivot = []
    for j in range(numpivots):
        dist_q_pivot.append(euclidean_distance(query, data[pivot_ids[j]]))
        dist_comps += 1

    # Βρες το κοντινότερο pivot στο q
    nearest_piv_to_q = min(range(numpivots), key=lambda j: dist_q_pivot[j])

    # Εφάρμοσε Naive kNN στο partition του κοντινότερου pivot
    heap = []
    for i in range(len(data)):
        if nearest_pivot[i] == nearest_piv_to_q:
            d = euclidean_distance(query, data[i])
            dist_comps += 1
            if len(heap) < k:
                heapq.heappush(heap, (-d, i))
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, i))

    # Για τα υπόλοιπα partitions
    idist_values = [x[0] for x in idistance_array]

    for j in range(numpivots):
        if j == nearest_piv_to_q:
            continue

        epsilon = -heap[0][0] if len(heap) == k else float('inf')

        # Prune ολόκληρο το partition
        if dist_q_pivot[j] - maxd_per_pivot[j] > epsilon:
            continue

        lower = maxd * j + dist_q_pivot[j] - epsilon
        upper = maxd * j + dist_q_pivot[j] + epsilon

        start = bisect.bisect_left(idist_values, lower)

        idx = start
        while idx < len(idistance_array) and idistance_array[idx][0] <= upper:
            oid = idistance_array[idx][1]
            if nearest_pivot[oid] == j:
                d = euclidean_distance(query, data[oid])
                dist_comps += 1
                if len(heap) < k:
                    heapq.heappush(heap, (-d, oid))
                elif d < -heap[0][0]:
                    heapq.heapreplace(heap, (-d, oid))
            idx += 1

    result = [(-dist, oid) for dist, oid in heap]
    return result, dist_comps

# ==================== MAIN ====================

if __name__ == "__main__":
    numpivots = 10
    epsilon = 0.2
    k = 5

    data = load_data("data10K10.txt")
    queries = load_data("queries10.txt")

    print(f"Υπολογισμός {numpivots} pivots")
    pivot_ids, distances = compute_pivots(data, numpivots)
    print(f"pivots: {pivot_ids}")

    print("Κατασκευή iDistance index")
    idistance_array, maxd_per_pivot, maxd, nearest_pivot = compute_idistance(
        data, pivot_ids, distances)

    nq = len(queries)

    # ---- Range Queries ----
    print("\n--- Range Queries (ε={}) ---".format(epsilon))


    # --- Naive ---
    total_time_naive = 0
    total_comps_naive = 0
    for q in queries:
        t0 = time.time()
        res, comps = range_naive(data, q, epsilon)
        total_time_naive += time.time() - t0
        total_comps_naive += comps


    # --- Pivot ---
    total_time_pivot = 0
    total_comps_pivot = 0
    for q in queries:
        t0 = time.time()
        res, comps = range_pivot(data, q, epsilon, pivot_ids, distances)
        total_time_pivot += time.time() - t0
        total_comps_pivot += comps

    # --- iDistance ---
    total_time_idist = 0
    total_comps_idist = 0
    for q in queries:
        t0 = time.time()
        res, comps = range_idistance(data, q, epsilon, pivot_ids, distances,
                                     idistance_array, maxd_per_pivot, maxd, nearest_pivot)
        total_time_idist += time.time() - t0
        total_comps_idist += comps

    print(f"average distance comp per query (Naive) = {total_comps_naive / nq}")
    print(f"average distance comp per query (Pivot-based) = {total_comps_pivot / nq}")
    print(f"average distance comp per query (iDistance) = {total_comps_idist / nq}")
    print(f"total time Naive = {total_time_naive}")
    print(f"total time Pivot-based = {total_time_pivot}")
    print(f"total time iDistance = {total_time_idist}")

    # ---- kNN Queries ----
    print("\n--- kNN Queries (k={}) ---".format(k))

    total_time_naive_knn = 0
    total_comps_naive_knn = 0
    for q in queries:
        t0 = time.time()
        res, comps = knn_naive(data, q, k)
        total_time_naive_knn += time.time() - t0
        total_comps_naive_knn += comps

    total_time_pivot_knn = 0
    total_comps_pivot_knn = 0
    for q in queries:
        t0 = time.time()
        res, comps = knn_pivot(data, q, k, pivot_ids, distances)
        total_time_pivot_knn += time.time() - t0
        total_comps_pivot_knn += comps

    total_time_idist_knn = 0
    total_comps_idist_knn = 0
    for q in queries:
        t0 = time.time()
        res, comps = knn_idistance(data, q, k, pivot_ids, distances,
                                   idistance_array, maxd_per_pivot, maxd, nearest_pivot)
        total_time_idist_knn += time.time() - t0
        total_comps_idist_knn += comps

    print(f"average distance comp per query (Naive kNN) = {total_comps_naive_knn / nq}")
    print(f"average distance comp per query (Pivot-based kNN) = {total_comps_pivot_knn / nq}")
    print(f"average distance comp per query (iDistance kNN) = {total_comps_idist_knn / nq}")
    print(f"total time Naive kNN = {total_time_naive_knn}")
    print(f"total time Pivot-based kNN = {total_time_pivot_knn}")
    print(f"total time iDistance kNN = {total_time_idist_knn}")
