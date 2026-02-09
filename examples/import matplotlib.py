import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams

# =========================
# Make all plots match LaTeX font
# =========================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['font.size'] = 14
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['legend.fontsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12


# =========================
# Load Data (from your tables)
# =========================

df_new = pd.DataFrame({
    "Drones":   [2,2,2,2,2,2,2,2,3,3,3,3,3,4,4],
    "Routes":   [2,3,4,5,6,7,8,9,3,4,5,6,7,4,5],
    "Makespan": [4.0,7.4,10.8,10.8,14.3,13.8,14.6,16.8,7.4,7.4,7.4,9.2,9.4,7.4,7.4],
    "QuboTerms":[8,15,24,27,38,51,66,83,27,42,48,66,87,64,74]
})

df_old = pd.DataFrame({
    "Drones":   [2,2,2,2,2,2,3,3,3,4],
    "Routes":   [2,3,4,5,6,7,3,4,5,4],
    "Makespan": [7.4,11.4,11.4,11.4,20.1,20.1,9.7,14.9,16.0,10.8],
    "QuboTerms":[12,21,32,43,58,75,36,54,72,80]
})

df_compare = pd.merge(df_old, df_new, on=["Drones","Routes"], suffixes=("_Old","_New"))


# =========================
# 1. Makespan Comparison (Side-by-Side Bars)
# =========================
plt.figure(figsize=(8,5))
bar_width = 0.35
x = range(len(df_compare))

plt.bar([i - bar_width/2 for i in x], df_compare["Makespan_Old"],
        width=bar_width, color="#b99999", label="Old System")
plt.bar([i + bar_width/2 for i in x], df_compare["Makespan_New"],
        width=bar_width, color="#8aa1c1", label="New System")

plt.xticks(x, [f"{r}" for r in df_compare["Routes"]])
plt.xlabel("Number of Routes")
plt.ylabel("Makespan")
plt.title("Makespan Comparison (Old vs New)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# =========================
# 2. QUBO Term Growth (Line Plot)
# =========================
plt.figure(figsize=(8,5))
sns.lineplot(data=df_old, x="Routes", y="QuboTerms", marker="o",
             label="Old System", color="#b27b7b")
sns.lineplot(data=df_new, x="Routes", y="QuboTerms", marker="o",
             label="New System", color="#7b8dad")

plt.xlabel("Number of Routes")
plt.ylabel("QUBO Terms")
plt.title("QUBO Term Growth (Old vs New)")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 3. Failure Rate Comparison (Unassigned Routes / Idle Drones)
# =========================

labels_new = [
    '2D/2R', '2D/3R', '2D/4R', '2D/5R', '2D/6R', '2D/7R', '2D/8R', '2D/9R',
    '3D/3R', '3D/4R', '3D/5R', '3D/6R', '3D/7R',
    '4D/4R', '4D/5R'
]

new_system_data = {
    'unassigned_routes_pct': [0.0]*15,
    'drones_zero_routes_pct': [0.0]*15
}

old_system_raw_data = {
    '2D/2R': {'unassigned': 0.0, 'idle_drones': 50.0},
    '2D/3R': {'unassigned': 0.0, 'idle_drones': 0.0},
    '2D/4R': {'unassigned': 0.0, 'idle_drones': 0.0},
    '2D/5R': {'unassigned': 6.7, 'idle_drones': 0.0},
    '2D/6R': {'unassigned': 5.6, 'idle_drones': 0.0},
    '2D/7R': {'unassigned': 9.5, 'idle_drones': 0.0},
    '3D/3R': {'unassigned': 0.0, 'idle_drones': 22.2},
    '3D/4R': {'unassigned': 0.0, 'idle_drones': 33.3},
    '3D/5R': {'unassigned': 0.0, 'idle_drones': 22.2},
    '4D/4R': {'unassigned': 0.0, 'idle_drones': 33.3}
}

old_system_data = {
    'unassigned_routes_pct': [old_system_raw_data.get(label, {}).get('unassigned', np.nan) for label in labels_new],
    'drones_zero_routes_pct': [old_system_raw_data.get(label, {}).get('idle_drones', np.nan) for label in labels_new]
}

x = np.arange(len(labels_new))
width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Failure Rate Comparison: Old vs New System', fontsize=18)

ax1.bar(x - width/2, new_system_data['unassigned_routes_pct'], width, label='New System', color="#8aa1c1")
ax1.bar(x + width/2, old_system_data['unassigned_routes_pct'], width, label='Old System', color="#b99999")
ax1.set_ylabel("Unassigned Routes (%)")
ax1.set_title("Unassigned Route Failures")
ax1.grid(axis='y', linestyle="--", alpha=0.5)
ax1.legend()

ax2.bar(x - width/2, new_system_data['drones_zero_routes_pct'], width, label='New System', color="#8aa1c1")
ax2.bar(x + width/2, old_system_data['drones_zero_routes_pct'], width, label='Old System', color="#b99999")
ax2.set_ylabel("Idle Drone Rate (%)")
ax2.set_xlabel("Drone / Route Configuration")
ax2.set_title("Zero-Route Drone Failures")
ax2.set_xticks(x)
ax2.set_xticklabels(labels_new, rotation=45, ha="right")
ax2.grid(axis='y', linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()




# Formulating QUBO for 2 drones and 2 routes
# Final QUBO has 12 terms.

# **Running 3 Trials: 2 Drones, 2 Routes**
# QUBO terms: 12
# Formulating QUBO for 2 drones and 2 routes
# Final QUBO has 12 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 0.98 seconds.

# 🔄 Trial 1
# Optimized Drone Assignments (route→drone): {1: 1, 0: 1}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [[0, 2, 10, 0], [0, 13, 7, 0]], **Total Load = 7.4**
# 📦 Coverage: 0/2 routes assigned exactly once (unassigned=0, overassigned=2)
# ⏱️ Makespan: 7.4000 | Load StdDev: 3.7000
# Formulating QUBO for 2 drones and 2 routes
# Final QUBO has 12 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 0.98 seconds.

# 🔄 Trial 2
# Optimized Drone Assignments (route→drone): {1: 1, 0: 1}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [[0, 2, 10, 0], [0, 13, 7, 0]], **Total Load = 7.4**
# 📦 Coverage: 0/2 routes assigned exactly once (unassigned=0, overassigned=2)
# ⏱️ Makespan: 7.4000 | Load StdDev: 3.7000
# Formulating QUBO for 2 drones and 2 routes
# Final QUBO has 12 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 1.02 seconds.

# 🔄 Trial 3
# Optimized Drone Assignments (route→drone): {1: 1, 0: 1}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [[0, 2, 10, 0], [0, 13, 7, 0]], **Total Load = 7.4**
# 📦 Coverage: 0/2 routes assigned exactly once (unassigned=0, overassigned=2)
# ⏱️ Makespan: 7.4000 | Load StdDev: 3.7000

# 📊 **Summary for Setting**
# 🧮 Drones = 2, Routes = 2, QUBO terms = 12
# ⏱️ Avg Makespan over 3 trials: 7.4000
# 📈 Avg Load StdDev: 3.7000
# ✅ Avg Exact-Assignment Coverage: 0.0%
# 📝 Trials detail (makespan per trial): [7.4, 7.4, 7.4]

# ✅ **Completed Runs for 2 Drones / 2 Routes** ✅

# Formulating QUBO for 2 drones and 3 routes
# Final QUBO has 21 terms.

# **Running 3 Trials: 2 Drones, 3 Routes**
# QUBO terms: 21
# Formulating QUBO for 2 drones and 3 routes
# Final QUBO has 21 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 1.65 seconds.

# 🔄 Trial 1
# Optimized Drone Assignments (route→drone): {2: 0, 1: 1, 0: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 13, 7, 0]], **Total Load = 11.399999999999999**
# 🚁 **Drone 1:** Routes [[0, 2, 10, 0]], **Total Load = 3.4**
# 📦 Coverage: 0/3 routes assigned exactly once (unassigned=0, overassigned=3)
# ⏱️ Makespan: 11.4000 | Load StdDev: 4.0000
# Formulating QUBO for 2 drones and 3 routes
# Final QUBO has 21 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 1.73 seconds.

# 🔄 Trial 2
# Optimized Drone Assignments (route→drone): {2: 0, 1: 1, 0: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 13, 7, 0]], **Total Load = 11.399999999999999**
# 🚁 **Drone 1:** Routes [[0, 2, 10, 0]], **Total Load = 3.4**
# 📦 Coverage: 0/3 routes assigned exactly once (unassigned=0, overassigned=3)
# ⏱️ Makespan: 11.4000 | Load StdDev: 4.0000
# Formulating QUBO for 2 drones and 3 routes
# Final QUBO has 21 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 1.76 seconds.

# 🔄 Trial 3
# Optimized Drone Assignments (route→drone): {2: 0, 0: 0, 1: 1}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 13, 7, 0]], **Total Load = 11.399999999999999**
# 🚁 **Drone 1:** Routes [[0, 2, 10, 0]], **Total Load = 3.4**
# 📦 Coverage: 1/3 routes assigned exactly once (unassigned=0, overassigned=2)
# ⏱️ Makespan: 11.4000 | Load StdDev: 4.0000

# 📊 **Summary for Setting**
# 🧮 Drones = 2, Routes = 3, QUBO terms = 21
# ⏱️ Avg Makespan over 3 trials: 11.4000
# 📈 Avg Load StdDev: 4.0000
# ✅ Avg Exact-Assignment Coverage: 11.1%
# 📝 Trials detail (makespan per trial): [11.4, 11.4, 11.4]

# ✅ **Completed Runs for 2 Drones / 3 Routes** ✅

# Formulating QUBO for 2 drones and 4 routes
# Final QUBO has 32 terms.

# **Running 3 Trials: 2 Drones, 4 Routes**
# QUBO terms: 32
# Formulating QUBO for 2 drones and 4 routes
# Final QUBO has 32 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 2.81 seconds.

# 🔄 Trial 1
# Optimized Drone Assignments (route→drone): {2: 0, 3: 1, 1: 1, 0: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 13, 7, 0]], **Total Load = 11.399999999999999**
# 🚁 **Drone 1:** Routes [[0, 15, 12, 0], [0, 2, 10, 0]], **Total Load = 8.6**
# 📦 Coverage: 0/4 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 11.4000 | Load StdDev: 1.4000
# Formulating QUBO for 2 drones and 4 routes
# Final QUBO has 32 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 2.95 seconds.

# 🔄 Trial 2
# Optimized Drone Assignments (route→drone): {2: 0, 3: 1, 1: 1, 0: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 13, 7, 0]], **Total Load = 11.399999999999999**
# 🚁 **Drone 1:** Routes [[0, 15, 12, 0], [0, 2, 10, 0]], **Total Load = 8.6**
# 📦 Coverage: 0/4 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 11.4000 | Load StdDev: 1.4000
# Formulating QUBO for 2 drones and 4 routes
# Final QUBO has 32 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 2.73 seconds.

# 🔄 Trial 3
# Optimized Drone Assignments (route→drone): {2: 0, 3: 1, 1: 1, 0: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 13, 7, 0]], **Total Load = 11.399999999999999**
# 🚁 **Drone 1:** Routes [[0, 15, 12, 0], [0, 2, 10, 0]], **Total Load = 8.6**
# 📦 Coverage: 0/4 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 11.4000 | Load StdDev: 1.4000

# 📊 **Summary for Setting**
# 🧮 Drones = 2, Routes = 4, QUBO terms = 32
# ⏱️ Avg Makespan over 3 trials: 11.4000
# 📈 Avg Load StdDev: 1.4000
# ✅ Avg Exact-Assignment Coverage: 0.0%
# 📝 Trials detail (makespan per trial): [11.4, 11.4, 11.4]

# ✅ **Completed Runs for 2 Drones / 4 Routes** ✅

# Formulating QUBO for 2 drones and 5 routes
# Final QUBO has 43 terms.

# **Running 3 Trials: 2 Drones, 5 Routes**
# QUBO terms: 43
# Formulating QUBO for 2 drones and 5 routes
# Final QUBO has 43 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 3.92 seconds.

# 🔄 Trial 1
# Optimized Drone Assignments (route→drone): {2: 0, 3: 1, 1: 1, 0: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 13, 7, 0]], **Total Load = 11.399999999999999**
# 🚁 **Drone 1:** Routes [[0, 15, 12, 0], [0, 2, 10, 0]], **Total Load = 8.6**
# 📦 Coverage: 0/5 routes assigned exactly once (unassigned=1, overassigned=4)
# ⏱️ Makespan: 11.4000 | Load StdDev: 1.4000
# Formulating QUBO for 2 drones and 5 routes
# Final QUBO has 43 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 4.14 seconds.

# 🔄 Trial 2
# Optimized Drone Assignments (route→drone): {2: 0, 3: 1, 1: 1, 4: 1, 0: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 13, 7, 0]], **Total Load = 11.399999999999999**
# 🚁 **Drone 1:** Routes [[0, 15, 12, 0], [0, 2, 10, 0], [0, 16, 0]], **Total Load = 8.6**
# 📦 Coverage: 0/5 routes assigned exactly once (unassigned=0, overassigned=5)
# ⏱️ Makespan: 11.4000 | Load StdDev: 1.4000
# Formulating QUBO for 2 drones and 5 routes
# Final QUBO has 43 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 3.91 seconds.

# 🔄 Trial 3
# Optimized Drone Assignments (route→drone): {2: 0, 3: 1, 1: 1, 4: 1, 0: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 13, 7, 0]], **Total Load = 11.399999999999999**
# 🚁 **Drone 1:** Routes [[0, 15, 12, 0], [0, 2, 10, 0], [0, 16, 0]], **Total Load = 8.6**
# 📦 Coverage: 0/5 routes assigned exactly once (unassigned=0, overassigned=5)
# ⏱️ Makespan: 11.4000 | Load StdDev: 1.4000

# 📊 **Summary for Setting**
# 🧮 Drones = 2, Routes = 5, QUBO terms = 43
# ⏱️ Avg Makespan over 3 trials: 11.4000
# 📈 Avg Load StdDev: 1.4000
# ✅ Avg Exact-Assignment Coverage: 0.0%
# 📝 Trials detail (makespan per trial): [11.4, 11.4, 11.4]

# ✅ **Completed Runs for 2 Drones / 5 Routes** ✅

# Formulating QUBO for 2 drones and 6 routes
# Final QUBO has 58 terms.

# **Running 3 Trials: 2 Drones, 6 Routes**
# QUBO terms: 58
# Formulating QUBO for 2 drones and 6 routes
# Final QUBO has 58 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 5.48 seconds.

# 🔄 Trial 1
# Optimized Drone Assignments (route→drone): {2: 0, 3: 0, 1: 1, 4: 1, 0: 0, 5: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 15, 12, 0], [0, 13, 7, 0], [0, 11, 4, 0]], **Total Load = 21.8**
# 🚁 **Drone 1:** Routes [[0, 2, 10, 0], [0, 16, 0]], **Total Load = 3.4**
# 📦 Coverage: 1/6 routes assigned exactly once (unassigned=0, overassigned=5)
# ⏱️ Makespan: 21.8000 | Load StdDev: 9.2000
# Formulating QUBO for 2 drones and 6 routes
# Final QUBO has 58 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 5.57 seconds.

# 🔄 Trial 2
# Optimized Drone Assignments (route→drone): {2: 0, 3: 1, 1: 1, 0: 0, 5: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 13, 7, 0], [0, 11, 4, 0]], **Total Load = 16.599999999999998**
# 🚁 **Drone 1:** Routes [[0, 15, 12, 0], [0, 2, 10, 0]], **Total Load = 8.6**
# 📦 Coverage: 0/6 routes assigned exactly once (unassigned=1, overassigned=5)
# ⏱️ Makespan: 16.6000 | Load StdDev: 4.0000
# Formulating QUBO for 2 drones and 6 routes
# Final QUBO has 58 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 5.52 seconds.

# 🔄 Trial 3
# Optimized Drone Assignments (route→drone): {2: 0, 3: 0, 1: 1, 4: 1, 0: 0, 5: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 15, 12, 0], [0, 13, 7, 0], [0, 11, 4, 0]], **Total Load = 21.8**
# 🚁 **Drone 1:** Routes [[0, 2, 10, 0], [0, 16, 0]], **Total Load = 3.4**
# 📦 Coverage: 1/6 routes assigned exactly once (unassigned=0, overassigned=5)
# ⏱️ Makespan: 21.8000 | Load StdDev: 9.2000

# 📊 **Summary for Setting**
# 🧮 Drones = 2, Routes = 6, QUBO terms = 58
# ⏱️ Avg Makespan over 3 trials: 20.0667
# 📈 Avg Load StdDev: 7.4667
# ✅ Avg Exact-Assignment Coverage: 11.1%
# 📝 Trials detail (makespan per trial): [21.8, 16.6, 21.8]

# ✅ **Completed Runs for 2 Drones / 6 Routes** ✅

# Formulating QUBO for 2 drones and 7 routes
# Final QUBO has 75 terms.

# **Running 3 Trials: 2 Drones, 7 Routes**
# QUBO terms: 75
# Formulating QUBO for 2 drones and 7 routes
# Final QUBO has 75 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 5.87 seconds.

# 🔄 Trial 1
# Optimized Drone Assignments (route→drone): {2: 0, 3: 0, 1: 1, 6: 0, 0: 0, 5: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 15, 12, 0], [0, 1, 0], [0, 13, 7, 0], [0, 11, 4, 0]], **Total Load = 23.8**
# 🚁 **Drone 1:** Routes [[0, 2, 10, 0]], **Total Load = 3.4**
# 📦 Coverage: 2/7 routes assigned exactly once (unassigned=1, overassigned=4)
# ⏱️ Makespan: 23.8000 | Load StdDev: 10.2000
# Formulating QUBO for 2 drones and 7 routes
# Final QUBO has 75 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 5.90 seconds.

# 🔄 Trial 2
# Optimized Drone Assignments (route→drone): {2: 0, 3: 1, 1: 0, 6: 1, 0: 0, 4: 1, 5: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 2, 10, 0], [0, 13, 7, 0], [0, 11, 4, 0]], **Total Load = 20.0**
# 🚁 **Drone 1:** Routes [[0, 15, 12, 0], [0, 1, 0], [0, 16, 0]], **Total Load = 7.2**
# 📦 Coverage: 3/7 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 20.0000 | Load StdDev: 6.4000
# Formulating QUBO for 2 drones and 7 routes
# Final QUBO has 75 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 5.57 seconds.

# 🔄 Trial 3
# Optimized Drone Assignments (route→drone): {2: 0, 3: 1, 1: 1, 4: 1, 0: 0, 5: 0}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 3, 19, 8, 0], [0, 13, 7, 0], [0, 11, 4, 0]], **Total Load = 16.599999999999998**
# 🚁 **Drone 1:** Routes [[0, 15, 12, 0], [0, 2, 10, 0], [0, 16, 0]], **Total Load = 8.6**
# 📦 Coverage: 0/7 routes assigned exactly once (unassigned=1, overassigned=6)
# ⏱️ Makespan: 16.6000 | Load StdDev: 4.0000

# 📊 **Summary for Setting**
# 🧮 Drones = 2, Routes = 7, QUBO terms = 75
# ⏱️ Avg Makespan over 3 trials: 20.1333
# 📈 Avg Load StdDev: 6.8667
# ✅ Avg Exact-Assignment Coverage: 23.8%
# 📝 Trials detail (makespan per trial): [23.8, 20.0, 16.6]

# ✅ **Completed Runs for 2 Drones / 7 Routes** ✅

# Formulating QUBO for 2 drones and 8 routes
# Final QUBO has 94 terms.

# ⛔ QUBO for 2 drones / 8 routes has 94 terms (> 90).
# ➡️  Stopping route growth for this drone count and moving to the next number of drones.

# Formulating QUBO for 3 drones and 3 routes
# Final QUBO has 36 terms.

# **Running 3 Trials: 3 Drones, 3 Routes**
# QUBO terms: 36
# Formulating QUBO for 3 drones and 3 routes
# Final QUBO has 36 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 3.43 seconds.

# 🔄 Trial 1
# Optimized Drone Assignments (route→drone): {2: 2, 0: 1, 1: 2}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [[0, 13, 7, 0]], **Total Load = 4.0**
# 🚁 **Drone 2:** Routes [[0, 3, 19, 8, 0], [0, 2, 10, 0]], **Total Load = 10.799999999999999**
# 📦 Coverage: 0/3 routes assigned exactly once (unassigned=0, overassigned=3)
# ⏱️ Makespan: 10.8000 | Load StdDev: 4.4582
# Formulating QUBO for 3 drones and 3 routes
# Final QUBO has 36 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 3.24 seconds.

# 🔄 Trial 2
# Optimized Drone Assignments (route→drone): {2: 2, 0: 1, 1: 2}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [[0, 13, 7, 0]], **Total Load = 4.0**
# 🚁 **Drone 2:** Routes [[0, 3, 19, 8, 0], [0, 2, 10, 0]], **Total Load = 10.799999999999999**
# 📦 Coverage: 0/3 routes assigned exactly once (unassigned=0, overassigned=3)
# ⏱️ Makespan: 10.8000 | Load StdDev: 4.4582
# Formulating QUBO for 3 drones and 3 routes
# Final QUBO has 36 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 3.64 seconds.

# 🔄 Trial 3
# Optimized Drone Assignments (route→drone): {2: 2, 0: 0, 1: 1}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 13, 7, 0]], **Total Load = 4.0**
# 🚁 **Drone 1:** Routes [[0, 2, 10, 0]], **Total Load = 3.4**
# 🚁 **Drone 2:** Routes [[0, 3, 19, 8, 0]], **Total Load = 7.3999999999999995**
# 📦 Coverage: 0/3 routes assigned exactly once (unassigned=0, overassigned=3)
# ⏱️ Makespan: 7.4000 | Load StdDev: 1.7613

# 📊 **Summary for Setting**
# 🧮 Drones = 3, Routes = 3, QUBO terms = 36
# ⏱️ Avg Makespan over 3 trials: 9.6667
# 📈 Avg Load StdDev: 3.5592
# ✅ Avg Exact-Assignment Coverage: 0.0%
# 📝 Trials detail (makespan per trial): [10.8, 10.8, 7.4]

# ✅ **Completed Runs for 3 Drones / 3 Routes** ✅

# Formulating QUBO for 3 drones and 4 routes
# Final QUBO has 54 terms.

# **Running 3 Trials: 3 Drones, 4 Routes**
# QUBO terms: 54
# Formulating QUBO for 3 drones and 4 routes
# Final QUBO has 54 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 5.27 seconds.

# 🔄 Trial 1
# Optimized Drone Assignments (route→drone): {2: 2, 3: 2, 0: 1, 1: 2}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [[0, 13, 7, 0]], **Total Load = 4.0**
# 🚁 **Drone 2:** Routes [[0, 3, 19, 8, 0], [0, 15, 12, 0], [0, 2, 10, 0]], **Total Load = 16.0**
# 📦 Coverage: 0/4 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 16.0000 | Load StdDev: 6.7987
# Formulating QUBO for 3 drones and 4 routes
# Final QUBO has 54 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 5.08 seconds.

# 🔄 Trial 2
# Optimized Drone Assignments (route→drone): {2: 2, 3: 2, 0: 1, 1: 1}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [[0, 13, 7, 0], [0, 2, 10, 0]], **Total Load = 7.4**
# 🚁 **Drone 2:** Routes [[0, 3, 19, 8, 0], [0, 15, 12, 0]], **Total Load = 12.6**
# 📦 Coverage: 0/4 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 12.6000 | Load StdDev: 5.1700
# Formulating QUBO for 3 drones and 4 routes
# Final QUBO has 54 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 5.66 seconds.

# 🔄 Trial 3
# Optimized Drone Assignments (route→drone): {2: 2, 3: 2, 0: 1, 1: 2}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [[0, 13, 7, 0]], **Total Load = 4.0**
# 🚁 **Drone 2:** Routes [[0, 3, 19, 8, 0], [0, 15, 12, 0], [0, 2, 10, 0]], **Total Load = 16.0**
# 📦 Coverage: 0/4 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 16.0000 | Load StdDev: 6.7987

# 📊 **Summary for Setting**
# 🧮 Drones = 3, Routes = 4, QUBO terms = 54
# ⏱️ Avg Makespan over 3 trials: 14.8667
# 📈 Avg Load StdDev: 6.2558
# ✅ Avg Exact-Assignment Coverage: 0.0%
# 📝 Trials detail (makespan per trial): [16.0, 12.6, 16.0]

# ✅ **Completed Runs for 3 Drones / 4 Routes** ✅

# Formulating QUBO for 3 drones and 5 routes
# Final QUBO has 72 terms.

# **Running 3 Trials: 3 Drones, 5 Routes**
# QUBO terms: 72
# Formulating QUBO for 3 drones and 5 routes
# Final QUBO has 72 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 6.09 seconds.

# 🔄 Trial 1
# Optimized Drone Assignments (route→drone): {2: 2, 3: 2, 1: 2, 4: 0, 0: 1}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 16, 0]], **Total Load = 0.0**
# 🚁 **Drone 1:** Routes [[0, 13, 7, 0]], **Total Load = 4.0**
# 🚁 **Drone 2:** Routes [[0, 3, 19, 8, 0], [0, 15, 12, 0], [0, 2, 10, 0]], **Total Load = 16.0**
# 📦 Coverage: 1/5 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 16.0000 | Load StdDev: 6.7987
# Formulating QUBO for 3 drones and 5 routes
# Final QUBO has 72 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 6.15 seconds.

# 🔄 Trial 2
# Optimized Drone Assignments (route→drone): {3: 2, 0: 1, 2: 2, 1: 2, 4: 1}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [[0, 13, 7, 0], [0, 16, 0]], **Total Load = 4.0**
# 🚁 **Drone 2:** Routes [[0, 15, 12, 0], [0, 3, 19, 8, 0], [0, 2, 10, 0]], **Total Load = 16.0**
# 📦 Coverage: 1/5 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 16.0000 | Load StdDev: 6.7987
# Formulating QUBO for 3 drones and 5 routes
# Final QUBO has 72 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 5.65 seconds.

# 🔄 Trial 3
# Optimized Drone Assignments (route→drone): {2: 2, 3: 2, 0: 0, 1: 2, 4: 1}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [[0, 13, 7, 0]], **Total Load = 4.0**
# 🚁 **Drone 1:** Routes [[0, 16, 0]], **Total Load = 0.0**
# 🚁 **Drone 2:** Routes [[0, 3, 19, 8, 0], [0, 15, 12, 0], [0, 2, 10, 0]], **Total Load = 16.0**
# 📦 Coverage: 1/5 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 16.0000 | Load StdDev: 6.7987

# 📊 **Summary for Setting**
# 🧮 Drones = 3, Routes = 5, QUBO terms = 72
# ⏱️ Avg Makespan over 3 trials: 16.0000
# 📈 Avg Load StdDev: 6.7987
# ✅ Avg Exact-Assignment Coverage: 20.0%
# 📝 Trials detail (makespan per trial): [16.0, 16.0, 16.0]

# ✅ **Completed Runs for 3 Drones / 5 Routes** ✅

# Formulating QUBO for 3 drones and 6 routes
# Final QUBO has 96 terms.

# ⛔ QUBO for 3 drones / 6 routes has 96 terms (> 90).
# ➡️  Stopping route growth for this drone count and moving to the next number of drones.

# Formulating QUBO for 4 drones and 4 routes
# Final QUBO has 80 terms.

# **Running 3 Trials: 4 Drones, 4 Routes**
# QUBO terms: 80
# Formulating QUBO for 4 drones and 4 routes
# Final QUBO has 80 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 6.48 seconds.

# 🔄 Trial 1
# Optimized Drone Assignments (route→drone): {2: 2, 3: 3, 0: 3, 1: 2}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [], **Total Load = 0**
# 🚁 **Drone 2:** Routes [[0, 3, 19, 8, 0], [0, 2, 10, 0]], **Total Load = 10.799999999999999**
# 🚁 **Drone 3:** Routes [[0, 15, 12, 0], [0, 13, 7, 0]], **Total Load = 9.2**
# 📦 Coverage: 0/4 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 10.8000 | Load StdDev: 5.0319
# Formulating QUBO for 4 drones and 4 routes
# Final QUBO has 80 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 6.35 seconds.

# 🔄 Trial 2
# Optimized Drone Assignments (route→drone): {2: 2, 3: 3, 0: 1, 1: 2}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [[0, 13, 7, 0]], **Total Load = 4.0**
# 🚁 **Drone 2:** Routes [[0, 3, 19, 8, 0], [0, 2, 10, 0]], **Total Load = 10.799999999999999**
# 🚁 **Drone 3:** Routes [[0, 15, 12, 0]], **Total Load = 5.2**
# 📦 Coverage: 0/4 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 10.8000 | Load StdDev: 3.8626
# Formulating QUBO for 4 drones and 4 routes
# Final QUBO has 80 terms.
# 🔹 Running QAOA Solver (GPU)...
# QAOA Solver completed in 6.82 seconds.

# 🔄 Trial 3
# Optimized Drone Assignments (route→drone): {2: 2, 0: 1, 1: 2, 3: 3}

# 🛠 **Drone Scheduling Results:**
# 🚁 **Drone 0:** Routes [], **Total Load = 0**
# 🚁 **Drone 1:** Routes [[0, 13, 7, 0]], **Total Load = 4.0**
# 🚁 **Drone 2:** Routes [[0, 3, 19, 8, 0], [0, 2, 10, 0]], **Total Load = 10.799999999999999**
# 🚁 **Drone 3:** Routes [[0, 15, 12, 0]], **Total Load = 5.2**
# 📦 Coverage: 0/4 routes assigned exactly once (unassigned=0, overassigned=4)
# ⏱️ Makespan: 10.8000 | Load StdDev: 3.8626

# 📊 **Summary for Setting**
# 🧮 Drones = 4, Routes = 4, QUBO terms = 80
# ⏱️ Avg Makespan over 3 trials: 10.8000
# 📈 Avg Load StdDev: 4.2524
# ✅ Avg Exact-Assignment Coverage: 0.0%
# 📝 Trials detail (makespan per trial): [10.8, 10.8, 10.8]

# ✅ **Completed Runs for 4 Drones / 4 Routes** ✅

# Formulating QUBO for 4 drones and 5 routes
# Final QUBO has 106 terms.

# ⛔ QUBO for 4 drones / 5 routes has 106 terms (> 90).
# ➡️  Stopping route growth for this drone count and moving to the next number of drones.

# Formulating QUBO for 5 drones and 5 routes
# Final QUBO has 145 terms.

# ⛔ QUBO for 5 drones / 5 routes has 145 terms (> 90).
# ➡️  Stopping route growth for this drone count and moving to the next number of drones.

# Formulating QUBO for 6 drones and 6 routes
# Final QUBO has 246 terms.

# ⛔ QUBO for 6 drones / 6 routes has 246 terms (> 90).
# ➡️  Stopping route growth for this drone count and moving to the next number of drones.