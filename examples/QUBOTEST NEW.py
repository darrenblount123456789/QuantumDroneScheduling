import sys
import os
import numpy as np
from collections import defaultdict
from itertools import combinations


# Add the src directory to the Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src_dir, 'src'))
from input_CMT_dataset import create_pvrp_problem
from qubo_helper_drone_schedule_NEW import Qubo
from QiskitSolversDroneScheduleNew import solve_qubo
import numpy as np
import math
# numpy, matplolib, networkx, qiskit_algorithms, qiskit_aer,qiskit_optimization
# Drone Scheduler Class
class DroneQUBOScheduler:
    def __init__(self, routes, costs, num_drones):
        self.routes = routes
        self.num_routes = len(routes)
        self.costs = costs
        self.num_drones = num_drones
        self.qubo = Qubo()

    def compute_route_costs(self):
        return [sum(self.costs[route[i]][route[i+1]] for i in range(len(route)-1)) for route in self.routes]

    def formulate_qubo(self):
        print(f"Formulating QUBO for {self.num_drones} drones and {self.num_routes} routes")

        route_costs = self.compute_route_costs()
        total_cost = float(sum(route_costs))
        max_cost = float(max(route_costs))
        sum_c_sq = float(sum(c*c for c in route_costs))
        min_cost   = float(min(route_costs)) if route_costs else 1.0


        # --- Penalties tuned for QAOA stability ---
        target_load = total_cost / float(self.num_drones)
        A = 3.0 * sum_c_sq
        B = 1.0
        D = 0.01 * min_cost     # tiny anti-dump regularizer on count of routes per drone


        # --- Binary assignment variables x_{i,a} ---
        x = {(i, a): f"x_{i}_{a}" for i in range(self.num_routes) for a in range(self.num_drones)}

        # (1) Hard constraint: each route assigned exactly once:  A * (1 - Σ_a x_{i,a})^2
        for i in range(self.num_routes):
            self.qubo.add_only_one_constraint([x[i, a] for a in range(self.num_drones)], A)

        # (1b) Anti-dump: penalize (sum_i x_{i,a})^2 so routes don't all pile on one drone
        for a in range(self.num_drones):
            # diagonal terms (∑ x_{i,a})
            for i in range(self.num_routes):
                self.qubo.add((x[i, a], x[i, a]), D)
            # pairwise (2 * ∑_{i<j} x_{i,a} x_{j,a})
            for i in range(self.num_routes):
                for j in range(i + 1, self.num_routes):
                    self.qubo.add((x[i, a], x[j, a]), 2.0 * D)


        # (2) **Balance to target**: minimize Σ_a (Σ_i c_i x_{i,a} - target_load)^2
        #     Expands to, for each a:
        #       Σ_i  (B * c_i^2) x_{i,a}
        #     + 2 Σ_{i<j} (B * c_i c_j) x_{i,a} x_{j,a}
        #     - 2 Σ_i (B * target_load * c_i) x_{i,a}
        # (constant terms in target^2 are dropped)
        for a in range(self.num_drones):
            # quadratic terms
            for i in range(self.num_routes):
                ci = route_costs[i]
                self.qubo.add((x[i, a], x[i, a]), B * (ci * ci))
            for i in range(self.num_routes):
                ci = route_costs[i]
                for j in range(i + 1, self.num_routes):
                    cj = route_costs[j]
                    self.qubo.add((x[i, a], x[j, a]), 2.0 * B * ci * cj)
            # linear (on-diagonal in QUBO) centering toward target_load
            for i in range(self.num_routes):
                ci = route_costs[i]
                self.qubo.add((x[i, a], x[i, a]), -2.0 * B * target_load * ci)

        # (3) Normalize coefficients for QAOA numerics (keep |coeff| ≤ 1)
        qd = self.qubo.get_dict()
        max_abs = max(abs(v) for v in qd.values()) if qd else 1.0
        if max_abs > 0:
            scale = 1.0 / max_abs
            for k in self.qubo.terms:
                self.qubo.terms[k] *= scale




        print(f"Final QUBO has {len(self.qubo.get_dict())} terms.")
        return self.qubo

    def solve(self):
        qubo_dict = self.formulate_qubo()
        sample = solve_qubo(qubo_dict, "qaoa")

        formatted_solution = {}
        for key, value in sample.items():
            if value == 1:
                parts = key.split('_')
                if parts[0] == 'x':
                    route_id, drone_id = int(parts[1]), int(parts[2])
                    formatted_solution[route_id] = drone_id

        print(f"Optimized Drone Assignments: {formatted_solution}")
        return formatted_solution



if __name__ == "__main__":
    # Example VRP solution (routes assigned to vehicles initially)
    # routes = [[0, 13, 7, 0], [0, 2, 10, 0], [0, 3, 19, 8, 0], [0, 15, 12, 0], [0, 16, 0], [0, 11, 4, 0], [0, 1, 0], [0, 6, 0], [0, 21, 5, 0], [0, 9, 18, 0], [0, 14, 17, 0], [0, 20, 0]]
    routes = [[0, 13, 7, 0], [0, 2, 10, 0], [0, 3, 19, 8, 0], [0, 15, 12, 0], [0, 16, 0]]




    problem_path = 'tests/pvrp/p-n22-k2.vrp'
    problem, g = create_pvrp_problem(problem_path)


    for num_drones in [2,3,4]:  # Run for both 2 drones and 3 drones
        print(f"\n**Running Drone Scheduling for {num_drones} Drones** \n")

        # Loop through 10 different penalty values
        for i in range(2):

            # # Initialize the scheduler with new penalties
            drone_scheduler = DroneQUBOScheduler(routes, problem.costs, num_drones)
            


            # Solve QUBO problem
            solution = drone_scheduler.solve()

            # Print the assigned routes per drone
            drone_assignments = {i: [] for i in range(num_drones)}
            total_routes = 0  # Track total number of assigned routes

            for route, drone in solution.items():
                drone_assignments[drone].append(routes[route])
                total_routes += 1  # Increment for each assigned route

            print("\n🛠 **Drone Scheduling Results:**")
            for drone, assigned_routes in drone_assignments.items():
                total_load = sum(sum(problem.costs[route[i]][route[i+1]] for i in range(len(route)-1)) for route in assigned_routes)
                print(f"🚁 **Drone {drone}:** Routes {assigned_routes}, **Total Load = {total_load}**")

            print(f"\n📊 **Total Routes Assigned in Iteration {i+1}: {total_routes}**")

        print(f"\n✅ **Completed Runs for {num_drones} Drones** ✅\n") 


# if __name__ == "__main__":
#     # Full VRP routes pool
#     routes_full = [
#         [0, 13, 7, 0], [0, 2, 10, 0], [0, 3, 19, 8, 0], [0, 15, 12, 0],
#         [0, 16, 0], [0, 11, 4, 0], [0, 1, 0], [0, 6, 0],
#         [0, 21, 5, 0], [0, 9, 18, 0], [0, 14, 17, 0], [0, 20, 0]
#     ]

#     problem_path = r'C:\Users\darre\Desktop\Quantum Research\Local_Drones\D-Wave-VRP-localsearch\tests\pvrp\p-n22-k2.vrp'
#     problem, g = create_pvrp_problem(problem_path)

#     MAX_TERMS = 90          # QUBO term limit
#     TRIALS_PER_SETTING = 3  # repeat each (drones, routes) setting
#     DRONES_START = 2
#     DRONES_END = min(6, len(routes_full))  # cap if you want

#     def route_cost(route, costs):
#         return sum(costs[route[i]][route[i+1]] for i in range(len(route)-1))

#     def compute_loads(assignments, routes, costs, num_drones):
#         # assignments: dict route_idx -> drone_idx
#         loads = [0.0] * num_drones
#         for r_idx, d_idx in assignments.items():
#             loads[d_idx] += route_cost(routes[r_idx], costs)
#         return loads

#     def parse_solution_to_assignments(sample):
#         """sample is dict like {'x_0_1': 1, 'x_1_0': 1, ...} -> {route_id: drone_id} for all variables == 1."""
#         assign = {}
#         for key, val in sample.items():
#             if not val:
#                 continue
#             parts = str(key).split('_')
#             if len(parts) == 3 and parts[0] == 'x':
#                 try:
#                     r = int(parts[1]); d = int(parts[2])
#                     # If multiple drones light up for one route, last one wins; we’ll measure violations separately.
#                     assign[r] = d
#                 except:
#                     pass
#         return assign

#     def assignment_violations(sample, num_routes, num_drones):
#         """Returns (unassigned_count, overassigned_count)."""
#         per_route_counts = [0] * num_routes
#         for key, val in sample.items():
#             if not val:
#                 continue
#             parts = str(key).split('_')
#             if len(parts) == 3 and parts[0] == 'x':
#                 r = int(parts[1])
#                 per_route_counts[r] += 1
#         unassigned = sum(1 for c in per_route_counts if c == 0)
#         overassigned = sum(1 for c in per_route_counts if c > 1)
#         return unassigned, overassigned

#     for num_drones in range(DRONES_START, DRONES_END + 1):
#         # routes count starts at current #drones (per your spec: 2/2, 2/3,... then 3/3, 3/4, ...)
#         start_routes = max(2, num_drones)
#         for num_routes in range(start_routes, len(routes_full) + 1):
#             routes = routes_full[:num_routes]

#             # Build QUBO once to check term count (don’t solve yet)
#             sched_probe = DroneQUBOScheduler(routes, problem.costs, num_drones)
#             qubo_probe = sched_probe.formulate_qubo()
#             terms_count = len(qubo_probe.get_dict())

#             if terms_count > MAX_TERMS:
#                 print(f"\n⛔ QUBO for {num_drones} drones / {num_routes} routes has {terms_count} terms (> {MAX_TERMS}).")
#                 print("➡️  Stopping route growth for this drone count and moving to the next number of drones.\n")
#                 break

#             print(f"\n**Running {TRIALS_PER_SETTING} Trials: {num_drones} Drones, {num_routes} Routes**")
#             print(f"QUBO terms: {terms_count}")

#             makespans = []
#             balances = []   # std dev of loads
#             coverages = []  # fraction of routes with exactly one assignment
#             per_trial_details = []

#             for trial in range(1, TRIALS_PER_SETTING + 1):
#                 # Build a fresh scheduler & QUBO for each trial (same instance is fine too)
#                 scheduler = DroneQUBOScheduler(routes, problem.costs, num_drones)
#                 qubo = scheduler.formulate_qubo()

#                 # Solve via QAOA
#                 sample = solve_qubo(qubo, solver="qaoa", shots=512)

#                 # Parse assignment
#                 assignments = parse_solution_to_assignments(sample)

#                 # Compute coverage diagnostics
#                 unassigned, overassigned = assignment_violations(sample, num_routes, num_drones)
#                 exactly_once = num_routes - (unassigned + overassigned)
#                 coverage = exactly_once / float(num_routes) if num_routes > 0 else 1.0
#                 coverages.append(coverage)

#                 # Build printed assignment lists & loads
#                 drone_assignments = {i: [] for i in range(num_drones)}
#                 for r_idx, d_idx in assignments.items():
#                     if 0 <= d_idx < num_drones and 0 <= r_idx < num_routes:
#                         drone_assignments[d_idx].append(routes[r_idx])

#                 loads = compute_loads(assignments, routes, problem.costs, num_drones)
#                 makespan = max(loads) if loads else 0.0
#                 makespans.append(makespan)
#                 # population std dev (or sample)—use population here
#                 mu = sum(loads) / num_drones if num_drones > 0 else 0.0
#                 variance = sum((L - mu) ** 2 for L in loads) / (num_drones if num_drones > 0 else 1)
#                 balances.append(variance ** 0.5)

#                 # Print trial details (similar to your existing output)
#                 print(f"\n🔄 Trial {trial}")
#                 print(f"Optimized Drone Assignments (route→drone): {assignments}")
#                 print("\n🛠 **Drone Scheduling Results:**")
#                 for d in range(num_drones):
#                     total_load = sum(route_cost(rte, problem.costs) for rte in drone_assignments[d])
#                     print(f"🚁 **Drone {d}:** Routes {drone_assignments[d]}, **Total Load = {total_load}**")
#                 print(f"📦 Coverage: {exactly_once}/{num_routes} routes assigned exactly once "
#                       f"(unassigned={unassigned}, overassigned={overassigned})")
#                 print(f"⏱️ Makespan: {makespan:.4f} | Load StdDev: {balances[-1]:.4f}")

#                 per_trial_details.append({
#                     "assignments": assignments,
#                     "loads": loads,
#                     "makespan": makespan,
#                     "coverage": coverage,
#                     "unassigned": unassigned,
#                     "overassigned": overassigned
#                 })

#             # Summary for this (num_drones, num_routes)
#             avg_makespan = sum(makespans) / len(makespans) if makespans else 0.0
#             avg_balance = sum(balances) / len(balances) if balances else 0.0
#             avg_coverage = 100.0 * (sum(coverages) / len(coverages)) if coverages else 0.0

#             print("\n📊 **Summary for Setting**")
#             print(f"🧮 Drones = {num_drones}, Routes = {num_routes}, QUBO terms = {terms_count}")
#             print(f"⏱️ Avg Makespan over {TRIALS_PER_SETTING} trials: {avg_makespan:.4f}")
#             print(f"📈 Avg Load StdDev: {avg_balance:.4f}")
#             print(f"✅ Avg Exact-Assignment Coverage: {avg_coverage:.1f}%")
#             print(f"📝 Trials detail (makespan per trial): {[round(m, 4) for m in makespans]}")
#             print(f"\n✅ **Completed Runs for {num_drones} Drones / {num_routes} Routes** ✅\n")
