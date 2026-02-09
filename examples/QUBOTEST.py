import sys
import os
import numpy as np
from collections import defaultdict

# Add the src directory to the Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src_dir, 'src'))
from input_CMT_dataset import create_pvrp_problem
from qubo_helper_drone_schedule import Qubo
from QiskitSolversDroneSchedule import solve_qubo
import numpy as np
import math

# Drone Scheduler Class
class DroneQUBOScheduler:
    def __init__(self, routes, costs, num_drones):
        self.routes = routes
        self.num_routes = len(routes)
        self.costs = costs
        self.num_drones = num_drones

        self.qubo = Qubo()
        self.A = 500  # Reduced to avoid excessive penalties
        self.B = 30  # Lower balancing penalty to limit QUBO size
        self.C = 350  # Makespan penalty

    def compute_route_costs(self):
        return [sum(self.costs[route[i]][route[i+1]] for i in range(len(route)-1)) for route in self.routes]

    def formulate_qubo(self):
        print(f"Formulating QUBO for {self.num_drones} drones and {self.num_routes} routes")
        route_costs = self.compute_route_costs()

        x = {(i, alpha): f"x_{i}_{alpha}" for i in range(self.num_routes) for alpha in range(self.num_drones)}
        
        for i in range(self.num_routes):
            self.qubo.add_only_one_constraint([x[i, alpha] for alpha in range(self.num_drones)], self.A)
        
        for alpha in range(self.num_drones):
            self.qubo.add_only_one_constraint([x[i, alpha] for i in range(self.num_routes)], self.A // 2)
        
        makespan = "M_max"
        for alpha in range(self.num_drones):
            for i in range(self.num_routes):
                self.qubo.add((makespan, x[i, alpha]), -self.C * route_costs[i])

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


def adjust_penalties(num_drones, num_routes):
    """
    Dynamically adjusts A, B, and C based on the number of drones and routes.
    
    Args:
    - num_drones (int): Number of drones used in the VRP solution.
    - num_routes (int): Number of routes to be assigned.

    Returns:
    - (tuple): Adjusted values of A, B, and C.
    """
    
    # Base values from best observations
    base_A = 666
    base_B = 39
    base_C = 435

    # Adjust A: Higher with more routes, lower with more drones
    A = base_A + int(5 * math.log(num_routes + 1)) - int(10 * math.log(num_drones + 1))

    # Adjust B: Lower with more routes (less penalty needed)
    B = base_B - int(2 * math.log(num_routes + 1)) + int(3 * math.log(num_drones + 1))
    B = max(5, B)  # Ensure B stays above a reasonable minimum

    # Adjust C: Higher if fewer drones (to prevent overload)
    C = base_C + int(10 * math.log(num_routes + 1)) - int(15 * math.log(num_drones + 1))
    C = max(300, C)  # Ensure C stays above a reasonable minimum

    return A, B, C


# if __name__ == "__main__":
#     # Example VRP solution (routes assigned to vehicles initially)
#     # routes = [[0, 13, 7, 0], [0, 2, 10, 0], [0, 3, 19, 8, 0], [0, 15, 12, 0], [0, 16, 0], [0, 11, 4, 0], [0, 1, 0], [0, 6, 0], [0, 21, 5, 0], [0, 9, 18, 0], [0, 14, 17, 0], [0, 20, 0]]

#     routes = [[0, 13, 7, 0], [0, 2, 10, 0], [0, 3, 19, 8, 0], [0, 15, 12, 0]]




#     problem_path = r'C:\Users\darre\Desktop\Quantum Research\Local_Drones\D-Wave-VRP-localsearch\tests\pvrp\p-n22-k2.vrp'
#     problem, g = create_pvrp_problem(problem_path)


#     for num_drones in [4,4,4]:  # Run for both 2 drones and 3 drones
#         print(f"\n**Running Drone Scheduling for {num_drones} Drones** \n")

#         # Loop through 10 different penalty values
#         for i in range(1):
#             # Generate random penalties between 300 and 700 for A, C and between 5 and 40 for B
#             # A = random.randint(650, 680)
#             # B = random.randint(35, 40)
#             # C = random.randint(400, 500)
#             A, B, C = adjust_penalties(num_drones, len(routes))
#             # A,B,C = 666, 38, 439

#             print(f"\n🔄 **Iteration {i+1}: Trying A={A}, B={B}, C={C}**")

#             # Initialize the scheduler with new penalties
#             drone_scheduler = DroneQUBOScheduler(routes, problem.costs, num_drones)
            
#             # Override penalty values
#             drone_scheduler.A = A
#             drone_scheduler.B = B
#             drone_scheduler.C = C

#             # Solve QUBO problem
#             solution = drone_scheduler.solve()

#             # Print the assigned routes per drone
#             drone_assignments = {i: [] for i in range(num_drones)}
#             total_routes = 0  # Track total number of assigned routes

#             for route, drone in solution.items():
#                 drone_assignments[drone].append(routes[route])
#                 total_routes += 1  # Increment for each assigned route

#             print("\n🛠 **Drone Scheduling Results:**")
#             for drone, assigned_routes in drone_assignments.items():
#                 total_load = sum(sum(problem.costs[route[i]][route[i+1]] for i in range(len(route)-1)) for route in assigned_routes)
#                 print(f"🚁 **Drone {drone}:** Routes {assigned_routes}, **Total Load = {total_load}**")

#             print(f"\n📊 **Total Routes Assigned in Iteration {i+1}: {total_routes}**")

#         print(f"\n✅ **Completed Runs for {num_drones} Drones** ✅\n") 

if __name__ == "__main__":
    # Full VRP routes pool
    routes_full = [
        [0, 13, 7, 0], [0, 2, 10, 0], [0, 3, 19, 8, 0], [0, 15, 12, 0],
        [0, 16, 0], [0, 11, 4, 0], [0, 1, 0], [0, 6, 0],
        [0, 21, 5, 0], [0, 9, 18, 0], [0, 14, 17, 0], [0, 20, 0]
    ]

    problem_path = r'C:\Users\darre\Desktop\Quantum Research\Local_Drones\D-Wave-VRP-localsearch\tests\pvrp\p-n22-k2.vrp'
    problem, g = create_pvrp_problem(problem_path)

    MAX_TERMS = 90          # QUBO term limit
    TRIALS_PER_SETTING = 3  # repeat each (drones, routes) setting
    DRONES_START = 2
    DRONES_END = min(6, len(routes_full))  # cap if you want

    def route_cost(route, costs):
        return sum(costs[route[i]][route[i+1]] for i in range(len(route)-1))

    def compute_loads(assignments, routes, costs, num_drones):
        # assignments: dict route_idx -> drone_idx
        loads = [0.0] * num_drones
        for r_idx, d_idx in assignments.items():
            loads[d_idx] += route_cost(routes[r_idx], costs)
        return loads

    def parse_solution_to_assignments(sample):
        """sample is dict like {'x_0_1': 1, 'x_1_0': 1, ...} -> {route_id: drone_id} for all variables == 1."""
        assign = {}
        for key, val in sample.items():
            if not val:
                continue
            parts = str(key).split('_')
            if len(parts) == 3 and parts[0] == 'x':
                try:
                    r = int(parts[1]); d = int(parts[2])
                    # If multiple drones light up for one route, last one wins; we’ll measure violations separately.
                    assign[r] = d
                except:
                    pass
        return assign

    def assignment_violations(sample, num_routes, num_drones):
        """Returns (unassigned_count, overassigned_count)."""
        per_route_counts = [0] * num_routes
        for key, val in sample.items():
            if not val:
                continue
            parts = str(key).split('_')
            if len(parts) == 3 and parts[0] == 'x':
                r = int(parts[1])
                per_route_counts[r] += 1
        unassigned = sum(1 for c in per_route_counts if c == 0)
        overassigned = sum(1 for c in per_route_counts if c > 1)
        return unassigned, overassigned

    for num_drones in range(DRONES_START, DRONES_END + 1):
        # routes count starts at current #drones (per your spec: 2/2, 2/3,... then 3/3, 3/4, ...)
        start_routes = max(2, num_drones)
        for num_routes in range(start_routes, len(routes_full) + 1):
            routes = routes_full[:num_routes]

            # Build QUBO once to check term count (don’t solve yet)
            sched_probe = DroneQUBOScheduler(routes, problem.costs, num_drones)
            qubo_probe = sched_probe.formulate_qubo()
            terms_count = len(qubo_probe.get_dict())

            if terms_count > MAX_TERMS:
                print(f"\n⛔ QUBO for {num_drones} drones / {num_routes} routes has {terms_count} terms (> {MAX_TERMS}).")
                print("➡️  Stopping route growth for this drone count and moving to the next number of drones.\n")
                break

            print(f"\n**Running {TRIALS_PER_SETTING} Trials: {num_drones} Drones, {num_routes} Routes**")
            print(f"QUBO terms: {terms_count}")

            makespans = []
            balances = []   # std dev of loads
            coverages = []  # fraction of routes with exactly one assignment
            per_trial_details = []

            for trial in range(1, TRIALS_PER_SETTING + 1):
                # Build a fresh scheduler & QUBO for each trial (same instance is fine too)
                scheduler = DroneQUBOScheduler(routes, problem.costs, num_drones)
                qubo = scheduler.formulate_qubo()

                # Solve via QAOA
                sample = solve_qubo(qubo, solver="qaoa", shots=512)

                # Parse assignment
                assignments = parse_solution_to_assignments(sample)

                # Compute coverage diagnostics
                unassigned, overassigned = assignment_violations(sample, num_routes, num_drones)
                exactly_once = num_routes - (unassigned + overassigned)
                coverage = exactly_once / float(num_routes) if num_routes > 0 else 1.0
                coverages.append(coverage)

                # Build printed assignment lists & loads
                drone_assignments = {i: [] for i in range(num_drones)}
                for r_idx, d_idx in assignments.items():
                    if 0 <= d_idx < num_drones and 0 <= r_idx < num_routes:
                        drone_assignments[d_idx].append(routes[r_idx])

                loads = compute_loads(assignments, routes, problem.costs, num_drones)
                makespan = max(loads) if loads else 0.0
                makespans.append(makespan)
                # population std dev (or sample)—use population here
                mu = sum(loads) / num_drones if num_drones > 0 else 0.0
                variance = sum((L - mu) ** 2 for L in loads) / (num_drones if num_drones > 0 else 1)
                balances.append(variance ** 0.5)

                # Print trial details (similar to your existing output)
                print(f"\n🔄 Trial {trial}")
                print(f"Optimized Drone Assignments (route→drone): {assignments}")
                print("\n🛠 **Drone Scheduling Results:**")
                for d in range(num_drones):
                    total_load = sum(route_cost(rte, problem.costs) for rte in drone_assignments[d])
                    print(f"🚁 **Drone {d}:** Routes {drone_assignments[d]}, **Total Load = {total_load}**")
                print(f"📦 Coverage: {exactly_once}/{num_routes} routes assigned exactly once "
                      f"(unassigned={unassigned}, overassigned={overassigned})")
                print(f"⏱️ Makespan: {makespan:.4f} | Load StdDev: {balances[-1]:.4f}")

                per_trial_details.append({
                    "assignments": assignments,
                    "loads": loads,
                    "makespan": makespan,
                    "coverage": coverage,
                    "unassigned": unassigned,
                    "overassigned": overassigned
                })

            # Summary for this (num_drones, num_routes)
            avg_makespan = sum(makespans) / len(makespans) if makespans else 0.0
            avg_balance = sum(balances) / len(balances) if balances else 0.0
            avg_coverage = 100.0 * (sum(coverages) / len(coverages)) if coverages else 0.0

            print("\n📊 **Summary for Setting**")
            print(f"🧮 Drones = {num_drones}, Routes = {num_routes}, QUBO terms = {terms_count}")
            print(f"⏱️ Avg Makespan over {TRIALS_PER_SETTING} trials: {avg_makespan:.4f}")
            print(f"📈 Avg Load StdDev: {avg_balance:.4f}")
            print(f"✅ Avg Exact-Assignment Coverage: {avg_coverage:.1f}%")
            print(f"📝 Trials detail (makespan per trial): {[round(m, 4) for m in makespans]}")
            print(f"\n✅ **Completed Runs for {num_drones} Drones / {num_routes} Routes** ✅\n")

# class DroneQUBOScheduler:
#     def __init__(self, routes, costs, num_drones, max_time=None):
#         """
#         Initializes QUBO formulation for drone job scheduling.

#         :param routes: List of vehicle routes from VRP solution.
#         :param costs: Cost matrix representing travel times/distances.
#         :param num_drones: Number of available drones.
#         """
#         self.routes = routes
#         self.num_routes = len(routes)
#         self.costs = costs
#         self.num_drones = num_drones

#         self.qubo = Qubo()  # QUBO formulation
#         self.A = 100  # Penalty for assignment constraint
#         self.B = 50  # Penalty for balancing workload

#     def compute_route_costs(self):
#         """
#         Computes the cost (travel time) of each route based on cost matrix.
#         """
#         route_costs = []
#         for route in self.routes:
#             total_cost = sum(self.costs[route[i]][route[i+1]] for i in range(len(route)-1))
#             route_costs.append(total_cost)
#         return route_costs

#     def formulate_qubo(self):
#         """
#         Constructs the QUBO problem for drone scheduling.
#         """
#         print(f"🔹 Formulating QUBO for {self.num_drones} drones with {self.num_routes} routes")
#         route_costs = self.compute_route_costs()

#         # Define binary variables: x_{i, α} (1 if route i is assigned to drone α)
#         x = {}
#         for i in range(self.num_routes):
#             for alpha in range(self.num_drones):
#                 x[(i, alpha)] = f"x_{i}_{alpha}"

#         # Constraint 1: Each route must be assigned to exactly one drone
#         for i in range(self.num_routes):
#             self.qubo.add_only_one_constraint([x[i, alpha] for alpha in range(self.num_drones)], self.A)

#         # Constraint 2: Ensure every drone gets at least one route
#         for alpha in range(self.num_drones):
#             self.qubo.add_only_one_constraint([x[i, alpha] for i in range(self.num_routes)], self.A // 2)

#         # Workload balancing constraint: Ensure workload per drone is even
#         for alpha in range(self.num_drones):
#             for i in range(self.num_routes):
#                 self.qubo.add((x[i, alpha], x[i, alpha]), route_costs[i] ** 2)
#                 if alpha > 0:
#                     self.qubo.add((x[i, alpha], x[i, 0]), -1 * route_costs[i])  # Penalize uneven workloads

#         # Objective function: Minimize the maximum workload (M_1)
#         for alpha in range(1, self.num_drones):
#             self.qubo.add((x[0, 0], x[0, alpha]), self.B)  # Encourage balanced workloads

#         print(f"Formulated QUBO with {len(self.qubo.get_dict())} terms.")
#         return self.qubo

#     def solve(self, solver_type='cpu'):
#         print(f"Solving QUBO using solver type: {solver_type}")
#         qubo_dict = self.formulate_qubo()
#         sample = solve_qubo(qubo_dict, solver_type)

#         # Convert QUBO output into a readable schedule
#         formatted_solution = {}
#         for key, value in sample.items():
#             if value == 1:  # Only include active assignments
#                 parts = key.split('_')
#                 if parts[0] == 'x':  # Extract route-drone assignment
#                     route_id, drone_id = int(parts[1]), int(parts[2])
#                     formatted_solution[route_id] = drone_id

#         print(f"Solved QUBO with formatted solution: {formatted_solution}")
#         return formatted_solution


# if __name__ == "__main__":
#     # Example VRP solution (routes assigned to vehicles initially)
#     routes = [
#         [0, 5, 14, 0], [0, 2, 6, 0], [0, 10, 3, 8, 9, 0], 
#         [0, 13, 7, 0], [0, 1, 0], [0, 11, 4, 0], [0, 15, 12, 0]
#     ]

#     problem_path = r'C:\Users\darre\Desktop\Quantum Research\Local_Drones\D-Wave-VRP-localsearch\tests\pvrp\p-n16-k8.vrp'
#     problem, g = create_pvrp_problem(problem_path)

#     num_drones = 3

#     # Initialize and solve QUBO for drone job scheduling
#     drone_scheduler = DroneQUBOScheduler(routes, problem.costs, num_drones)
#     solution = drone_scheduler.solve()

#     # Print the assigned routes per drone
#     drone_assignments = {i: [] for i in range(num_drones)}
#     for route, drone in solution.items():
#         drone_assignments[drone].append(routes[route])

#     print("\n🛠 Drone Scheduling Results:")
#     for drone, assigned_routes in drone_assignments.items():
#         print(f"Drone {drone}: Routes {assigned_routes}, Total Load = {sum(sum(problem.costs[route[:-1], route[1:]]) for route in assigned_routes)}")