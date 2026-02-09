import torch
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
from concurrent.futures import ThreadPoolExecutor
import time
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram



# def run_qaoa_on_quantum(qubo, shots=256):
#     """
#     Runs QAOA on a quantum simulator (AerSimulator with statevector or qasm backend).
#     """
#     backend = AerSimulator(method="matrix_product_state")  
#     sampler = BackendSampler(backend=backend, options={"shots": shots})

   
#     qaoa_solver = QAOA(sampler=sampler, optimizer=SPSA(), reps=3)
#     optimizer = MinimumEigenOptimizer(qaoa_solver)

#     return optimizer.solve(qubo)

# def solve_qubo(qubo, num_drones, shots=1024):
#     """
#     Solves QUBO using QAOA on a quantum simulator with batch processing.
#     """
#     X = qubo.get_dict()
#     model = QuadraticProgram("qubo")

#     var_names = set()
#     quadratic = {}

#     for (x, y), value in X.items():
#         if x != "bias" and y != "bias":  # Ensure "bias" is not treated as a variable
#             var_names.add(x)
#             var_names.add(y)
#             quadratic[(str(x), str(y))] = value

#     var_names = list(var_names)  

#     for var_name in var_names:
#         model.binary_var(name=str(var_name)) 

#     # Set the objective function with validated variables
#     try:
#         model.minimize(quadratic=quadratic)
#     except KeyError as e:
#         print(f" ERROR: Missing variable in QuadraticProgram: {e}")
#         print("Possible Fix: Ensure all variables are declared before constraints.")
#         raise  

#     with ThreadPoolExecutor(max_workers=2) as executor:
#         future = executor.submit(run_qaoa_on_quantum, model, shots)
#         result = future.result()

#     return {index: result.variables_dict.get(str(index), 0) for index in var_names}


# from itertools import product, combinations
# from collections import defaultdict

# class Qubo:
#     def __init__(self):
#         self.terms = defaultdict(float)  # Sparse storage for QUBO terms

#     def add(self, field, value):
#         """ Efficiently adds a value to a QUBO field, keeping it sparse. """
#         if value != 0:
#             self.terms[field] += value

#     def add_only_one_constraint(self, variables, const):
#         """
#         Ensures exactly one variable in 'variables' is 1, but in a low-memory way.
#         Uses a weaker but sufficient constraint to enforce uniqueness.
#         """
#         for var in variables:
#             self.add((var, var), -const)  # Strong self-penalty

#         for var1, var2 in product(variables, variables):
#             if var1 < var2:
#                 self.add((var1, var2), 2*const)  # Lower off-diagonal interactions

#     def add_linking(self, linking_var, product_terms, linking_weight):
#         """
#         Placeholder implementation to add linking penalty terms that force:
#             linking_var = ∏_{i=0}^{n-1} term_i
#         where each term in product_terms is either a binary variable (if the expected bit is 1)
#         or a string of the form "(1 - var)" (if the expected bit is 0).
        
#         This function enumerates over possible assignments for a small number of bits and adds
#         a penalty proportional to (linking_var - product_value)^2.
        
#         (This is a placeholder; in a production QUBO you would quadratize the high-order term.)
#         """
#         import itertools
#         n = len(product_terms)
#         for combo in itertools.product([0, 1], repeat=n):
#             prod_val = 1
#             for term, bit in zip(product_terms, combo):
#                 if term.strip().startswith("(1 -"):
#                     prod_val *= (1 - bit)
#                 else:
#                     prod_val *= bit
#             # For binary linking_var, (linking_var - prod_val)^2 expands to:
#             # linking_var - 2*prod_val*linking_var + prod_val.
#             penalty = linking_weight
#             self.add((linking_var, linking_var), penalty * (1 - 2 * prod_val))
#             # Note: Constant terms are omitted.

#     def add_diversity_constraint(self, variables, diversity_weight):
#         """
#         Adds a quadratic penalty enforcing (1 - sum(variables))^2.
#         Expanding, (1 - sum(v))^2 = 1 - 2 sum(v) + sum_{i,j} v_i v_j.
#         Since constant offsets can be dropped, we add:
#           - Linear terms: -2 * diversity_weight * v_i
#           - Quadratic terms: diversity_weight * 2 * v_i v_j for i < j, plus diversity_weight * v_i for each i (from v_i^2 = v_i).
#         """
#         # Add linear terms for each variable.
#         for var in variables:
#             self.add((var, var), -diversity_weight)
#         # Add quadratic terms for each distinct pair.
#         for var1, var2 in combinations(variables, 2):
#             self.add((var1, var2), 2 * diversity_weight)

#     def get_dict(self):
#         """ Returns a sparse dictionary of QUBO terms (avoids storing zeros). """
#         return {k: v for k, v in self.terms.items() if abs(v) > 1e-9}



import random
import sys
import os
import numpy as np
from collections import defaultdict
import math


# Add the src directory to the Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src_dir, 'src'))
from input_CMT_dataset import create_pvrp_problem
from qubo_helper_drone_schedule import Qubo
from QiskitSolversDroneSchedule import solve_qubo
import numpy as np
import math

#import torch
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
from concurrent.futures import ThreadPoolExecutor
import time
import os
import random
import numpy as np
import math
from itertools import product, combinations
from collections import defaultdict

# -----------------------------
# Quantum Solver Functions
# -----------------------------
def run_qaoa_on_quantum(qubo, shots=256, optimizer_choice='SPSA', reps=5):
    """
    Runs QAOA on a quantum simulator (AerSimulator with statevector or qasm backend)
    using the chosen optimizer and number of repetitions.
    """
    backend = AerSimulator(method="matrix_product_state")  
    sampler = BackendSampler(backend=backend, options={"shots": shots})

    if optimizer_choice.upper() == 'COBYLA':
        optimizer = COBYLA()
    else:
        optimizer = SPSA()

    qaoa_solver = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
    optimizer_instance = MinimumEigenOptimizer(qaoa_solver)
    return optimizer_instance.solve(qubo)

def solve_qubo(qubo, shots=1024, optimizer_choice='SPSA', reps=5):
    """
    Solves the QUBO using QAOA on a quantum simulator via a thread.
    Returns the result.variables_dict from the quantum solver.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(run_qaoa_on_quantum, qubo, shots, optimizer_choice, reps)
        result = future.result()
    return result.variables_dict

# -----------------------------
# QUBO Formulation
# -----------------------------
class Qubo:
    def __init__(self):
        self.terms = defaultdict(float)  # Sparse storage for QUBO terms

    def add(self, field, value):
        """Efficiently add a value to a QUBO field (sparse)."""
        if value != 0:
            self.terms[field] += value

    def add_only_one_constraint(self, variables, const):
        """
        Enforces that exactly one variable in 'variables' is 1.
        Uses a diagonal penalty and lower off-diagonals.
        """
        for var in variables:
            self.add((var, var), -const)
        for var1, var2 in product(variables, variables):
            if var1 < var2:
                self.add((var1, var2), 2 * const)

    def add_linking(self, linking_var, product_terms, linking_weight):
        """
        Placeholder for linking constraint: forcing linking_var = ∏ term_i.
        (For production use, one should quadratize the higher-order term.)
        """
        import itertools
        n = len(product_terms)
        for combo in itertools.product([0, 1], repeat=n):
            prod_val = 1
            for term, bit in zip(product_terms, combo):
                if term.strip().startswith("(1 -"):
                    prod_val *= (1 - bit)
                else:
                    prod_val *= bit
            self.add((linking_var, linking_var), linking_weight * (1 - 2 * prod_val))

    def add_diversity_constraint(self, variables, diversity_weight):
        """
        Adds a quadratic penalty enforcing (1 - sum(variables))^2.
        """
        for var in variables:
            self.add((var, var), -diversity_weight)
        for var1, var2 in combinations(variables, 2):
            self.add((var1, var2), 2 * diversity_weight)

    def get_dict(self):
        """Returns a sparse dictionary of QUBO terms (dropping near zeros)."""
        return {k: v for k, v in self.terms.items() if abs(v) > 1e-9}

# -----------------------------
# Hybrid Drone QUBO Scheduler
# -----------------------------
class DroneQUBOScheduler:
    def __init__(self, routes, costs, num_drones):
        self.routes = routes
        self.num_routes = len(routes)
        self.costs = costs
        self.num_drones = num_drones
        self.qubo = Qubo()
        self.A, self.B = self.adjust_penalties()
        self.x = {}  # Will store binary variable mapping for each route

    def compute_route_costs(self):
        return [sum(self.costs[route[i]][route[i+1]] for i in range(len(route)-1))
                for route in self.routes]

    
    def adjust_penalties(self):
        # Compute route costs and set penalties dynamically.
        route_costs = self.compute_route_costs()
        C_max = max(route_costs)
        L_avg = sum(route_costs) / self.num_drones
        constraint_penalty = self.num_drones * C_max  # minimum safe value
        penalty_weight = (C_max ** 2) / (L_avg ** 2)  # balancing term
        return constraint_penalty, penalty_weight

    def formulate_qubo(self, constraint_penalty, penalty_weight):
        # print(f"Formulating QUBO for {self.num_drones} drones and {self.num_routes} routes")
        # Binary encoding: each route gets ceil(log2(num_drones)) bits.
        binary_vars_per_route = math.ceil(math.log2(self.num_drones))
        self.x = {i: [f"x_{i}_{b}" for b in range(binary_vars_per_route)] for i in range(self.num_routes)}
        # print("🔹 Created binary variable mapping for each route.")

        # Enforce that each route is assigned exactly once.
        for i in range(self.num_routes):
            self.qubo.add_only_one_constraint(self.x[i], const=constraint_penalty)

        # Load balancing: apply a penalty based on an initial (naive) grouping.
        avg_routes_per_drone = self.num_routes / self.num_drones
        drone_load = {d: [] for d in range(self.num_drones)}
        for i in range(self.num_routes):
            assigned_drone = i % self.num_drones
            drone_load[assigned_drone].append(self.x[i])
        for d, routes_vars in drone_load.items():
            penalty_term = len(routes_vars) - avg_routes_per_drone
            for route_vars in routes_vars:
                for var in route_vars:
                    self.qubo.add((var, var), penalty_weight * (penalty_term ** 2))

        

        qubo_dict = self.qubo.get_dict()
        # print(f"Final QUBO has {len(qubo_dict)} nonzero terms.")

        # Build a QuadraticProgram from the QUBO dictionary.
        model = QuadraticProgram("qubo")
        var_names = set()
        for (var1, var2) in qubo_dict.keys():
            var_names.add(var1)
            var_names.add(var2)
        for var in var_names:
            model.binary_var(name=str(var))
        try:
            model.minimize(quadratic=qubo_dict)
        except KeyError as e:
            print(f"ERROR: Missing variable in QuadraticProgram: {e}")
            raise
        return model

    def decode_assignment(self, sample):
        """
        Decodes the binary-encoded assignment for each route from the quantum solution.
        Returns a dictionary mapping route index to the computed drone assignment.
        """
        quantum_assignments = {}
        for i in range(self.num_routes):
            bits = self.x[i]
            value = 0
            for j, var in enumerate(bits):
                bit_value = sample.get(var, 0)
                value += bit_value * (2 ** j)
            if value < self.num_drones:
                quantum_assignments[i] = value
        return quantum_assignments

    def solve(self, constraint_penalty, penalty_weight, shots=1024, optimizer_choice='SPSA', reps=5):
        """
        Solves the QUBO using QAOA and then applies a classical correction step.
        Returns the final hybrid assignment along with a breakdown of quantum vs classical assignments.
        """
        qubo_model = self.formulate_qubo(constraint_penalty, penalty_weight)
        sample = solve_qubo(qubo_model, shots=shots, optimizer_choice=optimizer_choice, reps=reps)
        
        # Decode the quantum solution.
        quantum_assignments = self.decode_assignment(sample)
        final_assignments = quantum_assignments.copy()
        classical_assignments = {}  # Record which routes are corrected classically.
        
        # -----------------------------
        # Step 1: Assign Unassigned Routes
        # -----------------------------
        # If any route is missing a quantum assignment, assign it to the least-loaded drone.
        unassigned_routes = set(range(self.num_routes)) - set(quantum_assignments.keys())
        for route in unassigned_routes:
            # Calculate the current load for each drone.
            current_load = {d: list(final_assignments.values()).count(d) for d in range(self.num_drones)}
            least_loaded_drone = min(current_load, key=current_load.get)
            final_assignments[route] = least_loaded_drone
            classical_assignments[route] = f"Assigned as unassigned to Drone {least_loaded_drone}"
        
        # -----------------------------
        # Step 2: Ensure Every Drone Has At Least One Route
        # -----------------------------
        # Build a dictionary mapping each drone to its assigned routes.
        drone_routes = {d: [] for d in range(self.num_drones)}
        for route, drone in final_assignments.items():
            drone_routes[drone].append(route)
        
        # Identify drones with zero routes.
        missing_drones = [d for d in range(self.num_drones) if not drone_routes[d]]
        if missing_drones:
            # Calculate current load for all drones.
            current_load = {d: len(drone_routes[d]) for d in range(self.num_drones)}
            for missing_drone in missing_drones:
                # Choose donor: the drone with the most routes.
                donor_drone = max(current_load, key=current_load.get)
                if current_load[donor_drone] == 0:
                    # If even the donor has no routes (unlikely), then break.
                    break
                # Prefer to take a route that was assigned quantumly (to minimally disturb unassigned ones).
                candidate_routes = [r for r in drone_routes[donor_drone] if r in quantum_assignments]
                if not candidate_routes:
                    candidate_routes = drone_routes[donor_drone]
                route_to_reassign = candidate_routes[0]
                final_assignments[route_to_reassign] = missing_drone
                classical_assignments[route_to_reassign] = f"Reassigned from Drone {donor_drone} to Drone {missing_drone} (drone had 0 routes)"
                
                # Update the drone_routes and current_load.
                drone_routes[donor_drone].remove(route_to_reassign)
                drone_routes[missing_drone].append(route_to_reassign)
                current_load[donor_drone] -= 1
                current_load[missing_drone] += 1

        # -----------------------------
        # Print Structured Breakdown
        # -----------------------------
        # print("\n=== Assignment Breakdown ===\n")
        
        # print("Quantum-derived Assignments:")
        # for route, drone in quantum_assignments.items():
        #     print(f"  - Route {route} → Assigned to Drone {int(drone)}")
        
        # print("\nClassical Corrections:")
        # for route, note in classical_assignments.items():
        #     print(f"  - Route {route}: {note}")
        
        # print("\nFinal Hybrid Assignments:")
        # for route, drone in final_assignments.items():
        #     print(f"  - Route {route} → Final Assignment: Drone {int(drone)}")
        
        return final_assignments






if __name__ == "__main__":
    # Example VRP solution (routes assigned to vehicles initially)
#     routes = [
#     [0, 42, 22, 28, 0], 
#     [0, 2, 48, 30, 0], 
#     [0, 3, 24, 49, 33, 0], 
#     [0, 4, 45, 34, 0], 
#     [0, 5, 47, 21, 0], 
#     [0, 6, 31, 10, 0], 
#     [0, 7, 14, 38, 0], 
#     [0, 9, 18, 32, 0], 
#     [0, 11, 53, 35, 0], 
#     [0, 12, 40, 17, 0], 
#     [0, 19, 54, 13, 46, 0], 
#     # [0, 26, 8, 52, 27, 0], 
#     # [0, 29, 15, 20, 37, 36, 0], 
#     # [0, 39, 25, 50, 44, 0], 
#     # [0, 43, 41, 23, 16, 51, 0]
# ]

    routes =   [[0, 2, 6, 0], [0, 11, 4, 0], [0, 5, 14, 0], [0, 10, 3, 8, 9, 0], [0, 7, 13, 0], [0, 15, 12, 0], [0, 1, 0]]

    # pn51
    # routes = [[0, 23, 7, 0], [0, 32, 0], [0, 44, 15, 0], [0, 46, 0], [0, 29, 10, 0], [0, 11, 3, 0], [0, 18, 19, 0], [0, 4, 17, 0], [0, 37, 0], [0, 45, 49, 0], [0, 12, 0], [0, 22, 35, 0], [0, 20, 40, 0], [0, 34, 30, 0], [0, 33, 39, 13, 0], [0, 42, 41, 0], [0, 36, 24, 0], [0, 1, 0], [0, 31, 28, 0], [0, 25, 21, 16, 0], [0, 26, 8, 0], [0, 14, 6, 0], [0, 47, 0], [0, 5, 38, 0], [0, 48, 0], [0, 50, 9, 0], [0, 27, 0], [0, 2, 43, 0]]

    # pn55
    # routes = [[0, 49, 23, 0], [0, 32, 25, 0], [0, 3, 16, 0], [0, 26, 28, 0], [0, 11, 0], [0, 27, 0], [0, 54, 19, 0], [0, 6, 0], [0, 10, 38, 0], [0, 44, 0], [0, 30, 0], [0, 33, 1, 0], [0, 22, 43, 0], [0, 40, 50, 0], [0, 47, 21, 0], [0, 17, 0], [0, 53, 0], [0, 2, 48, 0], [0, 51, 0], [0, 4, 0], [0, 14, 35, 0], [0, 52, 45, 0], [0, 8, 34, 0], [0, 15, 20, 0], [0, 29, 36, 0], [0, 5, 37, 0], [0, 7, 0], [0, 24, 18, 0], [0, 42, 41, 0], [0, 13, 0], [0, 46, 0], [0, 31, 39, 0], [0, 12, 9, 0]]



    problem_path = r'C:\Users\darre\Desktop\Quantum Research\Local_Drones\D-Wave-VRP-localsearch\tests\pvrp\p-n16-k8.vrp'
    problem, g = create_pvrp_problem(problem_path)



    num_trials = 3

    for num_drones in range(2, len(routes) + 1):  # Run from 2 up to number of routes
        # print(f"\n**Running Drone Scheduling for {num_drones} Drones**\n")
        
        makespans = []

        for trial in range(num_trials):
            # print(f"\n-- Trial {trial + 1} for {num_drones} Drones --")

            # Initialize the scheduler with new penalties
            drone_scheduler = DroneQUBOScheduler(routes, problem.costs, num_drones)

            # Solve QUBO problem
            constraint_penalty, penalty_weight = drone_scheduler.adjust_penalties()
            solution = drone_scheduler.solve(constraint_penalty, penalty_weight)

            # print(solution)

            # Step 1: Map routes to drones
            drone_assignments = {i: [] for i in range(num_drones)}
            for route, drone in solution.items():
                drone_assignments[drone].append(routes[route])

            # Step 2: Calculate total load per drone
            drone_loads = []
            # print("\n**Drone Scheduling Results:**")
            for drone, assigned_routes in drone_assignments.items():
                # Compute the load for each drone
                route_costs = [
                    sum(problem.costs[route[i]][route[i+1]] for i in range(len(route) - 1))
                    for route in assigned_routes
                ]
                total_load = sum(route_costs)

                # Add penalty once per drone if drone has more than one route
                penalty = 1.25 * (len(assigned_routes) - 1)
                total_load += penalty

                drone_loads.append(total_load)

                # print(f"**Drone {drone}:** Routes {assigned_routes}, **Total Load = {total_load}**")

            # Step 3: Record makespan (max drone load)
            makespan = max(drone_loads)
            makespans.append(makespan)

            # Step 4: Track total assigned routes
            total_routes = sum(len(routes) for routes in drone_assignments.values())
            # print(f"\n**Total Routes Assigned in Trial {trial + 1}: {total_routes}**")
            # print(f"**Trial {trial + 1} Makespan = {makespan}**")

        # After all trials, print average makespan
        avg_makespan = sum(makespans) / num_trials
        print(f"\n===> Average Makespan for {num_drones} Drones over {num_trials} Trials: {avg_makespan:.2f}\n")
        # print(f"**Completed Runs for {num_drones} Drones**\n{'='*60}")


    # amount_of_drones = 10
    # 2,2,2,3,3,3,4,4,4
    # 5,5,5,6,6,6,7,7,7,8,8,8,9,9,9
    #10, 11, 12
    # for num_drones in [2,2,2,3,3,3,4,4,4]:  # Run for a given number of drones
    #     print(f"\n**Running Drone Scheduling for {num_drones} Drones** \n")

    #     # Initialize the scheduler with new penalties
    #     drone_scheduler = DroneQUBOScheduler(routes, problem.costs, num_drones)

    #     # Solve QUBO problem
    #     # A = random.randint(1, 5000)
    #     # B = random.randint(1, 5000)
    #     # print(f'constraint: {A} penalty_weight: {B}')
    #     constraint_penalty, penalty_weight = drone_scheduler.adjust_penalties()

    #     solution = drone_scheduler.solve(constraint_penalty, penalty_weight)

    #     print(solution)

    #     #  **Step 1: Map routes to drones correctly**
    #     drone_assignments = {i: [] for i in range(num_drones)}
    #     for route, drone in solution.items():
    #         drone_assignments[drone].append(routes[route])  # Ensure correct mapping

    #     #  **Step 2: Print Drone Scheduling Results**
    #     print("\n**Drone Scheduling Results:**")
    #     for drone, assigned_routes in drone_assignments.items():
    #         total_load = sum(
    #             sum(problem.costs[route[i]][route[i+1]] for i in range(len(route)-1)) 
    #             for route in assigned_routes
    #         )
    #         print(f"**Drone {drone}:** Routes {assigned_routes}, **Total Load = {total_load}**")

    #     #  **Step 3: Track total assigned routes**
    #     total_routes = sum(len(routes) for routes in drone_assignments.values())
    #     print(f"\n**Total Routes Assigned in Iteration 1: {total_routes}**")

    #     print(f"\n**Completed Runs for {num_drones} Drones**\n")