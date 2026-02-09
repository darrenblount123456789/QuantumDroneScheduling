import math
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA

class DroneQUBOScheduler:
    def __init__(self, routes, costs, num_drones, A=500, B=30, M=None):
        """
        routes: list of routes, each a list of node indices.
        costs: a 2D numpy array (cost matrix) where costs[i][j] is the cost from node i to j.
        num_drones: number of drones to assign routes.
        A: penalty weight for hard constraints.
        B: weight for the makespan (max load) term.
        M: maximum allowed load difference (if None, computed as num_routes * max(route_cost)).
        """
        self.routes = routes
        self.costs = costs
        self.num_drones = num_drones
        self.num_routes = len(routes)
        self.A = A
        self.B = B
        self.route_costs = self.compute_route_costs()
        # M is related to the maximum auxiliary spin index required (M ≈ N * max(L_i))
        if M is None:
            self.M = int(np.ceil(max(self.route_costs) * self.num_routes))
        else:
            self.M = M

    def compute_route_costs(self):
        """Compute the cost of each route as the sum of edge costs."""
        costs_list = []
        for route in self.routes:
            route_cost = 0
            for i in range(len(route) - 1):
                route_cost += self.costs[route[i]][route[i+1]]
            costs_list.append(route_cost)
        return costs_list

    def formulate_qubo(self):
        """
        Build the QUBO that encodes:
          • Each route i is assigned to exactly one drone: A * (1 - ∑₍α₎ xᵢ,α)².
          • For each drone α (α ≠ 0) the balancing penalty: A * Lᵢ*(xᵢ,α - xᵢ,0)².
          • Auxiliary variables yₙ,α (for α ≠ 0, n = 1…M) with penalty A * n * yₙ,α.
          • Makespan term: B * Lᵢ * xᵢ,0.
        """
        qubo = QuadraticProgram("DroneSchedulingJobSequencing")
        
        # Create x variables for route assignment: x_{i,α} for each route i and drone α.
        x_vars = {}
        for i in range(self.num_routes):
            for alpha in range(self.num_drones):
                var_name = f"x_{i}_{alpha}"
                qubo.binary_var(name=var_name)
                x_vars[(i, alpha)] = var_name
        
        # Create auxiliary y variables for drones other than drone 0.
        y_vars = {}
        for alpha in range(1, self.num_drones):
            for n in range(1, self.M+1):
                var_name = f"y_{n}_{alpha}"
                qubo.binary_var(name=var_name)
                y_vars[(n, alpha)] = var_name

        linear = {}
        quadratic = {}

        # Constraint: Each route i is assigned exactly one drone.
        # Expand: (1 - ∑₍α₎ xᵢ,α)² = 1 - ∑₍α₎ xᵢ,α + 2∑₍α<β₎ xᵢ,α xᵢ,β   (ignoring constant).
        for i in range(self.num_routes):
            for alpha in range(self.num_drones):
                var = x_vars[(i, alpha)]
                linear[var] = linear.get(var, 0) - self.A
            for alpha in range(self.num_drones):
                for beta in range(alpha+1, self.num_drones):
                    key = (x_vars[(i, alpha)], x_vars[(i, beta)])
                    quadratic[key] = quadratic.get(key, 0) + 2 * self.A

        # Load balancing constraint for drones α ≠ 0:
        # For each route i and each drone α ≠ 0, add:
        #   A * Lᵢ * (xᵢ,α - xᵢ,0)² = A * Lᵢ * [xᵢ,α + xᵢ,0 - 2 xᵢ,α xᵢ,0]
        for i in range(self.num_routes):
            L_i = self.route_costs[i]
            for alpha in range(1, self.num_drones):
                var_alpha = x_vars[(i, alpha)]
                var_0 = x_vars[(i, 0)]
                linear[var_alpha] = linear.get(var_alpha, 0) + self.A * L_i
                linear[var_0] = linear.get(var_0, 0) + self.A * L_i
                key = (var_alpha, var_0) if var_alpha < var_0 else (var_0, var_alpha)
                quadratic[key] = quadratic.get(key, 0) - 2 * self.A * L_i

        # Auxiliary variables: for each drone α ≠ 0 and each n = 1,…,M, add penalty A * n * y_{n,α}.
        for alpha in range(1, self.num_drones):
            for n in range(1, self.M+1):
                var = y_vars[(n, alpha)]
                linear[var] = linear.get(var, 0) + self.A * n

        # Makespan objective: Add B * Lᵢ * xᵢ,0 for each route i (drone 0 is designated as the max-load drone).
        for i in range(self.num_routes):
            L_i = self.route_costs[i]
            var_0 = x_vars[(i, 0)]
            linear[var_0] = linear.get(var_0, 0) + self.B * L_i

        # Set the objective in the QUBO as minimization.
        qubo.minimize(linear=linear, quadratic=quadratic)
        return qubo

    def solve(self):
        qubo = self.formulate_qubo()
        
        # Set up the QAOA solver (using AerSimulator as backend)
        simulator = AerSimulator()
        sampler = BackendSampler(simulator)
        qaoa_solver = QAOA(sampler=sampler, optimizer=SPSA(), reps=1)
        optimizer = MinimumEigenOptimizer(qaoa_solver)
        result = optimizer.solve(qubo)

        # Extract the solution for the x variables (ignore auxiliary y variables for assignment)
        # The variables in the QUBO were added in order (first all x variables).
        sol_dict = {var: val for var, val in zip(qubo.variables, result.x)}
        
        route_to_drone = {}
        loads = [0] * self.num_drones
        for i in range(self.num_routes):
            for alpha in range(self.num_drones):
                var_name = f"x_{i}_{alpha}"
                if sol_dict.get(var_name, 0) == 1:
                    route_to_drone[i] = alpha
                    loads[alpha] += self.route_costs[i]

        # Output the solution in a style similar to the first code snippet.
        print("Optimal route assignments (Route -> Drone):")
        for i in range(self.num_routes):
            assigned = route_to_drone.get(i, None)
            print(f"  Route {i} -> Drone {assigned}")
        print(f"Total routes: {self.num_routes}, Drones: {self.num_drones}")
        print("Load per drone:", loads)
        print("Max load:", max(loads), "Min load:", min(loads))
        print("Load imbalance (max - min):", max(loads) - min(loads))
        return route_to_drone

if __name__ == "__main__":
    # Example usage:
    # Create a dummy symmetric cost matrix for a set of nodes (e.g. 0 to 15)
    num_nodes = 16
    np.random.seed(42)
    cost_matrix = np.random.randint(1, 10, size=(num_nodes, num_nodes))
    cost_matrix = (cost_matrix + cost_matrix.T) // 2  # make symmetric
    np.fill_diagonal(cost_matrix, 0)
    
    # Define some example routes (each route is a list of node indices)
    routes = [
        [0, 5, 14, 0],
        [0, 2, 6, 0],
        [0, 10, 3, 8, 9, 0],
        [0, 13, 7, 0],
        [0, 1, 0],
        [0, 11, 4, 0]
    ]
    
    num_drones = 3  # for example, schedule routes over 3 drones
    scheduler = DroneQUBOScheduler(routes, cost_matrix, num_drones, A=500, B=30)
    scheduler.solve()
