import math
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_aer import AerSimulator
from qiskit_optimization import QuadraticProgram



class DroneQUBOScheduler:
    def __init__(self, routes, costs, num_drones):
        self.routes = routes
        self.num_routes = len(routes)
        self.costs = costs
        self.num_drones = num_drones
        # Use binary encoding: only ceil(log2(num_drones)) variables per route.
        self.binary_vars_per_drone = math.ceil(math.log2(num_drones))
    
    def compute_route_costs(self):
        """Compute the load/cost for each route."""
        return [sum(self.costs[route[i]][route[i+1]] for i in range(len(route)-1))
                for route in self.routes]
    
    def adjust_penalties(self):
        """
        A simple heuristic to set penalty parameters based on route costs.
        (These values typically require tuning.)
        """
        route_costs = self.compute_route_costs()
        C_max = max(route_costs)
        C_total = sum(route_costs)
        L_avg = C_total / self.num_drones
        # For example, set a base penalty for invalid assignments and balancing.
        constraint_penalty = C_max * (C_total / self.num_drones)
        penalty_weight = (C_max / L_avg) * (constraint_penalty / self.num_drones)
        return constraint_penalty, penalty_weight

    def formulate_qubo_qaoa(self, penalty_invalid, diversity_weight):
        """
        Builds a QuadraticProgram representing a memory‑efficient QUBO.
        
        Each route i is assigned a drone by encoding:
            a_i = sum_{b=0}^{k-1} 2^b * x_{i,b},
        with k = ceil(log2(num_drones)).
        
        Two sets of penalty terms are added:
          (A) Invalid assignment penalty: for each route, if the encoded value
              is >= num_drones, add a penalty.
          (B) Diversity penalty: encourage the sum of decoded assignments to be near
              an even-spread target (which is R*(num_drones-1)/2).
        """
        qp = QuadraticProgram("DroneScheduling")
        R = self.num_routes
        D = self.num_drones
        k = self.binary_vars_per_drone
        
        # (1) Create binary variables: x_i_b for each route i and bit b.
        for i in range(R):
            for b in range(k):
                qp.binary_var(name=f"x_{i}_{b}")
        
        # Build the objective from our penalty terms.
        # We'll accumulate linear and quadratic terms.
        linear = {}
        quadratic = {}
        
        # (A) Invalid assignment penalty.
        # For each route i, if a_i (decoded as sum_{b}2^b * x_{i,b}) is >= D,
        # then for each invalid number v, add a penalty.
        # We use a simple scheme: for each v in {D, ..., 2^k - 1}, for each bit, add:
        #    If the bit should be 1 (in pattern for v): add penalty_invalid * (-2) * x_i_b.
        #    If the bit should be 0: add penalty_invalid * (+1) * x_i_b.
        for i in range(R):
            for v in range(D, 2**k):
                pattern = format(v, f"0{k}b")
                for b in range(k):
                    var = f"x_{i}_{b}"
                    bit_val = int(pattern[b])
                    # Update linear term. (Note: this is a heuristic linear penalty.)
                    if bit_val == 1:
                        linear[var] = linear.get(var, 0) + penalty_invalid * (-2)
                    else:
                        linear[var] = linear.get(var, 0) + penalty_invalid
        
        # (B) Diversity penalty.
        # Let decoded assignment for route i be a_i = sum_{b=0}^{k-1} 2^b * x_{i,b}.
        # Then the total S = sum_{i=0}^{R-1} a_i.
        # For evenly distributed assignments, target_sum = R*(D - 1)/2.
        target_sum = (R * (D - 1)) / 2.0
        # Expand (S - target_sum)^2.
        # S = sum_{i,b} 2^b * x_{i}_{b}
        # => (S - target_sum)^2 = S^2 - 2 target_sum S + target_sum^2.
        # We drop the constant target_sum^2.
        # Linear term: for each variable x_{i}_{b}, subtract 2 * diversity_weight * 2^b * target_sum.
        for i in range(R):
            for b in range(k):
                var = f"x_{i}_{b}"
                coef = -2 * diversity_weight * (2 ** b) * (target_sum / R)
                linear[var] = linear.get(var, 0) + coef
        # Quadratic term: for each pair (x_{i}_{b}, x_{j}_{b2}) add diversity_weight * 2^(b+b2).
        for i in range(R):
            for b in range(k):
                var1 = f"x_{i}_{b}"
                for j in range(i, R):
                    for b2 in range(k):
                        var2 = f"x_{j}_{b2}"
                        coef = diversity_weight * (2 ** b) * (2 ** b2)
                        if i == j and b == b2:
                            # When i==j and b==b2, add the self-term.
                            linear[var1] = linear.get(var1, 0) + coef
                        else:
                            # For off-diagonals, add coefficient symmetrically.
                            key = tuple(sorted((var1, var2)))
                            quadratic[key] = quadratic.get(key, 0) + 2 * coef
        
        # Set the objective to be minimization of the total penalty.
        qp.minimize(quadratic=quadratic, linear=linear)
        return qp

    def solve_qaoa(self, penalty_invalid, diversity_weight, reps=1):
        qp = self.formulate_qubo_qaoa(penalty_invalid, diversity_weight)
        # Use QAOA with a simulator backend.
        simulator = AerSimulator()
        # You may choose a classical optimizer and number of repetitions.
        qaoa = QAOA(sampler=simulator, optimizer=SPSA(), reps=reps)

        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(qp)
        decoded_assignments = {}
        k = self.binary_vars_per_drone
        # Decode each route's assignment from the x variables.
        for i in range(self.num_routes):
            a_i = sum(int(result.variables_dict.get(f"x_{i}_{b}", 0)) * (2 ** b)
                      for b in range(k))
            # If the decoded value is invalid, you could choose to default it to 0.
            if a_i >= self.num_drones:
                a_i = 0
            decoded_assignments[i] = a_i
            print(f"Route {i} decoded assignment: {a_i}")
        return decoded_assignments


# -- Main Execution ----------------------------------------------

if __name__ == "__main__":
    # Example VRP solution: each route is a list of location indices.
    routes = [
        [0, 9, 18, 32, 0],
        [0, 11, 53, 35, 0],
        [0, 12, 40, 17, 0],
        [0, 19, 54, 13, 46, 0],
        [0, 26, 8, 52, 27, 0],
        [0, 29, 15, 20, 37, 36, 0],
        [0, 39, 25, 50, 44, 0],
    ]
    # For demonstration, create a dummy cost matrix.
    n_locations = 60
    costs = [[abs(i - j) for j in range(n_locations)] for i in range(n_locations)]
    num_drones = 6
    scheduler = DroneQUBOScheduler(routes, costs, num_drones)
    # Penalty parameters (these must be tuned for your problem)
    penalty_invalid = 1e6
    diversity_weight = 1e6
    solution = scheduler.solve_qaoa(penalty_invalid, diversity_weight, reps=2)
    print("Final solution (route -> drone):", solution)
    
    # Map routes to drones.
    drone_assignments = {d: [] for d in range(num_drones)}
    for route_idx, drone in solution.items():
        if drone in drone_assignments:
            drone_assignments[drone].append(routes[route_idx])
        else:
            print(f"Warning: route {route_idx} assigned invalid drone {drone}")
    
    print("\nDrone Scheduling Results:")
    for d in range(num_drones):
        assigned_routes = drone_assignments[d]
        total_load = 0
        for route in assigned_routes:
            load = sum(costs[route[i]][route[i+1]] for i in range(len(route)-1))
            total_load += load
        print(f"Drone {d}: Routes {assigned_routes}, Total Load = {total_load}")
