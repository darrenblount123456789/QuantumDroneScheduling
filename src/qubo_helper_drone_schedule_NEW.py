from itertools import product
from collections import defaultdict
from itertools import combinations



class Qubo:
    def __init__(self):
        self.terms = defaultdict(float) 
        
    def create_not_exist_field(self, field):
        if field not in self.terms:
            self.terms[field] = 0.0

    def add_only_one_constraint(self, variables, const):
        """Adds a constraint ensuring exactly one variable is 1."""
        # Diagonal: -A * x
        for var in variables:
            self.create_not_exist_field((var, var))
            self.terms[(var, var)] -= const  # -A

        # Pairwise: +2A * x_i x_j
        for var1, var2 in combinations(variables, 2):
            self.create_not_exist_field((var1, var2))
            self.terms[(var1, var2)] += 2.0 * const  # +2A



    def add(self, field, value):
        if value != 0:
            self.terms[field] += value

    def get_dict(self):
        return {k: v for k, v in self.terms.items() if v != 0.0}

