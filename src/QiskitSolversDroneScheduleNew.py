import time
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def solve_qubo(qubo, solver="qaoa", shots=512):
    X = qubo.get_dict()
    model = QuadraticProgram("qubo")

    var_names = set()
    quadratic = {}
    for (x, y), value in X.items():
        var_names.add(x)
        var_names.add(y)
        quadratic[(str(x), str(y))] = value

    for var_name in var_names:
        model.binary_var(name=str(var_name))

    model.minimize(quadratic=quadratic)

    start_time = time.time()

    if solver == "qaoa":
        backend = AerSimulator()
        print("🔹 Running QAOA Solver (Aer MPS)...")
        sampler = BackendSampler(backend=backend, options={"shots": shots})

        from qiskit_algorithms.optimizers import SPSA
        qaoa_mes = QAOA(
            sampler=sampler,
            optimizer=SPSA(maxiter=200),
            reps=2
        )
        optimizer = MinimumEigenOptimizer(qaoa_mes)

    elif solver == "exact":
        print("Running Exact Solver (Classical CPU)...")
        exact_mes = NumPyMinimumEigensolver()
        optimizer = MinimumEigenOptimizer(exact_mes)

    else:
        raise ValueError("Invalid solver type.")

    result = optimizer.solve(model)
    sample = {index: result.variables_dict[str(index)] for index in var_names}

    end_time = time.time()
    print(f"{solver.upper()} Solver completed in {end_time - start_time:.2f} seconds.")

    return sample
