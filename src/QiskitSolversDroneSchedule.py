import time
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def solve_qubo(qubo, solver="qaoa", shots=1):
    """
    Solves a QUBO problem using QAOA (GPU/CPU), Exact Solver (CPU), or Simulated Annealing (CPU).

    Args:
        qubo (QuadraticProgram): The QUBO problem.
        solver (str): Choose between "qaoa" (GPU-based), "exact" (Classical), or "sa" (Simulated Annealing).
        shots (int): Number of shots for the quantum algorithm.

    Returns:
        dict: Solution mapping variable names to their optimized values.
    """
    
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

    start_time = time.time()  #Start timer

    if solver == "qaoa":
        try:
            #  Use CPU instead of GPU for debugging
            backend = AerSimulator()
            print("🔹 Running QAOA Solver (GPU)...")
        except MemoryError:
            print("CPU memory exceeded, switching to default CPU...")
            backend = AerSimulator(device="CPU")

        #  Configure BackendSampler with `shots`
        sampler = BackendSampler(backend=backend, options={"shots": shots})

        #  QAOA solver with SPSA optimizer
        qaoa_mes = QAOA(sampler=sampler, optimizer=SPSA(), reps=1)
        optimizer = MinimumEigenOptimizer(qaoa_mes)

    elif solver == "exact":
        #  Classical Exact Solver (CPU)
        print("Running Exact Solver (Classical CPU)...")
        exact_mes = NumPyMinimumEigensolver()
        optimizer = MinimumEigenOptimizer(exact_mes)


    else:
        raise ValueError("Invalid solver type. Choose 'qaoa' (GPU), 'exact' (CPU), or 'sa' (CPU).")

    result = optimizer.solve(model)
    sample = {index: result.variables_dict[str(index)] for index in var_names}

    end_time = time.time()  #  End timer
    print(f"{solver.upper()} Solver completed in {end_time - start_time:.2f} seconds.")

    return sample