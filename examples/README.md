# QUBO Vehicle Routing Problem (VRP) Setup Guide

## **File Placement Instructions**

* Replace the current versions of `QiskitSolvers.py` and `qubo_helper.py` in the `src` folder with the ones provided.
* Place `QUBOTEST.py` in the `examples` folder.
* Place the attached `p-n16-k8.vrp` file in your code directory.

## **Running the Code**

* Navigate to the `examples` folder:

    ```bash
    cd examples
    ```

* Run the script:

    ```bash
    python QUBOTEST.py
    ```

### **Modifications in `QUBOTEST.py`**

* Modify the solver type for CPU or GPU execution (Line 55).
    * Change the `"exact"` parameter to select execution mode:

        ```python
        sample = solve_qubo(qubo_dict, "exact")  # Change "exact" to:
        # "exact" for CPU execution
        # "qaoa" for GPU execution (requires a properly set up CUDA environment)
        ```

* Update the problem path to the location of `p-n16-k8.vrp` (Line 108).
    * Modify the file path accordingly:

        ```python
        problem_path = "path/to/p-n16-k8.vrp"  # Update this path accordingly
        ```

* Adjust the number of drones to test with (Line 112).
    * Modify the list to test different drone counts:

        ```python
        for num_drones in [4]:  # Modify this list to test different numbers of drones
        # Example: for num_drones in [2, 3]  # Runs the code for 2 and 3 drones
        ```

### **Notes**

* Ensure that the CUDA environment is set up correctly before running with GPU acceleration (`qaoa` mode).
* If you encounter any path issues, double-check the `problem_path` on line 108.
* Modify the drone count in line 112 to test various scenarios.