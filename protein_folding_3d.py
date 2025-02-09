import numpy as np
import os
import ctypes

############### LOAD C LIBRARY ###############
# Adjust the path below if necessary
# e.g. if the .so is in the same dir as this .py, use "./compute_energy.so"
# If it's in a 'c_code' subfolder, do "c_code/compute_energy.so"

# Attempt relative path
_libname = os.path.join(os.path.dirname(__file__), "compute_energy.so")
_lib = ctypes.CDLL(_libname)

# C function signature:
#   void compute_energy_and_gradient(const double *positions,
#                                    int n_beads,
#                                    double epsilon,
#                                    double sigma,
#                                    double bond_len,
#                                    double k_bond,
#                                    double *grad_out,
#                                    double *energy_out);
_compute_energy_grad = _lib.compute_energy_and_gradient
_compute_energy_grad.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # positions
    ctypes.c_int,                     # n_beads
    ctypes.c_double,                  # epsilon
    ctypes.c_double,                  # sigma
    ctypes.c_double,                  # bond_len
    ctypes.c_double,                  # k_bond
    ctypes.POINTER(ctypes.c_double),  # grad_out
    ctypes.POINTER(ctypes.c_double)   # energy_out
]
_compute_energy_grad.restype = None

def compute_energy_and_gradient(positions, n_beads,
                                epsilon=1.0, sigma=1.0,
                                bond_len=1.0, k_bond=100.0):
    """
    Python wrapper around the C function compute_energy_and_gradient.

    positions: shape (n_beads, 3) or flattened length 3*n_beads
    returns: (energy, grad) where grad has shape (3*n_beads,)
    """
    x_flat = positions.ravel().astype(np.float64)
    grad_out = np.zeros_like(x_flat, dtype=np.float64)
    energy_out = np.array([0.0], dtype=np.float64)

    # Prepare pointers
    c_pos = x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_grad = grad_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_energy = energy_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    _compute_energy_grad(c_pos,
                         n_beads,
                         ctypes.c_double(epsilon),
                         ctypes.c_double(sigma),
                         ctypes.c_double(bond_len),
                         ctypes.c_double(k_bond),
                         c_grad,
                         c_energy)

    return energy_out[0], grad_out


############### ORIGINAL CODE (TRIMMED) ###############
# Note that we do NOT alter initialize_protein, as per instructions.

def initialize_protein(n_beads, dimension=3, fudge = 1e-5):
    """
    Initialize a protein with `n_beads` arranged almost linearly in `dimension`-dimensional space.
    The `fudge` is a factor that, if non-zero, adds a spiral structure to the configuration.
    """
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1  # distance 1 along x
        positions[i, 1] = fudge * np.sin(i)
        positions[i, 2] = fudge * np.sin(i*i)
    return positions

def total_energy(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Just a convenience Python function that calls our C-based function
    but returns only energy. 
    The autograder might use or ignore this. We'll keep it for checking.
    """
    energy, _ = compute_energy_and_gradient(
        positions.reshape(n_beads, 3),
        n_beads,
        epsilon, sigma,
        b, k_b
    )
    return energy


############### CUSTOM BFGS IMPLEMENTATION ###############
class CustomOptimizeResult:
    """
    Minimal replicate of scipy.optimize.OptimizeResult
    so the autograder can access .x
    """
    def __init__(self, x, success=False, message="", niter=0, fun=None):
        self.x = x
        self.success = success
        self.message = message
        self.nit = niter
        self.fun = fun

def bfgs_minimize(
    x0,
    n_beads,
    maxiter=1000,
    tol=1e-4,
    epsilon=1.0,
    sigma=1.0,
    b=1.0,
    k_b=100.0,
    callback=None
):
    """
    Implements basic BFGS from scratch, with a simple backtracking line search.
    Returns (result, trajectory)
    """
    # x0 = initial guess, shape (3*n_beads,)
    x = x0.copy()
    dim = x.size

    # Evaluate initial energy/gradient
    f, g = compute_energy_and_gradient(x.reshape(n_beads, 3), n_beads, epsilon, sigma, b, k_b)
    gnorm = np.linalg.norm(g)
    if callback is not None:
        callback(x)

    # Identity for Hessian approx
    Hk = np.eye(dim)

    # BFGS iteration
    iteration = 0
    trajectory = [x.reshape(n_beads, 3).copy()]

    while iteration < maxiter and gnorm > tol:
        iteration += 1

        # p = - Hk * g
        p = -Hk.dot(g)

        # line search (backtracking + Armijo)
        alpha = 1.0
        c1 = 1e-4
        old_f = f
        # Simple backtracking Armijo
        while True:
            x_new = x + alpha * p
            f_new, g_new = compute_energy_and_gradient(
                x_new.reshape(n_beads,3), n_beads, epsilon, sigma, b, k_b
            )
            if f_new <= old_f + c1 * alpha * np.dot(g, p):
                # Armijo condition satisfied
                break
            alpha *= 0.5
            if alpha < 1e-16:
                # step too small, break
                break

        # Now we have x_new
        s_k = x_new - x
        y_k = g_new - g

        # BFGS Update
        # H_{k+1} = (I - rho*s_k*y_k^T)*Hk*(I - rho*y_k*s_k^T) + rho*s_k*s_k^T
        # but typically we store Bk^-1 or use direct formula. For simplicity:
        rho = 1.0 / (y_k.dot(s_k) + 1e-16)  # small denom guard
        I = np.eye(dim)
        # rank-1 updates
        A = I - rho * np.outer(s_k, y_k)
        B = I - rho * np.outer(y_k, s_k)
        Hk = A.dot(Hk).dot(B) + rho * np.outer(s_k, s_k)

        # update
        x = x_new
        f = f_new
        g = g_new
        gnorm = np.linalg.norm(g_new)

        # store trajectory
        if callback is not None:
            callback(x)
        trajectory.append(x.reshape(n_beads,3).copy())

        if iteration % 50 == 0:
            print(f"Iteration {iteration}, f={f:.6f}, gnorm={gnorm:.6e}")

    success = (gnorm <= tol)
    message = "Converged" if success else "Max iterations reached"

    # Prepare result
    result = CustomOptimizeResult(
        x = x,
        success=success,
        message=message,
        niter=iteration,
        fun=f
    )
    return result, trajectory

def bfgs_minimize_shaken(
    x0,
    n_beads,
    maxiter=1000,
    tol=1e-6,
    epsilon=1.0,
    sigma=1.0,
    b=1.0,
    k_b=100.0,
    callback=None,
    shake_interval=50,
    shake_scale=0.02,
    reset_hessian_after_shake=True
):
    """
    BFGS optimization with an occasional 'shake' to jump out of local minima.

    Parameters
    ----------
    x0 : np.ndarray
        Initial guess, shape (3*n_beads,).

    n_beads : int
        Number of beads (for passing to compute_energy_and_gradient).

    maxiter : int
        Maximum number of BFGS iterations.

    tol : float
        Convergence tolerance on the gradient norm.

    epsilon, sigma, b, k_b : float
        Lennard-Jones and bond parameters.

    callback : function or None
        If provided, called each iteration with the current x.

    shake_interval : int
        Perform a random shake every `shake_interval` iterations.

    shake_scale : float
        Standard deviation of the random perturbation for each coordinate.

    reset_hessian_after_shake : bool
        If True, re-initialize Hessian approximation to identity after shaking.

    Returns
    -------
    result : CustomOptimizeResult
        Contains final .x, .fun, etc.

    trajectory : list of np.ndarray
        List of all intermediate positions (n_beads, 3).
    """
    dim = x0.size
    x = x0.copy()

    # == Evaluate initial energy and gradient using your C code ==
    f, g = compute_energy_and_gradient(
        x.reshape(n_beads, 3), n_beads, epsilon, sigma, b, k_b
    )
    gnorm = np.linalg.norm(g)

    # Identity matrix for Hessian approximation
    Hk = np.eye(dim)

    iteration = 0
    trajectory = [x.reshape(n_beads, 3).copy()]
    if callback:
        callback(x)

    while iteration < maxiter and gnorm > tol:
        iteration += 1

        # BFGS search direction
        p = -Hk.dot(g)

        # --- Simple backtracking line search (Armijo) ---
        alpha = 1.0
        c1 = 1e-4
        old_f = f
        while True:
            x_new = x + alpha * p
            f_new, g_new = compute_energy_and_gradient(
                x_new.reshape(n_beads,3), n_beads, epsilon, sigma, b, k_b
            )
            if f_new <= old_f + c1 * alpha * np.dot(g, p):
                # Armijo condition satisfied
                break
            alpha *= 0.5
            if alpha < 1e-16:
                # step size too small, break to avoid infinite loop
                break

        # --- BFGS Update ---
        s_k = x_new - x          # displacement
        y_k = g_new - g          # grad change
        rho = 1.0 / (np.dot(y_k, s_k) + 1e-16)

        I = np.eye(dim)
        A = I - rho * np.outer(s_k, y_k)
        B = I - rho * np.outer(y_k, s_k)
        Hk = A.dot(Hk).dot(B) + rho * np.outer(s_k, s_k)

        # Update references
        x = x_new
        f = f_new
        g = g_new
        gnorm = np.linalg.norm(g)
        trajectory.append(x.reshape(n_beads, 3).copy())

        # --- Possibly do a "shake" every `shake_interval` iterations ---
        if iteration % shake_interval == 0:
            # Shake each coordinate by ~ N(0, shake_scale)
            noise = shake_scale * np.random.randn(dim)
            x += noise
            # Recompute energy and gradient after shake
            f, g = compute_energy_and_gradient(
                x.reshape(n_beads,3), n_beads, epsilon, sigma, b, k_b
            )
            gnorm = np.linalg.norm(g)
            if reset_hessian_after_shake:
                Hk = np.eye(dim)  # reset Hessian approximation
            trajectory.append(x.reshape(n_beads, 3).copy())

        if callback:
            callback(x)

        # (Optional) print debug info
        if iteration % 50 == 0:
            print(f"Iteration {iteration}, f={f:.6f}, gnorm={gnorm:.3e}")

    # Check convergence
    success = (gnorm <= tol)
    message = "Converged" if success else "Max iterations exceeded"

    result = CustomOptimizeResult(
        x=x,
        success=success,
        message=message,
        niter=iteration,
        fun=f
    )
    return result, trajectory

############### THE CRITICAL FUNCTION FOR AUTOGRADER ###############
"""
def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    
    Optimize the positions of the protein to minimize total energy.

    Parameters:
    ----------
    positions : np.ndarray
        A 2D NumPy array of shape (n_beads, d) representing the initial
        positions of the protein's beads in d-dimensional space.

    n_beads : int
        The number of beads (or units) in the protein model.

    write_csv : bool, optional (default=False)
        If True, the final optimized positions are saved to a CSV file.

    maxiter : int, optional (default=1000)
        The maximum number of iterations for the BFGS optimization algorithm.

    tol : float, optional (default=1e-6)
        The tolerance level for convergence in the optimization.

    Returns:
    -------
    result : CustomOptimizeResult
        The result of the optimization process, containing .x with final positions, etc.

    trajectory : list of np.ndarray
        A list of intermediate configurations (n_beads x 3).
        (Autograder currently ignores, but we return it anyway.)
    

    trajectory = []
    def callback(x):
        # For debugging or storing frames
        # x shape is (3*n_beads,)
        trajectory.append(x.reshape((n_beads, 3)).copy())

    # Flatten the initial guess
    x0 = positions.flatten()

    # Call our custom BFGS
    result, traj = bfgs_minimize(
        x0,
        n_beads=n_beads,
        maxiter=maxiter,
        tol=tol,
        # You could param-tune these LJ parameters
        epsilon=1.0,
        sigma=1.0,
        b=1.0,
        k_b=100.0,
        callback=callback
    )

    # If we want to combine callback frames with final frames:
    # but note that 'traj' from bfgs_minimize also has everything.
    # We'll just return `traj`.
    # (Make them consistent—some duplications are possible.)
    trajectory = traj

    if write_csv:
        csv_filepath = f"protein{n_beads}.csv"
        print(f"Writing data to file {csv_filepath}")
        np.savetxt(csv_filepath, result.x.reshape(n_beads, 3), delimiter=",")

    return result, trajectory
"""

def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    """
    Similar signature as original, but uses the 'shaken' BFGS approach.
    """
    trajectory = []
    def callback(x):
        trajectory.append(x.reshape((n_beads,3)).copy())

    x0 = positions.flatten()
    result, traj = bfgs_minimize_shaken(
        x0,
        n_beads=n_beads,
        maxiter=maxiter,
        tol=tol,
        epsilon=1.0,
        sigma=1.0,
        b=1.0,
        k_b=100.0,
        callback=callback,
        shake_interval=100,     # try a shake every 50 iterations
        shake_scale=0.01,      # magnitude of the shake
        reset_hessian_after_shake=False
    )

    if write_csv:
        # Write final positions to CSV
        csv_filepath = f"protein{n_beads}.csv"
        print(f"Writing data to file {csv_filepath}")
        np.savetxt(csv_filepath, result.x.reshape(n_beads, 3), delimiter=",")

    # If you want to unify 'trajectory' with 'traj', you can do so:
    # For now, let's just return 'traj' from bfgs_minimize_shaken
    return result, traj

############### OPTIONAL: UTILS FOR PLOTTING (OFF BY DEFAULT) ###############
# If you want, you can keep or remove, so long as it doesn't break headless runs.
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def plot_protein_3d(positions, title="Protein Conformation", ax=None):
        positions = positions.reshape((-1, 3))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', markersize=4)
        ax.set_title(title)
        plt.show()

    # Example usage
    n_beads = 500
    initial_positions = initialize_protein(n_beads)
    E_initial = total_energy(initial_positions, n_beads)
    print(f"Initial energy: {E_initial:.6f}")

    res, traj = optimize_protein(initial_positions, n_beads, write_csv=True, maxiter=10000, tol=1e-6)
    print(f"Optimization done. #iterations={res.nit}, final E={res.fun:.6f}")

    # Plot final result
    plot_protein_3d(res.x, title="Optimized Conformation")
