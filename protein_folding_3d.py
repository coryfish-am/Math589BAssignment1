import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Initialize protein positions
def initialize_protein(n_beads, dimension=3, fudge = 1e-5):
    """
    Initialize a protein with `n_beads` arranged almost linearly in `dimension`-dimensional space.
    The `fudge` is a factor that, if non-zero, adds a spiral structure to the configuration.
    """
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1  # Fixed bond length of 1 unit
        positions[i, 1] = fudge * np.sin(i)  # Fixed bond length of 1 unit
        positions[i, 2] = fudge * np.sin(i*i)  # Fixed bond length of 1 unit                
    return positions

# Lennard-Jones potential function
def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """
    Compute Lennard-Jones potential between two beads.
    """
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# Bond potential function
def bond_potential(r, b=1.0, k_b=100.0):
    """
    Compute harmonic bond potential between two bonded beads.
    """
    return k_b * (r - b)**2

# Total energy function
def total_energy(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Compute the total energy of the protein conformation.
    """
    positions = positions.reshape((n_beads, -1))
    energy = 0.0

    # Bond energy
    for i in range(n_beads - 1):
        r = np.linalg.norm(positions[i+1] - positions[i])
        energy += bond_potential(r, b, k_b)

    # Lennard-Jones potential for non-bonded interactions
    for i in range(n_beads):
        for j in range(i+1, n_beads):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 1e-2:  # Avoid division by zero
                energy += lennard_jones_potential(r, epsilon, sigma)

    return energy

# Optimization function
def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    """
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
        The result of the optimization process, with the optimized positions
        in result.x, plus iteration info, and a success flag.

    trajectory : list of np.ndarray
        A list of intermediate configurations during the optimization,
        where each element is (n_beads, d) representing the beads at that step.
    """
    import numpy as np

    # We'll store the optimization trajectory for plotting/animation
    trajectory = []

    # A local function to compute both the energy and the gradient
    def energy_grad_func(x_flat):
        """
        Returns (energy, gradient) for the potential.
        We'll do a simple finite-difference gradient for demonstration.
        For better performance, implement analytical gradients.
        """
        # Reshape for total_energy
        x_reshaped = x_flat.reshape(n_beads, -1)

        # 1) Evaluate the scalar function
        f = total_energy(x_reshaped.flatten(), n_beads)

        # 2) Compute gradient by finite differences
        grad = np.zeros_like(x_flat)
        epsilon = 1e-6
        for i in range(len(x_flat)):
            old_val = x_flat[i]

            # x_plus
            x_flat[i] = old_val + epsilon
            f_plus = total_energy(x_flat.reshape(n_beads, -1), n_beads)

            # x_minus
            x_flat[i] = old_val - epsilon
            f_minus = total_energy(x_flat.reshape(n_beads, -1), n_beads)

            # restore
            x_flat[i] = old_val

            grad[i] = (f_plus - f_minus) / (2.0 * epsilon)

        return f, grad

    # Our custom BFGS
    def bfgs(positions_flat, maxiter, tol):
        x = positions_flat.copy()
        n = len(x)
        B = np.eye(n)  # Inverse Hessian approx
        f, g = energy_grad_func(x)
        gnorm = np.linalg.norm(g)
        alpha_init = 1.0
        c1 = 1e-4
        rho = 0.9

        for it in range(maxiter):
            # Save trajectory for plotting
            trajectory.append(x.reshape(n_beads, -1))

            # Check convergence
            if gnorm < tol:
                return x, f, True, it

            # 1) Search direction
            p = -B.dot(g)

            # 2) Armijo line search
            alpha = alpha_init
            f_old = f
            gTp = np.dot(g, p)

            while True:
                x_new = x + alpha * p
                f_new, g_new = energy_grad_func(x_new)
                if f_new <= f_old + c1 * alpha * gTp:
                    break
                alpha *= rho
                if alpha < 1e-16:
                    # Step size too small => fail
                    return x, f, False, it

            # BFGS update
            s = alpha * p
            y = g_new - g
            sy = np.dot(s, y)

            if abs(sy) > 1e-14:
                By = B.dot(y)
                yTBy = np.dot(y, By)
                B += np.outer(s, s) * (1.0 + (yTBy / sy)) / sy
                B -= (np.outer(By, s) + np.outer(s, By)) / sy

            # Accept step
            x = x_new
            f = f_new
            g = g_new
            gnorm = np.linalg.norm(g)

        # If we exceed maxiter
        return x, f, False, maxiter

    # Run our custom BFGS
    x0_flat = positions.flatten()
    x_opt, f_opt, success, n_iter = bfgs(x0_flat, maxiter, tol)

    # Optionally save final positions
    if write_csv:
        print(f"Writing data to file protein{n_beads}.csv")
        np.savetxt(f"protein{n_beads}.csv", trajectory[-1], delimiter=",")

    # We define a small custom class for the result
    class CustomOptimizeResult:
        def __init__(self, x, success, nit):
            self.x = x  # This is what the autograder will look for
            self.success = success
            self.nit = nit

    result = CustomOptimizeResult(x_opt, success, n_iter)
    return result, trajectory


# 3D visualization function
def plot_protein_3d(positions, title="Protein Conformation", ax=None):
    """
    Plot the 3D positions of the protein.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    positions = positions.reshape((-1, 3))
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', markersize=6)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# Animation function
# Animation function with autoscaling
def animate_optimization(trajectory, interval=100):
    """
    Animate the protein folding process in 3D with autoscaling.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], '-o', markersize=6)

    def update(frame):
        positions = trajectory[frame]
        line.set_data(positions[:, 0], positions[:, 1])
        line.set_3d_properties(positions[:, 2])

        # Autoscale the axes
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_zlim(z_min - 1, z_max + 1)

        ax.set_title(f"Step {frame + 1}/{len(trajectory)}")
        return line,

    ani = FuncAnimation(
        fig, update, frames=len(trajectory), interval=interval, blit=False
    )
    plt.show()

# Main function
if __name__ == "__main__":
    n_beads = 10
    dimension = 3
    initial_positions = initialize_protein(n_beads, dimension)

    print("Initial Energy:", total_energy(initial_positions.flatten(), n_beads))
    plot_protein_3d(initial_positions, title="Initial Configuration")

    result, trajectory = optimize_protein(initial_positions, n_beads, write_csv = True)

    optimized_positions = result.x.reshape((n_beads, dimension))
    print("Optimized Energy:", total_energy(optimized_positions.flatten(), n_beads))
    plot_protein_3d(optimized_positions, title="Optimized Configuration")

    # Animate the optimization process
    animate_optimization(trajectory)