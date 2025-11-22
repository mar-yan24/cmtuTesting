import numpy as np
from stable_baselines3 import PPO
import gym

def collect_poincare_samples(env, model, marker_idx, state_indices,
                             n_crossings=200, skip_initial=20, render=False):
    """
    Run the trained policy in closed loop and collect states at Poincaré crossings.

    marker_idx: index in obs used as h(x) for crossing detection.
    state_indices: list/array of obs indices defining the reduced Poincaré state y.
    n_crossings: total number of crossings to keep (after skipping).
    skip_initial: number of initial crossings to ignore (transient).
    """
    obs = env.reset()
    # Handle Gym vs Gymnasium reset API:
    if isinstance(obs, tuple):
        obs, _info = obs

    h_prev = obs[marker_idx]
    samples = []
    crossings = 0

    # If your env has dt, use it; otherwise leave as None or constant
    dt = getattr(env, "dt", None)

    while crossings < n_crossings + skip_initial:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        # Gym: obs, reward, done, info
        # Gymnasium: obs, reward, terminated, truncated, info
        if len(step_out) == 4:
            obs_next, reward, done, info = step_out
            terminated, truncated = done, False
        else:
            obs_next, reward, terminated, truncated, info = step_out

        if render:
            env.render()

        h_curr = obs_next[marker_idx]

        # Crossing from negative to >= 0 (choose sign/direction you want)
        if h_prev < 0.0 and h_curr >= 0.0:
            crossings += 1
            if crossings > skip_initial:
                samples.append(obs_next[state_indices])

        h_prev = h_curr
        obs = obs_next

        if terminated or truncated:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs, _info = obs
            h_prev = obs[marker_idx]

    return np.array(samples)


def estimate_poincare_linearization(samples):
    """
    samples: array of shape (N, d) of y_k (Poincaré states).
    Returns: y_star, J, eigenvalues of J.
    """
    # Pair y_k with y_{k+1}
    X = samples[:-1, :]   # y_k
    Y = samples[1:, :]    # y_{k+1}

    # Estimate fixed point as mean (you can also do a more careful fit if needed)
    y_star = X.mean(axis=0)

    dX = X - y_star
    dY = Y - y_star

    d, = y_star.shape
    J = np.zeros((d, d))

    # For each output dimension, regress dY[:, j] on dX
    # dY[:, j] ≈ J[j, :] @ dX.T
    for j in range(d):
        beta, *_ = np.linalg.lstsq(dX, dY[:, j], rcond=None)
        J[j, :] = beta

    eigvals, eigvecs = np.linalg.eig(J)
    return y_star, J, eigvals


def estimate_1d_poincare(samples):
    # y_k, y_{k+1}
    x = samples[:-1]
    y = samples[1:]
    # Fit y ≈ a x + b
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    x_star = b / (1.0 - a)
    slope_at_fixed = a  # eigenvalue of 1D map
    return x_star, slope_at_fixed

# ==== 1. Load env and model ====
env_id = "myoAssistLegImitation-v0"  # or your exo-free env
env = gym.make(env_id)

checkpoint_dir = "path/to/checkpoint_dir"  # folder with policy.pth, data, etc.
model = PPO.load(checkpoint_dir, env=env, device="cuda")  # or "cpu"

# ==== 2. Define which obs indices to use ====
# Example: suppose:
# - marker_idx = index of some scalar defining the phase (e.g., hip angle)
# - state_indices = indices of a reduced state vector (e.g., several joint angles/velocities)
marker_idx = 0                # <-- replace with your section function index
state_indices = [0, 1, 2, 3]  # <-- replace with the indices you care about

# ==== 3. Collect Poincaré samples ====
samples = collect_poincare_samples(
    env, model,
    marker_idx=marker_idx,
    state_indices=state_indices,
    n_crossings=300,
    skip_initial=50,
    render=False,
)

print("Collected samples shape:", samples.shape)

# ==== 4. Analyze stability ====
y_star, J, eigvals = estimate_poincare_linearization(samples)
print("Estimated fixed point y*:", y_star)
print("Jacobian J:\n", J)
print("Eigenvalues of J:", eigvals)
print("Spectral radius:", np.max(np.abs(eigvals)))