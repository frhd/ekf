import numpy as np
import matplotlib.pyplot as plt

def f(state, dt):
    x, y, theta, v, omega = state
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + omega * dt
    return np.array([x_new, y_new, theta_new, v, omega])

def F_jacobian(state, dt):
    x, y, theta, v, omega = state
    F = np.eye(5)
    F[0, 2] = -v * np.sin(theta) * dt
    F[0, 3] = np.cos(theta) * dt
    F[1, 2] = v * np.cos(theta) * dt
    F[1, 3] = np.sin(theta) * dt
    F[2, 4] = dt
    return F

def h(state):
    x, y, theta, v, omega = state
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([r, phi])

def H_jacobian(state):
    x, y, theta, v, omega = state
    r = np.sqrt(x**2 + y**2)
    if r < 1e-4:
        r = 1e-4
    H = np.zeros((2, 5))
    H[0, 0] = x / r
    H[0, 1] = y / r
    H[1, 0] = -y / (r**2)
    H[1, 1] = x / (r**2)
    return H

def ekf_predict(x, P, Q, dt):
    x_pred = f(x, dt)
    F = F_jacobian(x, dt)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z, R):
    z_pred = h(x_pred)
    y_res = z - z_pred
    H = H_jacobian(x_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y_res
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_upd, P_upd

def pf_initialize(mean, cov, N):
    return np.random.multivariate_normal(mean, cov, N)

def pf_predict(particles, dt, Q):
    N = particles.shape[0]
    new_particles = np.zeros_like(particles)
    for i in range(N):
        noise = np.random.multivariate_normal(np.zeros(5), Q)
        new_particles[i] = f(particles[i], dt) + noise
    return new_particles

def pf_update(particles, z, R):
    N = particles.shape[0]
    weights = np.zeros(N)
    for i in range(N):
        z_pred = h(particles[i])
        diff = z - z_pred
        exponent = -0.5 * diff.T @ np.linalg.inv(R) @ diff
        weights[i] = np.exp(exponent)
    weights += 1e-300
    weights /= np.sum(weights)
    return weights

def pf_resample(particles, weights):
    N = len(weights)
    indices = np.random.choice(np.arange(N), size=N, p=weights)
    return particles[indices]

dt = 0.1
T = 20
steps = int(T / dt)
true_state = np.array([0, 0, np.pi / 4, 1, 0.1])
Q = np.diag([0.01, 0.01, 0.001, 0.1, 0.01])
R = np.diag([0.5, 0.1])
ekf_state = np.array([0, 0, np.pi / 4, 1, 0.1])
ekf_P = np.eye(5)
pf_N = 500
pf_particles = pf_initialize(ekf_state, ekf_P, pf_N)
ekf_estimates = []
pf_estimates = []
true_states = []

for step in range(steps):
    true_state = f(true_state, dt) + np.random.multivariate_normal(np.zeros(5), Q)
    true_states.append(true_state.copy())
    if step % 10 == 5:
        z = None
    else:
        z = h(true_state) + np.random.multivariate_normal(np.zeros(2), R)
    if z is not None:
        ekf_state, ekf_P = ekf_predict(ekf_state, ekf_P, Q, dt)
        ekf_state, ekf_P = ekf_update(ekf_state, ekf_P, z, R)
        pf_particles = pf_predict(pf_particles, dt, Q)
        weights = pf_update(pf_particles, z, R)
        pf_particles = pf_resample(pf_particles, weights)
    else:
        ekf_state, ekf_P = ekf_predict(ekf_state, ekf_P, Q, dt)
        pf_particles = pf_predict(pf_particles, dt, Q)
    ekf_estimates.append(ekf_state.copy())
    pf_estimates.append(np.mean(pf_particles, axis=0))

ekf_estimates = np.array(ekf_estimates)
pf_estimates = np.array(pf_estimates)
true_states = np.array(true_states)

plt.figure(figsize=(10, 5))
plt.plot(true_states[:, 0], true_states[:, 1], 'k-', label='True')
plt.plot(ekf_estimates[:, 0], ekf_estimates[:, 1], 'b--', label='EKF')
plt.plot(pf_estimates[:, 0], pf_estimates[:, 1], 'r:', label='PF')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Tracking with EKF and PF')
plt.show()
