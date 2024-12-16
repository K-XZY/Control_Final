import numpy as np
import matplotlib.pyplot as plt
from observer import Observer
from dc_model import SysDyn
from regulator_model import RegulatorModel
from scipy.linalg import solve_discrete_are, inv
from numpy.linalg import matrix_rank

# Motor Parameters
J = 0.01      # Inertia (kg*m^2)
b = 0.1       # Friction coefficient (N*m*s)
K_t = 1       # Motor torque constant (N*m/A)
K_e = 0.01    # Back EMF constant (V*s/rad)
R_a = 1.0     # Armature resistance (Ohm)
L_a = 0.001   # Armature inductance (H)

# Desired Eigenvalues for Observer
lambda_1 = -0.5  # Observer eigenvalue 1
lambda_2 = -0.8  # Observer eigenvalue 2

# Simulation Parameters
t_start = 0.0
t_end = 0.04
dt = 0.00001  # Smaller time step for Euler integration
time = np.arange(t_start, t_end, dt)  # Time vector
num_steps = len(time)  # Number of simulation steps

# Initial Conditions for the System [omega, I_a]
x_init = np.array([0.0, 0.0])  # True system state [omega, I_a]
motor_model = SysDyn(J, b, K_t, K_e, R_a, L_a, dt, x_init)  # Initialize the motor model
motor_model.checkControlabilityContinuos()  # Check controllability of the continuous system

# Initial Conditions for the Observer [omega_hat, I_a_hat]
x_hat_init = np.array([0.0, 0.0])  # Initial guess for the observer state [omega_hat, I_a_hat]
observer = Observer(motor_model.A, motor_model.B, motor_model.C, dt, x_hat_init)

# Compute the observer gain L
# Place the eigenvalues of (A - L C) at desired locations
observer.ComputeObserverGains(lambda_1, lambda_2)

# Initialize MPC
# Define the matrices
num_states = 2  # Number of states in the system
num_controls = 1  # Number of control inputs
constraints_flag = False  # Constraint flag (disable by default)

# ATTENTION! Here we do not use the MPC but only use its functions to compute A, B, Q, and R
# Horizon length
N_mpc = 10  # Prediction horizon
regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states, constr_flag=constraints_flag)
regulator.setSystemMatrices(dt, motor_model.getA(), motor_model.getB())  # Define system matrices
regulator.checkStability()  # Check the stability of the discretized system
regulator.checkControllabilityDiscrete()  # Check controllability of the discretized system

# Define the cost matrices
Qcoeff = [100, 0.0]  # State cost coefficients
Rcoeff = [0.01] * num_controls  # Control effort cost coefficients
regulator.setCostMatrices(Qcoeff, Rcoeff)

Q, R = regulator.getCostMatrices()  # Retrieve cost matrices
A = regulator.getDiscreteA()  # Discretized A matrix
B = regulator.getDiscreteB()  # Discretized B matrix

# Desired state x_d
x_ref = np.array([10, 0])  # Desired reference for LQR

# Solve the discrete-time algebraic Riccati equation
P = solve_discrete_are(A, B, Q, R)

# Calculate the optimal control law K
K = inv(R + B.T @ P @ B) @ B.T @ P @ A

# Calculate the feedforward term
# Compute pseudoinverse of B
B_pinv = np.linalg.pinv(B)  # Result is a (1x2) matrix
# Compute Delta x (for the discrete system)
delta_x = A @ x_ref  # Result is a (2x1) vector
# Compute u_ff
u_ff = - B_pinv @ delta_x  # Feedforward control input

# Display the results
print("P matrix:")
print(P)
print("K matrix (control gains):")
print(K)
print("Feedforward control (u_ff):")
print(u_ff)

# Preallocate arrays for storing results
omega = np.zeros(num_steps)  # Angular velocity (rad/s)
I_a = np.zeros(num_steps)  # Armature current (A)
hat_omega = np.zeros(num_steps)  # Estimated angular velocity (rad/s)
hat_I_a = np.zeros(num_steps)  # Estimated armature current (A)
T_m_true = np.zeros(num_steps)  # True motor torque (N*m)
T_m_estimated = np.zeros(num_steps)  # Estimated motor torque (N*m)
V_terminal = np.zeros(num_steps)  # Measured terminal voltage (V)
V_terminal_hat = np.zeros(num_steps)  # Estimated terminal voltage (V)

x_cur = x_init  # Initialize current state
x_hat_cur = x_hat_init  # Initialize estimated state
x_i_k = np.zeros(num_states)  # Integral of output error
x_i_all = np.zeros((num_steps, num_states))  # Store integral of output error

# Simulation loop using Euler integration
for k in range(num_steps):
    t = time[k]  # Current timestamp
    # LQR control input
    V_a = - K @ (x_cur - x_ref) + u_ff  # LQR with feedforward
    cur_y = motor_model.step(V_a)  # Update the system model
    x_cur = motor_model.getCurrentState()  # True state of the system

    V_terminal[k] = cur_y  # Output measurement (Terminal Voltage)
    x_hat_cur, y_hat_cur = observer.update(V_a, cur_y)  # Update the observer

    # Store results
    omega[k] = x_cur[0]
    I_a[k] = x_cur[1]
    hat_omega[k] = x_hat_cur[0]
    hat_I_a[k] = x_hat_cur[1]
    T_m_true[k] = K_t * I_a[k]
    T_m_estimated[k] = K_t * hat_I_a[k]
    V_terminal_hat[k] = y_hat_cur

# Plotting the results
plt.figure(figsize=(12, 10))


# Add overarching title
plt.suptitle('Task 1.3: LQR Control for DC Motor with Observer Design', fontsize=16, fontweight='bold')


# Angular velocity
plt.subplot(5, 1, 1)
plt.plot(time, omega, label='True $\omega$ (rad/s)')
plt.title('Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

# Armature current
plt.subplot(5, 1, 2)
plt.plot(time, I_a, label='True $I_a$ (A)')
plt.title('Armature Current')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend()
plt.grid(True)

# Torque
plt.subplot(5, 1, 3)
plt.plot(time, T_m_true, label='True $T_m$ (N*m)')
plt.title('Motor Torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (N*m)')
plt.legend()
plt.grid(True)

# Terminal Voltage
plt.subplot(5, 1, 4)
plt.plot(time, V_terminal, label='Measured $V_{terminal}$ (V)')
plt.title('Terminal Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)

# Integral of output error
plt.subplot(5, 1, 5)
plt.plot(time, x_i_all[:, 0], label='Integral of output error $\int e_1$')
plt.title('Integral of output error')
plt.xlabel('Time (s)')
plt.ylabel('Integral of output error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#------------- Performance Metrics for LQR (Task 1.3) --------------------
ref = 10.0  # Reference angular velocity
final_value = omega[-1]  # Final angular velocity value
max_value = np.max(omega)  # Maximum angular velocity
overshoot = ((max_value - ref) / abs(ref)) * 100 if ref != 0 else 0  # Overshoot percentage

# Settling time: within ±5% of reference
tolerance = 0.05 * abs(ref)  # 5% tolerance band
settling_time = None
for i in range(num_steps):
    if np.all(np.abs(omega[i:] - ref) < tolerance):
        settling_time = time[i]  # Time at which settling occurs
        break

# Steady-state error
sse = final_value - ref  # Difference between final value and reference

# Print performance metrics
print("\n--- Performance Metrics (LQR) ---")
print(f"Reference: {ref} rad/s")
print(f"Final value: {final_value:.4f} rad/s")
print(f"Overshoot: {overshoot:.2f}%")
if settling_time is not None:
    print(f"Settling time (5% band): {settling_time:.6f} s")
else:
    print("Settling time not found within given tolerance.")
print(f"Steady-state error: {sse:.4f} rad/s")
