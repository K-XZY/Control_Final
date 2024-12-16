import numpy as np
import matplotlib.pyplot as plt
from observer import Observer
from dc_model import SysDyn
from regulator_model import RegulatorModel 
from scipy.linalg import solve_discrete_are, inv

def ParameterSimulation(K_e, R_a, L_a, J=0.01, b=0.1, K_t=1, LQR=False):
    # Nominal parameters
    K_e_true = 0.01   
    R_a_true = 1.0   
    L_a_true = 0.001  

    if LQR:
        lambda_1 = -0.5
        lambda_2 = -0.8

        t_start = 0.0
        t_end = 0.04
        dt = 0.00001
        time = np.arange(t_start, t_end, dt)
        num_steps = len(time)

        x_init = np.array([0.0, 0.0]) 
        motor_model = SysDyn(J, b, K_t, K_e_true, R_a_true, L_a_true, dt, x_init)

        # Perturbed model for regulator design
        motor_model_perturbed = SysDyn(J, b, K_t, K_e, R_a, L_a, dt, x_init)
        motor_model.checkControlabilityContinuos()

        x_hat_init = np.array([0.0, 0.0])
        observer = Observer(motor_model.A, motor_model.B, motor_model.C, dt, x_hat_init)
        observer.ComputeObserverGains(lambda_1, lambda_2)

        num_states = 2
        num_controls = 1
        constraints_flag = False
        N_mpc = 10
        regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states, constr_flag=constraints_flag)
        regulator.setSystemMatrices(dt, motor_model_perturbed.getA(), motor_model_perturbed.getB())
        regulator.checkStability()
        regulator.checkControllabilityDiscrete()

        Qcoeff = [100, 0.0]
        Rcoeff = [0.01]*num_controls
        regulator.setCostMatrices(Qcoeff, Rcoeff)
        Q, R = regulator.getCostMatrices()
        A = regulator.getDiscreteA()
        B = regulator.getDiscreteB()

        x_ref = np.array([10, 0])
        P = solve_discrete_are(A, B, Q, R)
        K = inv(R + B.T @ P @ B) @ B.T @ P @ A

        B_pinv = np.linalg.pinv(B)
        delta_x = A @ x_ref
        u_ff = - B_pinv @ delta_x

        print("P matrix:")
        print(P)
        print("K matrix (control gains):")
        print(K)
        print("Feedforward control (u_ff):")
        print(u_ff)

        omega = np.zeros(num_steps)
        I_a = np.zeros(num_steps)
        hat_omega = np.zeros(num_steps)
        hat_I_a = np.zeros(num_steps)
        T_m_true = np.zeros(num_steps)
        T_m_estimated = np.zeros(num_steps)
        V_terminal = np.zeros(num_steps)
        V_terminal_hat = np.zeros(num_steps)

        x_cur = x_init
        x_hat_cur = x_hat_init
        x_i_all = np.zeros((num_steps, 2))

        for k in range(num_steps):
            V_a = - K @ (x_cur - x_ref) + u_ff
            cur_y = motor_model.step(V_a)
            x_cur = motor_model.getCurrentState()
            V_terminal[k] = cur_y
            x_hat_cur, y_hat_cur = observer.update(V_a, cur_y)

            omega[k] = x_cur[0]
            I_a[k] = x_cur[1]
            hat_omega[k] = x_hat_cur[0]
            hat_I_a[k] = x_hat_cur[1]
            T_m_true[k] = K_t * I_a[k]
            T_m_estimated[k] = K_t * hat_I_a[k]
            V_terminal_hat[k] = y_hat_cur

        return time, omega, I_a, T_m_true, V_terminal, x_i_all[:, 0]

    else:
        # MPC case
        lambda_1 = -10
        lambda_2 = -15

        t_start = 0.0
        t_end = 0.05
        dt = 0.00001
        time = np.arange(t_start, t_end, dt)
        num_steps = len(time)

        x_init = np.array([0.0, 0.0]) 
        motor_model = SysDyn(J, b, K_t, K_e_true, R_a_true, L_a_true, dt, x_init)
        motor_model_perturbed = SysDyn(J, b, K_t, K_e, R_a, L_a, dt, x_init)

        x_hat_init = np.array([0.0, 0.0])
        observer = Observer(motor_model.A, motor_model.B, motor_model.C, dt, x_hat_init)
        observer.ComputeObserverGains(lambda_1, lambda_2)

        num_states = 2
        num_controls = 1
        constraints_flag = False
        N_mpc = 10
        regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states, constr_flag=constraints_flag)
        regulator.setSystemMatrices(dt, motor_model_perturbed.getA(), motor_model_perturbed.getB())

        Qcoeff = [1000.0, 0.0]
        Rcoeff = [0.01]*num_controls
        regulator.setCostMatrices(Qcoeff, Rcoeff)
        x_ref = np.array([-10,0])
        regulator.propagation_model_regulator_fixed_std(x_ref)

        B_in = {'max': np.array([1e14]*num_controls), 'min': np.array([-1e12]*num_controls)}
        B_out = {'max': np.array([1e8,1e9]), 'min': np.array([-1e8,-1e9])}
        regulator.setConstraintsMatrices(B_in,B_out)
        regulator.compute_H_and_F()

        omega = np.zeros(num_steps)
        I_a = np.zeros(num_steps)
        hat_omega = np.zeros(num_steps)
        hat_I_a = np.zeros(num_steps)
        T_m_true = np.zeros(num_steps)
        T_m_estimated = np.zeros(num_steps)
        V_terminal = np.zeros(num_steps)
        V_terminal_hat = np.zeros(num_steps)

        x_cur = x_init
        x_hat_cur = x_hat_init

        for k in range(num_steps):
            u_mpc = regulator.compute_solution(x_hat_cur)
            V_a = u_mpc[0]

            cur_y = motor_model.step(V_a)
            x_cur = motor_model.getCurrentState()
            V_terminal[k] = cur_y
            x_hat_cur, y_hat_cur = observer.update(V_a, cur_y)

            omega[k] = x_cur[0]
            I_a[k] = x_cur[1]
            hat_omega[k] = x_hat_cur[0]
            hat_I_a[k] = x_hat_cur[1]
            T_m_true[k] = K_t * I_a[k]
            T_m_estimated[k] = K_t * hat_I_a[k]
            V_terminal_hat[k] = y_hat_cur

        return time, omega, I_a, T_m_true, V_terminal

J = 0.01
b = 0.1
K_t = 1
K_e = 0.01
R_a = 1.0
L_a = 0.001

num_sim = 3
LQR=True   # change it to False or True

# Adjusted initialization based on the maximum expected number of steps
max_num_steps = 5000
omega_list = np.zeros((num_sim, max_num_steps))
I_a_list = np.zeros((num_sim, max_num_steps))
T_m_true_list = np.zeros((num_sim, max_num_steps))
V_terminal_list = np.zeros((num_sim, max_num_steps))

for i in range(num_sim):
    K_e = 0.08 + i * 0.002
    R_a = 0.8 + i * 0.2
    L_a = 0.0008 + i * 0.0002

    if LQR:
        time, omega, I_a, T_m_true, V_terminal, x_i_all = ParameterSimulation(K_e, R_a, L_a, LQR=LQR)
    else:
        time, omega, I_a, T_m_true, V_terminal = ParameterSimulation(K_e, R_a, L_a, LQR=LQR)

    # Dynamically determine the number of steps
    num_steps = len(omega)

    # Store results in the respective arrays
    omega_list[i, :num_steps] = omega
    I_a_list[i, :num_steps] = I_a
    T_m_true_list[i, :num_steps] = T_m_true
    V_terminal_list[i, :num_steps] = V_terminal

# Determine the title based on the controller type
if LQR:
    main_title = (
        "Task 1.4: Robustness Analysis of LQR Control with Parameter Perturbations\n"
        "Altered Parameters: $K_e$, $R_a$, $L_a$"
    )
else:
    main_title = (
        "Task 1.4: Robustness Analysis of MPC Control with Parameter Perturbations\n"
        "Altered Parameters: $K_e$, $R_a$, $L_a$"
    )


# Plot results
legend = ['-20%', 'Nominal', '+20%']

plt.figure(figsize=(12, 10))
# Add overarching title
plt.suptitle(main_title, fontsize=14, fontweight='bold')


plt.subplot(4, 1, 1)
for i in range(num_sim):
    plt.plot(time, omega_list[i, :len(time)])
plt.title('Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend(legend)
plt.grid(True)

plt.subplot(4, 1, 2)
for i in range(num_sim):
    plt.plot(time, I_a_list[i, :len(time)])
plt.title('Armature Current')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend(legend)
plt.grid(True)

plt.subplot(4, 1, 3)
for i in range(num_sim):
    plt.plot(time, T_m_true_list[i, :len(time)])
plt.title('Motor Torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (N*m)')
plt.legend(legend)
plt.grid(True)

plt.subplot(4, 1, 4)
for i in range(num_sim):
    plt.plot(time, V_terminal_list[i, :len(time)])
plt.title('Terminal Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend(legend)
plt.grid(True)

plt.tight_layout()
plt.show()
