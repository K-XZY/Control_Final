import numpy as np
import matplotlib.pyplot as plt
import os

# Load data1 and data2
print("Loading data1...")
data1 = np.load("data1.npy", allow_pickle=True).item()
print("Loading data2...")
data2 = np.load("data2.npy", allow_pickle=True).item()

# Extract arrays from data1
q_mes_all1 = np.array(data1['q_mes_all'])   # shape: T x 7
qd_mes_all1 = np.array(data1['qd_mes_all']) # shape: T x 7
u_mpc_all1 = np.array(data1['u_mpc_all'])   # not used for plotting error, but we have it
q_d_all1 = np.array(data1['q_d_all'])       # shape: T x 7
qd_d_all1 = np.array(data1['qd_d_all'])     # shape: T x 7
time_all1 = np.array(data1['time_all'])     # shape: T

# Extract arrays from data2
q_mes_all2 = np.array(data2['q_mes_all'])
qd_mes_all2 = np.array(data2['qd_mes_all'])
u_mpc_all2 = np.array(data2['u_mpc_all'])   # not used for error plotting
q_d_all2 = np.array(data2['q_d_all'])
qd_d_all2 = np.array(data2['qd_d_all'])
time_all2 = np.array(data2['time_all'])

time_all = time_all1  # Using data1 time as reference
if time_all2.shape != time_all1.shape:
    print("Warning: time arrays differ in shape, consider aligning them.")
    # For simplicity, this code assumes they match.

# Compute errors
# error_q for data1: error_q1 = q_d_all1 - q_mes_all1
error_q1 = q_d_all1 - q_mes_all1
# error_qd for data1: error_qd1 = qd_d_all1 - qd_mes_all1
error_qd1 = qd_d_all1 - qd_mes_all1

# error_q for data2: error_q2 = q_d_all2 - q_mes_all2
error_q2 = q_d_all2 - q_mes_all2
# error_qd for data2: error_qd2 = qd_d_all2 - qd_mes_all2
error_qd2 = qd_d_all2 - qd_mes_all2

def rolling_mean(data, N):
    """
    Compute the rolling mean with a window size of N.
    For the first few time steps where there are fewer than N values,
    the mean is computed using all available values.
    
    Parameters:
        data (ndarray): Input data array of shape (T, joints)
        N (int): Window size
    
    Returns:
        ndarray: Rolling mean of shape (T, joints)
    """
    T, joints = data.shape
    rolling_mean_array = np.zeros((T, joints))
    for t in range(T):
        start_idx = max(0, t - N + 1)
        rolling_mean_array[t] = np.mean(data[start_idx:t+1], axis=0)
    return rolling_mean_array

# Define the rolling window size
N = 1000

# Compute rolling means for each dataset
error_q_mean1 = rolling_mean(error_q1, N)   # shape: (T, 7)
error_qd_mean1 = rolling_mean(error_qd1, N) # shape: (T, 7)

error_q_mean2 = rolling_mean(error_q2, N)   # shape: (T, 7)
error_qd_mean2 = rolling_mean(error_qd2, N) # shape: (T, 7)
# Create output directory
output_dir = "Q3P3"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory {output_dir} for storing figures.")

# Plotting styles
plt.rcParams['figure.figsize'] = (16, 9)
colors = {
    'data1': 'blue',
    'data2': 'red',
    'data1_mean': 'cyan',
    'data2_mean': 'orange'
}

print("Generating plots...")

for i in range(7):
    joint_name = f"Joint {i+1}"
    
    # Plot for q
    fig_q = plt.figure()
    plt.plot(time_all, error_q1[:, i], color=colors['data1'], label=f"error_q1 (Gravity Cancelation)")
    plt.plot(time_all, error_q2[:, i], color=colors['data2'], label=f"error_q2 (LTV)")
    # Plot mean lines as horizontal lines
    # Plotting the cumulative mean error
    plt.plot(time_all, error_q_mean1[:, i], color=colors['data1_mean'], linestyle='--', label=f"mean error_q1 (Gravity Cancelation) of {N} steps")
    plt.plot(time_all, error_q_mean2[:, i], color=colors['data2_mean'], linestyle='--', label=f"mean error_q2 (LTV) of {N} steps")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.title(f"{joint_name} Position Tracking Error (rad)")
    plt.legend()
    filename_q = os.path.join(output_dir, f"error_q_joint_{i+1}.png")
    plt.savefig(filename_q)
    plt.close(fig_q)
    print(f"Saved {filename_q}")
    
    # plot for qd
    fig_qd = plt.figure()
    plt.plot(time_all, error_qd1[:, i], color=colors['data1'], label=f"error_qd1 (Gravity Cancelation)")
    plt.plot(time_all, error_qd2[:, i], color=colors['data2'], label=f"error_qd2 (LTV)")
    # Plot mean lines as horizontal lines
    plt.plot(time_all, error_qd_mean1[:, i], color=colors['data1_mean'], linestyle='--', label=f"mean error_q1 (Gravity Cancelation) of {N} steps")
    plt.plot(time_all, error_qd_mean2[:, i], color=colors['data2_mean'], linestyle='--', label=f"mean error_q2 (LTV) of {N} steps")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad/s)")
    plt.title(f"{joint_name} Velocity Tracking Error (rad/s)")
    plt.legend()
    filename_qd = os.path.join(output_dir, f"error_qd_joint_{i+1}.png")
    plt.savefig(filename_qd)
    plt.close(fig_qd)
    print(f"Saved {filename_qd}")

print(f"All plots have been generated and saved in {output_dir}")
