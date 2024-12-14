import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load data from the standard MPC run (Task 3.1)
    data_std = np.load("data1.npy", allow_pickle=True).item()
    q_mes_all_std = np.array(data_std['q_mes_all'])
    qd_mes_all_std = np.array(data_std['qd_mes_all'])
    q_d_all_std = np.array(data_std['q_d_all'])
    qd_d_all_std = np.array(data_std['qd_d_all'])
    time_all_std = np.array(data_std['time_all'])
    print(time_all_std[-1])

    # Load data from the LTV MPC run (Task 3.2)
    data_ltv = np.load("data2.npy", allow_pickle=True).item()
    q_mes_all_ltv = np.array(data_ltv['q_mes_all'])
    qd_mes_all_ltv = np.array(data_ltv['qd_mes_all'])
    q_d_all_ltv = np.array(data_ltv['q_d_all'])
    qd_d_all_ltv = np.array(data_ltv['qd_d_all'])
    time_all_ltv = np.array(data_ltv['time_all'])

    # Compute errors
    # error_p = q_mes - q_d
    error_p_std = q_mes_all_std - q_d_all_std
    error_p_ltv = q_mes_all_ltv - q_d_all_ltv

    # error_pd = qd_mes - qd_d
    error_pd_std = qd_mes_all_std - qd_d_all_std
    error_pd_ltv = qd_mes_all_ltv - qd_d_all_ltv

    # For plotting a single representative measure, we take the mean across all joints at each time step
    mean_error_p_std = np.mean(error_p_std, axis=1)   # shape: (time,)
    mean_error_p_ltv = np.mean(error_p_ltv, axis=1)
    mean_error_pd_std = np.mean(error_pd_std, axis=1)
    mean_error_pd_ltv = np.mean(error_pd_ltv, axis=1)

    # Compute cumulative mean (running average) over time
    # cumulative mean at time t is mean(error[0:t]) 
    # Note: We assume time_all_std and time_all_ltv have the same length and sampling.
    # If not, interpolation or time alignment is needed.
    def running_mean(data):
        return np.cumsum(data) / (np.arange(len(data)) + 1)

    cumm_mean_error_p_std = running_mean(mean_error_p_std)
    cumm_mean_error_p_ltv = running_mean(mean_error_p_ltv)
    cumm_mean_error_pd_std = running_mean(mean_error_pd_std)
    cumm_mean_error_pd_ltv = running_mean(mean_error_pd_ltv)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Top subplot: Position error
    ax1.plot(time_all_std, mean_error_p_std, label='Standard MPC Position Error', color='blue')
    ax1.plot(time_all_ltv, mean_error_p_ltv, label='LTV MPC Position Error', color='orange', linestyle='--')
    ax1.plot(time_all_std, cumm_mean_error_p_std, label='Standard MPC Cumulative Mean Pos. Error', color='blue', linewidth=2, alpha=0.7)
    ax1.plot(time_all_ltv, cumm_mean_error_p_ltv, label='LTV MPC Cumulative Mean Pos. Error', color='orange', linewidth=2, alpha=0.7, linestyle='--')

    ax1.set_title('Position Tracking Error Comparison', fontsize=14)
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Position Error [rad]', fontsize=12)
    ax1.grid(True)
    ax1.legend(fontsize=10)

    # Bottom subplot: Velocity (position derivative) error
    ax2.plot(time_all_std, mean_error_pd_std, label='Standard MPC Velocity Error', color='green')
    ax2.plot(time_all_ltv, mean_error_pd_ltv, label='LTV MPC Velocity Error', color='red', linestyle='--')
    ax2.plot(time_all_std, cumm_mean_error_pd_std, label='Standard MPC Cumulative Mean Vel. Error', color='green', linewidth=2, alpha=0.7)
    ax2.plot(time_all_ltv, cumm_mean_error_pd_ltv, label='LTV MPC Cumulative Mean Vel. Error', color='red', linewidth=2, alpha=0.7, linestyle='--')

    ax2.set_title('Velocity Tracking Error Comparison', fontsize=14)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Velocity Error [rad/s]', fontsize=12)
    ax2.grid(True)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('position_velocity_error_comparison.png')
    plt.show()

if __name__ == '__main__':
    main()
