import numpy as np
from scipy.optimize import minimize

class TrackerModel:
    def __init__(self, N, q, m, n, delta,constr_flag=True):
        #A_cont = A_continuos
        #B_cont = B_continuos
        #C_cont = C_continuos
        self.A = None # state transition matrix
        self.B = None # input matrix
        self.C = None # output matrix
        self.Q = None # cost matrix
        self.R = None # cost matrix
        self.W = None # constraints matrix
        self.G = None # constraints matrix
        self.S = None # constraints matrix  
        self.N = N # prediction horizon
        self.q = q # number of outputs  
        self.m = m # Number of control inputs
        self.orig_n = n # Number of states#
        self.n = n      # current state dimension
        self.delta = delta
        self.constr_flag = constr_flag

       

    def tracker_std(self, S_bar, T_bar, Q_hat, Q_bar, R_bar):
        """
        Compute the Hessian and linear terms for the MPC tracking problem
        """
        # Compute H
        H = R_bar + S_bar.T @ Q_bar @ S_bar
        
        # Compute F_tra
        # F_tra should match dimension of augmented state 
        # [x_ref; x_cur; u_cur]
        first = -Q_hat @ S_bar  # For reference trajectory
        second = T_bar.T @ Q_bar @ S_bar  # For current state
        third = np.zeros((self.m, S_bar.shape[1]))  # For current control
        
        F_tra = np.vstack([first, second, third])
        
        return H, F_tra
    
    def setSystemMatrices(self, q_d, qd_d, dyn_model=None):
        """
        Set system matrices based on desired trajectory point
        """
        if dyn_model is None:
            raise ValueError("Must provide dynamics model")
        
        # Get linearized continuous time matrices
        A_cont, B_cont = self.compute_linearized_matrices(q_d, qd_d, dyn_model)
        
        num_joints = self.m
        num_states = 2 * num_joints
        
        # Discrete time conversion
        self.A = np.eye(num_states) + self.delta * A_cont
        self.B = self.delta * B_cont
        
        # Output matrix C maps full state
        self.C = np.eye(num_states)

    # TODO you can change this function to allow for more passing a vector of gains
    def setCostMatrices(self, Qcoeff, Rcoeff):
        """
        Set the cost matrices Q and R for the MPC controller.

        Parameters:
        Qcoeff: float or array-like
            State cost coefficient(s). If scalar, the same weight is applied to all states.
            If array-like, should have a length equal to the number of states.

        Rcoeff: float or array-like
            Control input cost coefficient(s). If scalar, the same weight is applied to all control inputs.
            If array-like, should have a length equal to the number of control inputs.

        Sets:
        self.Q: ndarray
            State cost matrix.
        self.R: ndarray
            Control input cost matrix.
        """

        num_states = self.orig_n
        num_controls = self.m

        # Process Qcoeff
        if np.isscalar(Qcoeff):
            # If Qcoeff is a scalar, create an identity matrix scaled by Qcoeff
            Q = Qcoeff * np.eye(num_states)
        else:
            # Convert Qcoeff to a numpy array
            Qcoeff = np.array(Qcoeff)
            if Qcoeff.ndim != 1 or len(Qcoeff) != num_states:
                raise ValueError(f"Qcoeff must be a scalar or a 1D array of length {num_states}")
            # Create a diagonal matrix with Qcoeff as the diagonal elements
            Q = np.diag(Qcoeff)

        # Process Rcoeff
        if np.isscalar(Rcoeff):
            # If Rcoeff is a scalar, create an identity matrix scaled by Rcoeff
            R = Rcoeff * np.eye(num_controls)
        else:
            # Convert Rcoeff to a numpy array
            Rcoeff = np.array(Rcoeff)
            if Rcoeff.ndim != 1 or len(Rcoeff) != num_controls:
                raise ValueError(f"Rcoeff must be a scalar or a 1D array of length {num_controls}")
            # Create a diagonal matrix with Rcoeff as the diagonal elements
            R = np.diag(Rcoeff)

        # Assign the matrices to the object's attributes
        self.Q = Q
        self.R = R

    def propagation_model_tracker_fixed_std(self):
        # Determine sizes and initialize matrices
        S_bar = np.zeros((self.n * self.N, self.m * self.N))
        S_bar_C = np.zeros((self.q * self.N, self.m * self.N))
        T_bar = np.zeros((self.n * self.N, self.n))
        T_bar_C = np.zeros((self.q * self.N, self.n))
        Q_hat = np.zeros((self.q * self.N, self.n * self.N))
        Q_bar = np.zeros((self.n * self.N, self.n * self.N))
        R_bar = np.zeros((self.m * self.N, self.m * self.N))

        # Loop to calculate matrices
        for k in range(1, self.N + 1):
            for j in range(1, k + 1):
                idx_row_S = slice(self.n * (k-1), self.n * k)
                idx_col_S = slice(self.m * (k-j), self.m * (k-j+1))
                S_bar[idx_row_S, idx_col_S] = np.linalg.matrix_power(self.A, j-1) @ self.B

                idx_row_SC = slice(self.q * (k-1), self.q * k)
                S_bar_C[idx_row_SC, idx_col_S] = self.C @ np.linalg.matrix_power(self.A, j-1) @ self.B

            idx_row_T = slice(self.n * (k-1), self.n * k)
            T_bar[idx_row_T, :] = np.linalg.matrix_power(self.A, k)

            idx_row_TC = slice(self.q * (k-1), self.q * k)
            T_bar_C[idx_row_TC, :] = self.C @ np.linalg.matrix_power(self.A, k)

            idx_row_QH = slice(self.q * (k-1), self.q * k)
            idx_col_QH = slice(self.n * (k-1), self.n * k)
            Q_hat[idx_row_QH, idx_col_QH] = self.Q @ self.C

            idx_row_col_QB = slice(self.n * (k-1), self.n * k)
            Q_bar[idx_row_col_QB, idx_row_col_QB] = self.C.T @ self.Q @ self.C

            idx_row_col_R = slice(self.m * (k-1), self.m * k)
            R_bar[idx_row_col_R, idx_row_col_R] = self.R

        return S_bar, S_bar_C, T_bar, T_bar_C, Q_hat, Q_bar, R_bar
    
    def tracker_G_std(self, S_bar_C):
        #G = np.vstack([S_bar_C, -S_bar_C, np.eye(self.N * self.m), -np.eye(self.N * self.m)])
        G = np.vstack([S_bar_C,-S_bar_C,np.tril(np.ones((self.N * self.m, self.N * self.m))),-np.tril(np.ones((self.N * self.m, self.N * self.m)))])
        return G

    def tracker_S_std(self, T_bar_C):
        S = np.vstack([
            -T_bar_C,
            T_bar_C,
            np.zeros((self.N * self.m, self.n)),
            np.zeros((self.N * self.m, self.n))
        ])
        return S

    
    # B_In input bound constraints (dict): A dictionary containing the input bound constraints.
    # B_Out output bound constraints (dict): A dictionary containing the output bound constraints.
    def tracker_W_std(self, B_Out, B_In):
        # Check if 'min' fields are not empty and exist in B_Out and B_In
        out_min_present = 'min' in B_Out and B_Out['min'] is not None
        in_min_present = 'min' in B_In and B_In['min'] is not None

        # if out_min_present:
        #     if in_min_present:  # out min true, in min true
        #         block1 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        #         block2 = np.kron(np.ones((self.N, 1)), -B_Out['min'])
        #         block3 = np.kron(np.ones((self.N, 1)), B_In['max'])
        #         block4 = np.kron(np.ones((self.N, 1)), -B_In['min'])
        #     else:  # out min true, in min false
        #         W = np.vstack([
        #             np.kron(np.ones((self.N, 1)), B_Out['max']),
        #             np.kron(np.ones((self.N, 1)), -B_Out['min']),
        #             np.kron(np.ones((self.N, 1)), B_In['max']),
        #             np.kron(np.ones((self.N, 1)), B_In['max'])
        #         ])
        # elif in_min_present:  # out min false, in min true
        #     W = np.vstack([
        #         np.kron(np.ones((self.N, 1)), B_Out['max']),
        #         np.kron(np.ones((self.N, 1)), B_Out['max']),
        #         np.kron(np.ones((self.N, 1)), B_In['max']),
        #         np.kron(np.ones((self.N, 1)), B_In['min'])
        #     ])
        # else:  # out min false, in min false
        #     W = np.vstack([
        #         np.kron(np.ones((self.N, 1)), B_Out['max']),
        #         np.kron(np.ones((self.N, 1)), B_Out['max']),
        #         np.kron(np.ones((self.N, 1)), B_In['max']),
        #         np.kron(np.ones((self.N, 1)), B_In['max'])
        #     ])

        #if out_min_present:
            #if in_min_present:  # out min true, in min true
        block1 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        block2 = np.kron(np.ones((self.N, 1)), -B_Out['min'])
        block3 = np.kron(np.ones((self.N, 1)), B_In['max'])
        block4 = np.kron(np.ones((self.N, 1)), -B_In['min'])
        #     else:  # out min true, in min false
        #         block1 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        #         block2 = np.kron(np.ones((self.N, 1)), -B_Out['min'])
        #         block3 = np.kron(np.ones((self.N, 1)), B_In['max'])
        #         block4 = np.kron(np.ones((self.N, 1)), B_In['max'])  # Using max since min is not present
        # else:
        #     if in_min_present:  # out min false, in min true
        #         block1 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        #         block2 = np.kron(np.ones((self.N, 1)), B_Out['max'])  # Repeating max since min is not present
        #         block3 = np.kron(np.ones((self.N, 1)), B_In['max'])
        #         block4 = np.kron(np.ones((self.N, 1)), -B_In['min'])
        #     else:  # out min false, in min false
        #         block1 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        #         block2 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        #         block3 = np.kron(np.ones((self.N, 1)), B_In['max'])
        #         block4 = np.kron(np.ones((self.N, 1)), B_In['max'])

        vec1 = block1.flatten().reshape(-1, 1)  
        vec2 = block2.flatten().reshape(-1, 1)
        vec3 = block3.flatten().reshape(-1, 1)
        vec4 = block4.flatten().reshape(-1, 1)

        # Step 3: Vertically stack all vectors
        W = np.vstack([vec1, vec2, vec3, vec4])  
        return W

    # added function to compute constraints matrices
    def setConstraintsMatrices(self,B_in,B_out,S_bar_C,T_bar_C):

        self.G = self.tracker_G_std(S_bar_C)
        self.S = self.tracker_S_std(T_bar_C)
        self.W = self.tracker_W_std(B_out, B_in)
    

    def computesolution(self, x_ref, x_cur, u_cur, H, F_tra, initial_guess=None):
        """
        Compute optimal control sequence for MPC
        """
        # Check dimensions
        n_states = 2 * self.m
        expected_x_ref_size = self.N * n_states
        expected_x_cur_size = n_states
        expected_u_cur_size = self.m
        
        assert x_ref.size == expected_x_ref_size, f"x_ref size mismatch: {x_ref.size} vs expected {expected_x_ref_size}"
        assert x_cur.size == expected_x_cur_size, f"x_cur size mismatch: {x_cur.size} vs expected {expected_x_cur_size}"
        assert u_cur.size == expected_u_cur_size, f"u_cur size mismatch: {u_cur.size} vs expected {expected_u_cur_size}"
        
        # Form augmented state vector [x_ref; x_cur; u_cur]
        x0_mpc_aug = np.concatenate([x_ref, x_cur, u_cur])
        
        # Compute cost terms
        F = x0_mpc_aug @ F_tra
        
        def objective(z, H, F):
            return 0.5 * z.T @ H @ z + F @ z
        
        # Initial guess
        if initial_guess is None:
            z0 = np.zeros(self.m * self.N)
        else:
            z0 = initial_guess
        
        # Optimization options
        options = {
            'ftol': 1e-12,
            'maxiter': 1000,
            'disp': False
        }
        
        if self.constr_flag:
            def constraint(z, G, W, S, x0):
                return W.flatten() + S @ x0 - G @ z
            
            constraints = {'type': 'ineq',
                        'fun': constraint,
                        'args': (self.G, self.W, self.S, x_cur)} # Note: using x_cur not augmented state
        else:
            constraints = None

        # Solve optimization problem 
        result = minimize(objective, z0,
                        args=(H, F),
                        method='SLSQP',
                        constraints=constraints,
                        options=options)
        
        return result.x

    def compute_linearized_matrices(self, q_d, qd_d, dyn_model):
        """
        Compute linearized A_k and B_k matrices around desired trajectory point
        """
        num_joints = self.m
        num_states = 2 * num_joints  # Position and velocity states
        
        # Get Mass matrix at desired configuration
        dyn_model.ComputeMassMatrix(q_d)
        M = dyn_model.res.M
        if M.shape[0] > num_joints:  # If mass matrix includes base coordinates
            M = M[-num_joints:, -num_joints:]  # Extract joint portion only
        M_inv = np.linalg.inv(M)
        
        # Get Coriolis matrix at desired state 
        dyn_model.ComputeCoriolisMatrix(q_d, qd_d)
        C = dyn_model.res.N
        if C.shape[0] > num_joints:  # If Coriolis matrix includes base coordinates
            C = C[-num_joints:, -num_joints:]  # Extract joint portion only
        
        # Form A_k (2n x 2n)
        A_k = np.zeros((num_states, num_states))
        A_k[:num_joints, num_joints:] = np.eye(num_joints)
        A_k[num_joints:, num_joints:] = -M_inv @ C
        
        # Form B_k (2n x m)
        B_k = np.zeros((num_states, num_joints))
        B_k[num_joints:, :] = M_inv
        
        return A_k, B_k