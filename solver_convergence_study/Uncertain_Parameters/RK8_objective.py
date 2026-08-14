import numpy as np
from scipy.integrate import solve_ivp
import EOM

def objective(non_spring_coeff_info, stochastic_info, n_samples):
    """
    Objective finds the average and the std KE ratio for the random input space for a given c2,c3 pair

    ------
    inputs:
    ------
    non_spring_coeff_info = [c2,c3] 
    stochastic_info = [M_normal, V_normal, zeta_sweep]
        M_normal = 0.045 + 0.015*(2*rand(n_samples,1) - 1);       
        V_normal = 0.8 + 0.3*(2*rand(n_samples,1) - 1);          
        zeta_sweep = 0.015 + 0.01*(2*rand(n_samples,1) - 1);KE_ratio
    n_samples = number of random samples from the input space

    ------
    outputs:
    ------
    KE_ratio = average KE ratio for stochastic system given c2, c3
    KE_std = standard devation of KE ratio for stochastic system given c2, c3
    """
    ############################################################
    # getting info
    ############################################################

    N = 20;                                             # number of unit cells
    
    # random inputs
    M_normal   = stochastic_info[0]
    V_normal   = stochastic_info[1]
    zeta_sweep = stochastic_info[2]

    #unit cell mass - nondimensionally defined
    m = 1

    # nonlinear spring coeffs
    c1 = 1
    c2 = non_spring_coeff_info[0]
    c3 = non_spring_coeff_info[1]

    non_spring_info = [c1,c2,c3]

    lin_spring_info = [c1,0,0]

    # defined impactor mass
    M0_impactor = N/2

    ############################################################
    # ODE Setup
    ############################################################
    k_f = max(1,1*c1+2*c2+3*c3)
    f0 = 1/(2*np.pi)*np.sqrt(k_f/M0_impactor)
    cycles = 50
    output_per_cyc = 5000

    f = 1/(2*np.pi)*np.sqrt(k_f/M0_impactor)
    dt_cyc = 1/f
    dt = dt_cyc/output_per_cyc
    T = (1/f0)*cycles

    t_eval = np.arange(0,T,dt)

    ############################################################
    # desired values
    ############################################################
    KE_ratios = np.zeros(n_samples)

    for k in range(n_samples):
        ############################################################
        # Materials
        ############################################################
        # Random Variables
        alpha = M_normal[k]
        beta = V_normal[k]
        zeta = zeta_sweep[k]

        # Impactor Properties
        V0_impactor = 1
        M_impactor = alpha*M0_impactor
        V_impactor = beta/V0_impactor
        C_impact = 1.2e4

        #Material Properties
        mat_info = [m, M_impactor, C_impact, zeta]

        # initial values
        init_vals = np.zeros(2*(N+1))
        init_vals[1] = V_impactor

        ############################################################
        # Solving ODE
        ############################################################
        non_sol = solve_ivp(
            fun = lambda t, x: EOM.impact_EOM(t,x,non_spring_info, mat_info,0),
            t_span = (0,T),
            y0 = init_vals,
            method='DOP853',
            t_eval=t_eval,
            rtol = 1e-8,
            atol = 1e-8*np.ones(2*(N+1))
        )
        t_non = non_sol.t
        X_non = non_sol.y.T

        x_dot_non = X_non[:,1::2]
        x_non = X_non[:,0::2]

        lin_sol = solve_ivp(
            fun = lambda t, x: EOM.impact_EOM(t,x,lin_spring_info, mat_info,0),
            t_span = (0,T),
            y0 = init_vals,
            method='DOP853',
            t_eval=t_eval,
            rtol = 1e-8,
            atol = 1e-8*np.ones(2*(N+1))
        )
        t_lin = lin_sol.t
        X_lin = lin_sol.y.T

        x_dot_lin = X_lin[:,1::2]
        x_lin = X_lin[:,0::2]

        ############################################################
        # KE Calculations
        ############################################################
        KE_non = 1/2*m*x_dot_non[:,-1]**2
        KE_lin = 1/2*m*x_dot_lin[:,-1]**2

        max_KE_non = max(KE_non)
        max_KE_lin = max(KE_lin)

        KE_ratios[k] = max_KE_non/max_KE_lin

    ############################################################
    # Calculate Cost Function
    ############################################################
    KE_avg = np.average(KE_ratios)
    
    # only taking upper standard deviation into account because thats what we want to minimize
    KE_upper_vals = KE_ratios[KE_ratios > KE_avg]
    KE_std = 0
    if n_samples >1:
        KE_std = np.std(KE_upper_vals, ddof=1)

    if KE_std >0:
        J_function = KE_avg + KE_std
    else:
        J_function = KE_avg

    return [J_function, KE_avg, KE_std, KE_ratios]

