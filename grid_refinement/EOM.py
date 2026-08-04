import numpy as np

def impact_force(C,x1,x2):
    """
    calculates the force from the impactor
    """
    f=0
    if x1 > x2:
        f = C*(x1-x2)**(3/2)
    else:
        f = 0

    return f

def impact_EOM(t,x,nonlinear_spring_info, material_info, F):
    """
    calculates velocity and acceleration for all unit cells at all time

    ------
    inputs:
    ------
    x = [x,x_dot] = [displacement, velocity]
    t = time
    nonlinear_spring_info = [c1,c2,c3]
    material_info = [m,M_impact, C_impact, zeta] = [unit cell mass, impactor mass, contact spring coeff, damping coeff]

    ------
    outputs:
    ------
    dx = [x_dot, x_ddot] = [velocity, acceleration]
    """

    # initialization of parameters
    num_mass = len(x)//2
    dx = np.zeros(len(x))
    c1 = nonlinear_spring_info[0]
    c2 = nonlinear_spring_info[1]
    c3 = nonlinear_spring_info[2]

    m = material_info[0]
    M_impact = material_info[1]
    C_impact = material_info[2]
    zeta = material_info[3]

    ###
    # First Unit Cell contact with impactor
    ###
    i = 0
    dx[2*i] = x[2*i+1]
    dx[2*i+1] = (F-impact_force(C_impact,x[2*i], x[2*(i+1)]))/M_impact

    ###
    # Actual first unit cell 
    ###
    i=1
    dx[2*i] = x[2*i+1]
    if x[2*(i+1)] >= x[2*i]:
        dx[2*i+1] = (c1 * (x[2*(i+1)] - x[2*i]) 
                      + impact_force(C_impact, x[2*(i-1)], x[2*i]) 
                      - 2*zeta*(-x[2*(i+1)+1] + x[2*i+1])) / m

    else:
        dx[2*i+1] = (c1 * (x[2*(i+1)]-x[2*i]) 
                     - c2 * (x[2*(i+1)] - x[2*i])**2 
                     + c3 * (x[2*(i+1)] - x[2*i])**3 
                     + impact_force(C_impact, x[2*(i-1)], x[2*i]) 
                     - 2*zeta*(-x[2*(i+1)+1] + x[2*i+1]))/m

    ####
    # Middle Cells
    ####
    for i in range(2,num_mass-1):
        dx[2*i] = x[2*i+1]
        # force from right spring
        if x[2*(i+1)] >= x[2*i]:
            force_spring_R = c1*(x[2*(i+1)] - x[2*i])
        else:
            force_spring_R = (c1*(x[2*(i+1)] - x[2*i]) 
                              - c2*(x[2*(i+1)] - x[2*i])**2
                              + c3*(x[2*(i+1)] - x[2*i])**3)
        
        # force from left spring
        if x[2*i] >= x[2*(i-1)]:
            force_spring_L = - c1*(x[2*i] - x[2*(i-1)])
        else:
            force_spring_L = (- c1*(x[2*i] - x[2*(i-1)]) 
                              + c2*(x[2*i] - x[2*(i-1)])**2
                              - c3*(x[2*i] - x[2*(i-1)])**3)
        dx[2*i+1] = (force_spring_R+force_spring_L - 2*zeta*(-x[2*(i+1)+1] + 2*x[2*i+1]-x[2*(i-1)+1]))/m
    
    
    ###
    #last spring
    ###
    i = num_mass-1
    dx[2*i] = x[2*i+1]

    # force from the right spring
    if 0>= x[2*i]:
        force_spring_R = c1*(0-x[2*i])
    else:
        force_spring_R = (  c1*(0-x[2*i]) 
                          - c2*(0-x[2*i])**2 
                          + c3*(0-x[2*i])**3)
    
    # force from the right spring
    if x[2*i] >= x[2*(i-1)]:
        force_spring_L = -c1*(x[2*i] - x[2*(i-1)])
    else:
        force_spring_L = (- c1*(x[2*i] - x[2*(i-1)])
                          + c2*(x[2*i] - x[2*(i-1)])**2
                          - c3*(x[2*i] - x[2*(i-1)])**3)
    dx[2*i+1] = (force_spring_R + force_spring_L 
                 - 2*zeta*(x[2*i+1] - x[2*(i-1)+1])
                 - 2*zeta*(x[2*i+1]))/m

    return dx

