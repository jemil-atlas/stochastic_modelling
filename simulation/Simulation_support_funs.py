"""
The goal of this set of functions is to provide fast and accessible simulation
tools

"""

import numpy as np
import random




"""
The goal of this function is to simulate a realization of a zero-mean random 
field based on a tensordecomposition. Provided the correlation structures
in x and y direction, the simulation proceeds to patch together a Karhunen-
Loewe decomposition of the random field based on the spectral decompositions
of the Covariance matrices associated to the individual dimensions.
For this, do the following:
    1. Definitions and imports
    2. Spectral decompositions and setups
    3. Simulation loop
    4. Assemble solution
    
INPUTS
    Name                 Interpretation                             Type
x               1D array of x values                            Array
y               1D array of y values                            Array
cov_x           Covariance function in x direction              Function
cov_y           Covariance function in y direction              Function

OUTPUTS
    Name                 Interpretation                             Type
random_field    2D realization of a random field                Array
K_x             2D array of Covariance values                   Array
K_y             2D array of Covariance values                   Array

Example usage:
x=np.linspace(0,1,10)
y=np.linspace(0,2,20)

def cov_fun(s,t):
    return np.exp(-(np.abs(s-t)/0.2)**2)
RF, K_x, K_y= Simulate_random_field_fast(x,y,cov_fun,cov_fun)
"""






def Simulate_random_field_fast(x,y,cov_x,cov_y):
    
    
    
    """
    1. Definitions and imports -----------------------------------------------
    """
    
    
    # i) Definitions
    
    n_x=np.shape(x)[0]
    n_y=np.shape(y)[0]
    
    K_x=np.zeros([n_x,n_x])
    K_y=np.zeros([n_y,n_y])
    
    for k in range(n_x):
        for l in range(n_x):
            K_x[k,l]=cov_x(x[k],x[l])
        
    for k in range(n_y):
        for l in range(n_y):
            K_y[k,l]=cov_y(y[k],y[l])
    
    

    """
    2. Spectral decompositions and setups ------------------------------------
    """
    
    
    # i) Spectral decomposition
    
    u_x,l_x,v_x = np.linalg.svd(K_x,hermitian=True)
    u_y,l_y,v_y = np.linalg.svd(K_y,hermitian=True)
    
    l_xy=np.kron(l_x,l_y)
    
    
    # ii) Sort 
    
    l_xy_sorted=np.sort(l_xy)[::-1]
    sort_index=np.argsort(l_xy)[::-1]
    sort_index_multi=np.unravel_index(sort_index, [n_x,n_y])
    
    
    """
    3. Simulation loop -------------------------------------------------------
    """
    
    
    # i) Threshold noise variables for 95% explained var
    
    l_total=np.sum(l_xy_sorted)
    l_cumsum=np.cumsum(l_xy_sorted)
    
    n_max=np.argmax(l_cumsum>=0.95*l_total)
    
    
    # ii) Loop over eigenvectors
    
    xi=np.random.normal(0,1,[n_max])
    rf=np.zeros([n_x,n_y])
    
    for k in range(n_max):
        eigenvec=np.kron(u_x[:,sort_index_multi[0][k]],u_y[:,sort_index_multi[1][k]]).reshape([n_x,n_y])
        delta=xi[k]*l_xy_sorted[k]*eigenvec
        rf=rf+delta
    
    
    
    """    
    4. Assemble solution -----------------------------------------------------    
    """
    
    
    # i) Formulate solution
    
    random_field=rf
    
    
    return random_field, K_x, K_y





def Simulate_random_field_3D_fast(x, y, z, cov_x, cov_y, cov_z):
    
    
    
    """
    1. Definitions and imports -----------------------------------------------
    """
    
    
    # i) Definitions
    
    n_x=np.shape(x)[0]
    n_y=np.shape(y)[0]
    n_z=np.shape(z)[0]
    
    K_x=np.zeros([n_x,n_x])
    K_y=np.zeros([n_y,n_y])
    K_z=np.zeros([n_z,n_z])
    
    for k in range(n_x):
        for l in range(n_x):
            K_x[k,l]=cov_x(x[k],x[l])
        
    for k in range(n_y):
        for l in range(n_y):
            K_y[k,l]=cov_y(y[k],y[l])
            
    for k in range(n_z):
        for l in range(n_z):
            K_z[k,l]=cov_z(z[k],z[l])
    
    

    """
    2. Spectral decompositions and setups ------------------------------------
    """
    
    
    # i) Spectral decomposition
    
    u_x,l_x,v_x = np.linalg.svd(K_x,hermitian=True)
    u_y,l_y,v_y = np.linalg.svd(K_y,hermitian=True)
    u_z,l_z,v_z = np.linalg.svd(K_z,hermitian=True)
    
    l_xy=np.kron(l_x,l_y)
    l_xyz=np.kron(l_xy,l_z)
    
    # ii) Sort 
    
    l_xyz_sorted=np.sort(l_xyz)[::-1]
    sort_index=np.argsort(l_xyz)[::-1]
    sort_index_multi=np.unravel_index(sort_index, [n_x,n_y,n_z])
    
    
    """
    3. Simulation loop -------------------------------------------------------
    """
    
    
    # i) Threshold noise variables for 95% explained var
    
    l_total=np.sum(l_xyz_sorted)
    l_cumsum=np.cumsum(l_xyz_sorted)
    
    n_max=np.argmax(l_cumsum>=0.95*l_total)
    
    
    # ii) Loop over eigenvectors
    
    xi=np.random.normal(0,1,[n_max])
    rf=np.zeros([n_x,n_y,n_z])
    
    for k in range(n_max):
        eigenvec=np.kron(np.kron(u_x[:,sort_index_multi[0][k]],u_y[:,sort_index_multi[1][k]]),u_z[:,sort_index_multi[2][k]]).reshape([n_x,n_y,n_z])
        delta=xi[k]*l_xyz_sorted[k]*eigenvec
        rf=rf+delta
    
    
    
    """    
    4. Assemble solution -----------------------------------------------------    
    """
    
    
    # i) Formulate solution
    
    random_field=rf
    
    
    return random_field, K_x, K_y, K_z















































