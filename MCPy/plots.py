from MCPy.MCPy import *
import numpy as np
import math
import IPython.display as display
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def OneVariablePlot(LB, UB, nx, fun): 
    '''
    Description: plot the relaxations and the orginal for one-dimensional equation 
    Input: 
        LB -- lower bound
        UB -- upper bound
        nx -- number of partitions of X
        fun -- function
    Output:
        plot for the function and corresponding relaxations
    '''
    
    x_points = np.linspace(LB,UB,nx)
    cv_points = np.zeros(nx)
    cc_points = np.zeros(nx)
    y_points = np.zeros(nx)

    index = 0
    for x in x_points:
        x_MCPy = MCPy(np.array([LB,UB]), np.array([x, x]), np.matrix([[1, 1]]))
        f_MCPy = fun(x_MCPy)
        cv_points[index] = f_MCPy.MC[0]
        cc_points[index] = f_MCPy.MC[1]
        y_points[index] = fun(x)
        index += 1

    p1, = plt.plot(x_points, y_points, color="black") 
    p2, = plt.plot(x_points, cv_points, 'r--')        
    p3, = plt.plot(x_points, cc_points, 'b--''') 
    plt.legend([p1,p2,p3], ['f','convex','concave'])

    plt.show()
    

def TwoVariablesPlot(LB, UB, n_x1, n_x2, fun, view):  
    '''
    Description: plot the relaxations and the orginal for two-dimensional equation 
    Input: 
        LB   -- lower bounds
        UB   -- upper bounds
        n_x1 -- number of sampling points of X1
        n_x1 -- number of sampling points of X2
        fun  -- function
        view -- view point
    Output:
        plot for the function and corresponding relaxations
    '''
    
    x1_points = np.linspace(LB[0], UB[0], n_x1)
    x2_points = np.linspace(LB[1], UB[1], n_x2)

    cv_points = np.zeros((n_x1, n_x2))
    cc_points = np.zeros((n_x1, n_x2))
    y_points = np.zeros((n_x1, n_x2))

    index1 = 0
    index2 = 0
    for x1 in x1_points:
        for x2 in x2_points:
            x1_MCPy = MCPy(np.array([LB[0],UB[0]]), np.array([x1, x1]), np.matrix([[1, 1], [0, 0]]))
            x2_MCPy = MCPy(np.array([LB[1],UB[1]]), np.array([x2, x2]), np.matrix([[0, 0], [1, 1]]))
            f_MCPy = fun(x1_MCPy, x2_MCPy)
            cv_points[index1, index2] = f_MCPy.MC[0]
            cc_points[index1, index2] = f_MCPy.MC[1]
            y_points[index1, index2] = fun(x1, x2)
            index2 += 1
        index1 += 1
        index2 = 0

    fig = plt.figure(figsize=(8,5))
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(x1_points, x2_points)
    p1 = ax.plot_wireframe(X, Y, y_points.T,  rstride=1, cstride=1, color="black") 
    p2 = ax.plot_wireframe(X, Y, cv_points.T,  rstride=1, cstride=1, color="red", linestyle="dashed")        

    ax.view_init(view[0], view[1])
    ax.set_xticks(np.linspace(LB[0], UB[0], 5))                               
    ax.set_yticks(np.linspace(LB[1], UB[1], 5))                               
    ax.set_zticks([])
    ax.set_title('convex relaxations')
    ax.legend([p1,p2],['f','convex'])
    plt.show()

    fig = plt.figure(figsize=(8,5))
    ax = fig.gca(projection='3d')
    p3 = ax.plot_wireframe(X, Y, y_points.T,  rstride=1, cstride=1, color="black") 
    p4 = ax.plot_wireframe(X, Y, cc_points.T,  rstride=1, cstride=1, color="blue", linestyle="dashed") 
    ax.view_init(view[2], view[3])
    ax.set_xticks(np.linspace(LB[0], UB[0], 5))                               
    ax.set_yticks(np.linspace(LB[1], UB[1], 5))                               
    ax.set_zticks([])
    ax.set_title('concave relaxations')
    ax.legend([p3,p4],['f','concave'])
    plt.show()

    
def animation3D(LB, UB, n_x1, n_x2, fun, address1, address2):
    '''
    Description: make mp4 animations 
    Input: 
        LB   -- lower bounds
        UB   -- upper bounds
        n_x1 -- number of sampling points of X1
        n_x1 -- number of sampling points of X2
        fun  -- function
        address1 -- address of animation1
        address2 -- address of animation2
    Output:
        animation1
        animation2
    '''
        
    def rotate(angle):
        ax.view_init(azim=angle)

    x1_points = np.linspace(LB[0], UB[0], n_x1)
    x2_points = np.linspace(LB[1], UB[1], n_x2)

    cv_points = np.zeros((n_x1, n_x2))
    cc_points = np.zeros((n_x1, n_x2))
    y_points = np.zeros((n_x1, n_x2))

    index1 = 0
    index2 = 0
    for x1 in x1_points:
        for x2 in x2_points:
            x1_MCPy = MCPy(np.array([LB[0],UB[0]]), np.array([x1, x1]), np.matrix([[1, 1], [0, 0]]))
            x2_MCPy = MCPy(np.array([LB[1],UB[1]]), np.array([x2, x2]), np.matrix([[0, 0], [1, 1]]))
            f_MCPy = fun(x1_MCPy, x2_MCPy)
            cv_points[index1, index2] = f_MCPy.MC[0]
            cc_points[index1, index2] = f_MCPy.MC[1]
            y_points[index1, index2] = fun(x1, x2)
            index2 += 1
        index1 += 1
        index2 = 0

    fig = plt.figure(figsize=(8,5))
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(x1_points, x2_points)
    p1 = ax.plot_wireframe(X, Y, y_points.T,  rstride=1, cstride=1, color="black") 
    p2 = ax.plot_wireframe(X, Y, cv_points.T,  rstride=1, cstride=1, color="red", linestyle="dashed")             

    ax.set_xticks(np.linspace(LB[0], UB[0], 5))                               
    ax.set_yticks(np.linspace(LB[1], UB[1], 5))                               
    ax.set_zticks([])
    ax.set_title('convex relaxations')
    ax.legend([p1,p2],['f','convex'])
    
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360,4),interval=90)
    rot_animation.save(address1)
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(x1_points, x2_points)
    p1 = ax.plot_wireframe(X, Y, y_points.T,  rstride=1, cstride=1, color="black") 
    p2 = ax.plot_wireframe(X, Y, cc_points.T,  rstride=1, cstride=1, color="blue", linestyle="dashed")             

    ax.set_xticks(np.linspace(LB[0], UB[0], 5))                               
    ax.set_yticks(np.linspace(LB[1], UB[1], 5))                               
    ax.set_zticks([])
    ax.set_title('concave relaxations')
    ax.legend([p1,p2],['f','concave'])
    
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360,4),interval=90)
    rot_animation.save(address2)
    
    
def convergencePlot(x, fun):
    '''
    convergence plot for one-variable functions
    '''
    
    mag = 5
    eps_points = 10**(-np.linspace(3,2+mag,mag))
    cv_points = np.zeros(mag)
    cc_points = np.zeros(mag)
    delta_points = np.zeros(mag)

    index = 0
    for eps in eps_points:
        x_MCPy = MCPy(np.array([x-eps,x+eps]), np.array([x, x]), np.matrix([[1, 1]]))
        f_MCPy = fun(x_MCPy)
        cv_points[index] = f_MCPy.MC[0]
        cc_points[index] = f_MCPy.MC[1]
        delta_points[index] = f_MCPy.MC[1] - f_MCPy.MC[0]
        index += 1

    plt.loglog(eps_points, delta_points, color="red") 
    plt.grid(True,which="both",ls="-")
    plt.title('log-log scale convergence plot')
    plt.xlabel('half-width $\epsilon$')
    plt.ylabel('$f^{cv}(x)-f^{cc}(x)$')
    plt.show()