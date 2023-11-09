import jax
import jax.numpy as jnp
from jax import random,  vmap, jit
from jax.config import config
from jax import lax
from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, linalg, csc_matrix, csr_matrix


def gen_initX_ball(key,J,r,pos):
    k1,k2,k3=random.split(key,3)
    #
    X=random.normal(k1,(J,3))
    X=X/jnp.linalg.norm(X,axis=1,keepdims=True)
    X=X*(random.uniform(k2,(J,1))**(1/3))*r
    X=X+random.choice(k3,pos,shape=(J,))
    '''Test below'''
    plt.hist2d(X[:,0],X[:,1])
    return X

def gen_initX(key,J,r,pos,plot=False):
    k1,k2,k3=random.split(key,3)
    #
    X=random.normal(k1,(J,3))
    X=X/jnp.linalg.norm(X,axis=1,keepdims=True)
    X=X*(random.uniform(k2,(J,1))**(1/3))*r
    X=X+random.choice(k3,pos,shape=(J,))
    if plot:
        L=5
        plt3d_alpha=0.3
        plt3d_range=np.array([-L/4,L/4])
        fig=plt.figure(figsize=(6,2))
        ax1=fig.add_subplot(131)
        c1=ax1.hist2d(X[:,0],X[:,1],bins=40,range=[[-L/2,L/2],[-L/2,L/2]])[3]
        plt.colorbar(c1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        ax2=fig.add_subplot(132)
        c2=ax2.hist2d(X[:,0],X[:,3],bins=40,range=[[-L/2,L/2],[-L/2,L/2]])[3]
        plt.colorbar(c2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        
        ax3=fig.add_subplot(133,projection='3d')
        ax3.plot3D(X[:,0],X[:,1],X[:,2],',', alpha=plt3d_alpha)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')
        
        ax3.set_xlim(plt3d_range)
        ax3.set_ylim(plt3d_range)
        ax3.set_zlim(plt3d_range)
        
        plt.show()
        
    return X

def FDM_KS(M0, N= 5000,eps=1e-4,k=1e-1,mu=1.0,chi=1.0,T=1,dt=1e-5,R=20):
    r = np.linspace(0, R, N+2) # Spatial grid
    dt = 1e-4  # Time step size
    dr = r[1] - r[0]  # Spatial step size

    Nt = int(T/dt)  # Number of iterations
    rho_supp=np.where(r<=1, 1, 0)
    rho0=rho_supp/np.sum(rho_supp * 4*np.pi* r **2 *dr) *M0
    rho=np.copy(rho0)
    c=np.zeros(N+2)
    # Create the matrix A
    C_diag = np.zeros(N)
    C_upper = np.zeros(N-1)
    C_lower = np.zeros(N-1)
    rho_diag = np.zeros(N)
    rho_upper = np.zeros(N-1)
    rho_lower = np.zeros(N-1)
    b_rho = np.zeros(N)
    b_c = np.zeros(N)
    clog=np.zeros(Nt)
    
    for it in range(Nt):

        rho_diag[0]= 1 + dt * mu / dr ** 2 + 2 * mu * dt / (r[1] * dr)
        rho_diag[0]+= -chi * dt * (c[2]-c[1]) / dr**2 + chi * dt * (c[2]-2*c[1]+c[0]) / dr ** 2 + 2*chi * dt * (c[2]-c[1]) / (r[1] * dr)
        rho_diag[1:N-1] = 1 + 2 * dt * mu / dr ** 2 + 2 * mu * dt / (r[2:N] * dr)
        rho_diag[1:N - 1] += - chi * dt * np.diff(c[2:N + 1]) / dr**2 + chi * dt * np.diff(np.diff(c[1:N + 1])) / dr ** 2 + 2 * chi * dt * np.diff(c[2:N + 1]) / (r[2:N] * dr)
        rho_diag[N - 1] = 1 + dt * mu / dr ** 2
        rho_diag[N - 1] += chi * dt * (c[N] - 2 * c[N - 1] + c[N - 2]) / dr ** 2 + 2 * chi * dt * (c[N] - c[N - 1]) / (r[N ] * dr)
        rho_upper = -dt * mu / dr ** 2 - 2*mu * dt / (r[1:N] * dr) + dt * chi * np.diff(c[1:N+1]) / dr ** 2
        rho_lower = -dt * mu / dr ** 2
        rho_matrix = diags([rho_upper, rho_diag, rho_lower], [1, 0, -1])
        rho_matrix = csc_matrix(rho_matrix)

        b_rho = rho[1:N+1]
        rho[1:N+1] = linalg.spsolve(rho_matrix, b_rho)
        rho[0] = rho[1]
        rho[-1] = rho[-2]

        C_diag[0]= 1 + dt/eps * (1/ dr ** 2 + 2 / (r[1] * dr)+k**2)
        C_diag[1:N-1] = 1 + dt/eps * (2/ dr ** 2 + 2 / (r[2:N] * dr)+k**2)
        C_diag[N-1] = 1 + dt / eps * (1 / dr ** 2 + k ** 2)
        C_upper = -dt/eps * (1/ dr ** 2 + 2 / (r[1:N] * dr))
        C_lower = -dt/eps * 1 / dr ** 2
        C = diags([C_upper, C_diag, C_lower], [1, 0, -1])
        C = csc_matrix(C)

        b_c = c[1:N+1] + dt * rho[1:N+1]/eps
        c[1:N+1] = linalg.spsolve(C, b_c)
        c[0]=c[1]
        c[-1]=c[-2]
        clog[it]=c[0]
    return np.arange(Nt)*dt,clog,c,r,rho