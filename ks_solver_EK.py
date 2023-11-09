import jax
import jax.numpy as jnp
from jax import random,  vmap, jit
from jax.config import config
# from jax.ops import index_update, index
from jax import lax
from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt

import queue


## Fully Elliptic Kernel Formulation


class TDKS:
    def __init__(self,\
                 M0=100,H=24,L=8,k=0.1,eps=1e-4,\
                 reg=0,chi=1,mu=1,rng_key=random.PRNGKey(123)):
        start_time = time.time()
        self.H=H
        self.L=L
        self.k=k
        self.eps=eps
        self.reg=reg
        self.chi=chi
        self.mu=mu
        self.key=rng_key

        # computation
        self.M0=M0
        self.plt3d_alpha=0.1
        self.plt3d_range=np.array([-self.L/4,self.L/4])
        self.hist3d_range=((-self.L/4,self.L/4),(-self.L/4,self.L/4),(-self.L/4,self.L/4))
        cmesh0=jnp.linspace(-0.5,0.5,self.H,endpoint=False)*self.L
        self.cmeshlen=self.L/self.H
        
        self.cmeshx,self.cmeshy,self.cmeshz=jnp.meshgrid(cmesh0,cmesh0,cmesh0,indexing='ij')
        self.intfac3d=(self.L/self.H)**3
        self.cmesh=jnp.concatenate((self.cmeshx.reshape(-1,1),self.cmeshy.reshape(-1,1),self.cmeshz.reshape(-1,1)),axis=1)
        cspec0=jnp.concatenate((jnp.linspace(0,self.H/2,int(self.H/2)+1),-jnp.linspace(self.H/2-1,1,int(self.H/2)-1)))/self.L*2*jnp.pi # Fourier Variale Ahead of x
        self.cspec=jnp.meshgrid(cspec0,cspec0,cspec0,indexing='ij')
        # initialization
        self.Calpha=jnp.zeros((self.H,self.H,self.H))
        self.key, subkey = random.split(self.key)
        
        # logs
        self.varlog=[]
        self.tlog=[]
        self.cmaxlog=[]
        self.cintlog=[]
        self.rhomaxlog=[]
        print("--- Build time %s s ---" % (time.time() - start_time))
        
 
    def solve(self,t,dt,show_num=5,show_flag=True,single_plot=False):
        # beta version, use Hankel tranform to update C
        start_time0 = time.time()
        nt=int(t/dt)
        # initialize for n-1 steps
                    
        self.J=self.X.shape[0]
        self.M=self.M0/self.J
        indj,indl,indm=self.cspec
        
        self.cspec_flatten=jnp.array([indj.flatten(),indl.flatten(),indm.flatten()])
        cspec_sgn0=jnp.array((-1)**jnp.arange(self.H))
        self.cspec_sng=jnp.ones((self.H,self.H,self.H))*cspec_sgn0.reshape((self.H,1,1))*cspec_sgn0.reshape((1,self.H,1))*cspec_sgn0.reshape((1,1,self.H))
            
            # self.cspec_flatten=jnp.array([indj1,indl1,indm1]).reshape((3,self.H*self.H*self.H))
        self.beta=jnp.sqrt(self.k**2+self.eps/dt)
        self.GkernelCoeff=-1/(indm**2+indj**2+indl**2+self.beta**2) 
        start_time = time.time()
        for it in range(nt):
            ''' plot and show '''
            t=(it)*dt
            if show_num != 0:
                if it % int(nt/show_num) == 0:
                    if show_flag:
                        self.show_sol_x(single_plot=single_plot)
                        
                    print("t= %.2e, step %s s, total %s s; Var %.2e; E(intC) %.2e" % \
                          (t,time.time() - start_time,time.time() - start_time0,jnp.var(self.X), \
                           jnp.real((self.Calpha[0,0,0]*self.L**3))/ self.cal_Cint(t)-1))
                    start_time = time.time()
            
            ''' Itegrator '''
            self.key, subkey = random.split(self.key)
            # n+1 step X
            self.X1=self.onestepX(self.X,self.Calpha,dt,subkey)
            # n step C 
            self.Calpha=self.onestepC_hankel(self.X,self.Calpha,dt)
            #self.Calpha=self.onestepC(self.X,self.Calpha,dt)
            self.X=self.X1
            
            ''' logs '''
            self.varlog.append(jax.device_get(jnp.var(self.X)*self.M0))
            self.tlog.append(t+dt)
            self.cmaxlog.append(jnp.max(jnp.abs(jnp.fft.ifftn(self.Calpha)*self.H**3)))
            self.cintlog.append(jnp.real((self.Calpha[0,0,0]*self.L**3))/ self.cal_Cint(t+dt)-1)
            histdd,_=jnp.histogramdd(self.X,bins=self.H,range=self.hist3d_range,density=True)
            self.rhomaxlog.append(jnp.max(histdd))
            
            
        # show final sol
        if show_flag:
            self.show_sol_x(single_plot=single_plot)
        print("dt= %.2e, t= %.2e, total %s s; Var %.2e;  E(intC) %.2e; Cmax %.2e" % (dt,t+dt,time.time() - start_time0,jnp.var(self.X),jnp.real((self.Calpha[0,0,0]*self.L**3))/ self.cal_Cint(t+dt)-1,self.cmaxlog[-1]))
        start_time = time.time()
        return
    
    '''Integrators'''
    
    # Update C     # Elliptic Kernel Formulation
    @partial(jit, static_argnums=(0))
    def onestepC(self,X,Calpha,dt):
        K_conv_rho=self.vgkernel(self.cmesh,X)/4/jnp.pi*self.M
        Calpha1=-self.GkernelCoeff*Calpha*self.eps/dt-jnp.fft.fftn(K_conv_rho.reshape(self.H,self.H,self.H))/self.H**3
        return lax.stop_gradient(Calpha1)
    
    def tmp1(self,X,Calpha,dt):
        K_conv_rho=self.vgkernel(self.cmesh,X)/4/jnp.pi*self.M
        return jnp.fft.fftn(K_conv_rho.reshape(self.H,self.H,self.H))/self.H**3
        
    def vgkernel(self,cmesh,x):
        val=jnp.nan_to_num(vmap(self.vgkernel1,(0,None))(cmesh,x))
        return val
    def vgkernel1(self,cmesh,x):
        valk=jnp.nansum(vmap(self.gkernel,(None,0))(cmesh,x))
        return valk
    def gkernel(self,X1,X2):
        absX=jnp.sqrt(jnp.sum((X1-X2)**2))
        return -jnp.exp(-absX*self.beta)/(absX+self.reg)
    
    # utilize Hankel transform of Kernel
    @partial(jit, static_argnums=(0))
    def onestepC_hankel(self,X,Calpha,dt):
        K_conv_rho_hankel=-vmap(self.hankel_kernel,(1,None))(self.cspec_flatten,X)*self.M/self.L**3
        Calpha1=-self.GkernelCoeff*Calpha*self.eps/dt \
        -(K_conv_rho_hankel.reshape(self.H,self.H,self.H))* self.cspec_sng
        return lax.stop_gradient(Calpha1)
    
    
    def hankel_kernel(self,k,X):
        k0,k1,k2=k
        F=jnp.sum(vmap(self.hankel_kernel1,(None,0))(k,X))/(self.beta**2+(k0**2+k1**2+k2**2))
        return F
    
    def hankel_kernel1(self,k,x):
        k0,k1,k2=k
        x0,x1,x2=x
        return jnp.exp(1j*(-k0*x0-k1*x1-k2*x2))
    
    
    # Update X
    
    @partial(jit, static_argnums=(0))
    def onestepX(self,X,Calpha,dt,key):     
        # K*C
        crossterm1=vmap(self.cal_gradK_C,(0,None))(X,Calpha)
    
        X1=X-self.chi*self.eps*crossterm1 #K*C
        
        # K*rho, instant attraction
        crossterm2=vmap(self.cal_crossterm2,(0,None))(X1,X1)
        X2=X1-dt*self.chi*crossterm2 
        
        VfuncVals=vmap(self.Vfunc,(0))(X2)
        X3=X2+dt*VfuncVals
        
        X4=jnp.clip(X3+jnp.sqrt(2*self.mu*dt)*random.normal(key,(self.J,3)),-self.L/2,self.L/2)
        return lax.stop_gradient(X4)
    
    
    def gradK(self,X):
        xnorm=jnp.linalg.norm(X,axis=1,keepdims=True)
        return jnp.nan_to_num(X/xnorm/(xnorm**2+self.reg)*jnp.exp(-self.beta*xnorm)*(self.beta*(xnorm+self.reg)+1)/4/jnp.pi)
    
    def cal_gradK_C(self,X,Calpha):  # gradK * C, while mesh re-assignment
        X=X.reshape(1,3)
        Xh=self.cmeshlen/2-jnp.mod(X,self.cmeshlen) # 1 * 3; bar X n,p in the paper
        gradKvals=self.gradK(X+Xh-self.cmesh) # H^3 * 3
        Calpha1=Calpha*jnp.exp(1j*(-Xh[0,0]*self.cspec[0]-Xh[0,1]*self.cspec[1]-Xh[0,2]*self.cspec[2]))
        Cvals=jnp.fft.ifftn(Calpha1).reshape((-1))*self.H**3
        crossterm1=jnp.real(jnp.dot(Cvals,gradKvals))*self.intfac3d
        return crossterm1
    
    def cal_crossterm2(self,X,Xall):
        X=X.reshape(1,-1)
        crossterm2=jnp.nansum(self.gradK(X-Xall),axis=0)*self.M
        return crossterm2
    
    ''' Model Validation '''   
    
    def cal_Cint(self,t):
        return self.M0/self.k**2*(1-np.exp(-self.k**2/self.eps * t))
        
    def show_sol_x(self,fig=None,single_plot=False):
        X=jax.device_get(self.X)
        Calpha=jax.device_get(self.Calpha)
        cval=np.real(np.fft.ifftn(Calpha))*self.H**3;
        if single_plot:
            # Plots that is shown in the manuscript
            # fig1 histogram of x,y
            fig1=plt.figure(figsize=(4,3),layout='constrained')
            ax1=fig1.add_subplot(111)
            c1=ax1.hist2d(X[:,0],X[:,1],bins=40,range=[[-self.L/2,self.L/2],[-self.L/2,self.L/2]])[3]
            plt.colorbar(c1)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            # c(x,y,0)
            fig2=plt.figure(figsize=(4,3),layout='constrained')
            ax2=fig2.add_subplot(111)
            halfind=int(self.H/2)
            c2=ax2.contourf(self.cmeshx[:,:,halfind],self.cmeshy[:,:,halfind],cval[:,:,halfind])
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            plt.colorbar(c2)
            # 3D Scattering
            fig3=plt.figure(figsize=(4,3))
            ax3=fig3.add_subplot(111,projection='3d')
            ax3.plot(X[:,0],X[:,1],X[:,2], '.',alpha=0.3,fillstyle='full',markersize=1.5,markeredgewidth=0)
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_zlabel('z')
            ax3.set_xlim(self.plt3d_range)
            ax3.set_ylim(self.plt3d_range)
            ax3.set_zlim(self.plt3d_range)
            # Plt right now
            plt.show()
            
                            
        else:
            if fig is None:
                fig=plt.figure(figsize=(6,2))
            ax1=fig.add_subplot(131)
            c1=ax1.hist2d(X[:,0],X[:,1],bins=40,range=[[-self.L/2,self.L/2],[-self.L/2,self.L/2]])[3]
            plt.colorbar(c1)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            halfind=int(self.H/2)
            ax4=fig.add_subplot(132)
            c4=ax4.contourf(self.cmeshx[:,:,halfind],self.cmeshy[:,:,halfind],cval[:,:,halfind])
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')
            plt.colorbar(c4)

            ax3=fig.add_subplot(133,projection='3d')
            ax3.plot3D(X[:,0],X[:,1],X[:,2],',', alpha=self.plt3d_alpha)
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_zlabel('z')

            ax3.set_xlim(self.plt3d_range)
            ax3.set_ylim(self.plt3d_range)
            ax3.set_zlim(self.plt3d_range)

            plt.show()
            fig.canvas.draw()
        
    def Calpha_compare(self,ref):
        C=jax.device_get(self.Calpha)
        return jax.device_get(jnp.linalg.norm(C-ref)/jnp.linalg.norm(ref))
    def Variance_compare(self,ref):
        return jax.device_get(jnp.abs(jnp.var(self.X)-ref)/jnp.abs(ref))
        