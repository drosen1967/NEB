import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import convolve2d, remez,oaconvolve
from scipy.interpolate import RegularGridInterpolator as RGI
from sympy import *
from PIL import Image
from minimization_methods import minimize_nag
import time

MtEvansPeak = np.array([2866,1255])
MtBierstadtPeak = np.array([693,1883])
#MtBierstadtPeak = np.array([3500,2500])

#{{{ NEB
def test_NEB():
    m = 10
    n = 2
    x = np.linspace(0,10,m)
    y = x * 2 + np.random.randn(m) 
    X = np.vstack([x,y]).T
    N = NEB(X, k = 1e-2,projected=False)
    N.plot()
    N.minimize()
    N.plot()

class NEB:
    '''
    optimziation locations stored by variable x as shape m by n
    '''
    def __init__(self, x, k=1e-2, spring_type = 'quadratic', projected = False):
        self.x , self.spring_type, self.k, self.projected = x, spring_type,k,projected
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.solved = False
        self.tol = 1e-6
        
    def spring_cost(self, x = None):
        if x is None: x = self.x[1:-1,:]
        X = self.x
        X[1:-1,:] = x.reshape(self.m-2,self.n)
        left  = X[0:-1,:]
        right = X[1:,  :]
        spring_potential = np.linalg.norm(left - right, axis=1)**2
        return np.sum(spring_potential * 0.5 * self.k)
    
    def quad_spring_gradient(self, x = None):
        if x is None: x = self.x[1:-1,:].flatten()
        X = self.x
        X[1:-1,:] = x.reshape(self.m-2,self.n)
        spring_force = self.k * ( 
                       (X[1:-1,:] - X[0:-2,:]) + 
                       (X[1:-1,:] - X[2:  ,:])
                       )
        return spring_force.flatten()
    
    def linear_spring_gradient(self, x = None):
        if x is None: x = self.x[1:-1,:].flatten()
        X = self.x
        X[1:-1,:] = x.reshape(self.m-2,self.n)
        spring_force = self.k * ( 
                       (X[1:-1,:] - X[0:-2,:])/np.linalg.norm(X[1:-1,:] - X[0:-2,:], axis=1)[:,None] + 
                       (X[1:-1,:] - X[2:  ,:])/np.linalg.norm(X[1:-1,:] - X[2:  ,:], axis=1)[:,None]
                       )
        return spring_force.flatten()
    
    def obj_cost(self,x = None):
        return 0
    
    def obj_gradient(self, x = None):
        return np.zeros( (self.m - 2) * self.n )
    
    def cost(self, x = None):
        return self.spring_cost(x) + self.obj_cost(x)
    
    def gradient(self, x = None):
        objective = self.obj_gradient(x)
        if self.spring_type == 'quadratic':
            spring = self.quad_spring_gradient(x) 
        if self.spring_type == 'linear':
            spring = self.linear_spring_gradient(x) 
        if x is None: x = self.x[1:-1,:].flatten()
        spring = spring.reshape(self.m-2,self.n)
        objective = objective.reshape(self.m-2,self.n)
        
        if self.projected == False: return (spring + objective).flatten()
        else:
            X = self.x
            X[1:-1,:] = x.reshape(self.m-2,self.n)
            tau =  X[2:,:] - X[0:-2,:]
            tau/=np.linalg.norm(tau,axis=1)[:,None]
            spring_par = np.sum(spring * tau,axis=1)[:,None] * tau
            object_par = np.sum(objective * tau,axis=1)[:,None] * tau
            object_perp = objective - object_par
            return (spring_par + object_perp).flatten()
    
    def plot(self):
        if self.n == 2:
            plt.scatter(self.x[0,0],self.x[0,1],color='b')
            plt.scatter(self.x[-1,0],self.x[-1,1],color='b')
            plt.scatter(self.x[1:-1,0],self.x[1:-1,1],color='r')
            plt.plot(self.x[:,0], self.x[:,1])
            plt.show()
                
            
    def minimize(self):
        x0 = self.x[1:-1,:].flatten()
        fun = self.cost
        jac = self.gradient
        result = minimize(fun,x0,jac=jac,method='BFGS', tol=1e-12, options=
                          dict(gtol=self.tol * self.k, norm=2, disp=True, maxiter = 10000, return_all=True))
        self.x[1:-1,:] = result.x.reshape(self.m-2,self.n)
        self.history = np.array(result.allvecs)
        self.solved = result.success

#}}}
#{{{ Caternary

def test_Catenary():
    m = 30
    n = 2
    x = np.linspace(0,10,m)
    y = np.ones_like(x) * 10
    X = np.vstack([x,y]).T
    N = Catenary(X, k = 100, mg = 10, projected=True)
    N.plot()
    N.minimize()
    N.plot()

class Catenary(NEB):
    def __init__(self, x, k=20, mg =5, spring_type = 'quadratic', projected = True):
        self.x , self.spring_type, self.k, self.mg, self.projected = x, spring_type,k,mg, projected
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.solved = False
        self.tol = self.m  * 1e-2
        
    def obj_cost(self, x = None):
        if x is None: x = self.x[1:-1,:]
        X = self.x
        X[1:-1,:] = x.reshape(self.m-2,self.n)
        return np.sum((X[:,1]) * self.mg)
    
    def obj_gradient(self, x = None):
        if x is None: x = self.x[1:-1,:].flatten()
        X = self.x
        X[1:-1,:] = x.reshape(self.m-2,self.n)
        spring_force = self.k * ( 
                       (X[1:-1,:] - X[0:-2,:]) + 
                       (X[1:-1,:] - X[2:  ,:])
                       ).flatten()
        vertical_force = np.ones_like(X[1:-1,:]) * self.mg 
        vertical_force[:,0] = 0 # Null the x forces
        return vertical_force.flatten() + spring_force
#}}}
def test_MinMap():
    m = 80
    blend = np.linspace(0, 1, m)
    X = np.outer(blend, MtEvansPeak) + np.outer(1-blend,MtBierstadtPeak)
    N = MinMap(X,filter = None, k = 0.5, projected=False)
    N.plot()
    N.minimize(method='NAG')
    N.plot()

class MinMap(NEB):
    def __init__(self, x, filter=None, k=.01, spring_type = 'quadratic', projected = True,mapfile = 'final.tif'
                 ):
        self.x , self.spring_type, self.k, self.projected = x, spring_type, k, projected
        self.mapfile = mapfile
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.solved = False
        self.eps = 1e-5
        self.image = np.array(Image.open(self.mapfile))
        if filter is not None:
            numtaps = 129
            filt = remez(numtaps, [0,filter-.02,filter+.02, .5],desired = [1,1e-4])
            filt = np.outer(filt,filt)
            plt.imshow(filt)
            plt.show()
            self.image = oaconvolve(self.image, filt,mode='same')
        self.map = RGI([np.linspace(0,self.image.shape[1]-1,self.image.shape[1]),
                        np.linspace(0,self.image.shape[0]-1,self.image.shape[0])],
                        self.image.T)

    def obj_cost(self, x = None):
        if x is None: x = self.x[1:-1,:]
        X = self.x
        X[1:-1,:] = x.reshape(self.m-2,self.n)
        values = self.map(X)
        return -np.sum(values)
        
    
    def obj_gradient(self, x = None):
        if x is None: x = self.x[1:-1,:].flatten()
        X = self.x
        X[1:-1,:] = x.reshape(self.m-2,self.n)
        G = np.zeros_like(X)
        g0 = self.obj_cost(X[1:-1])
        for i in range(self.m-2):
            e = np.zeros_like(X)
            e[i+1,0] = self.eps
            g1 = self.obj_cost((X + e)[1:-1,:])
            G[i+1,0] = g1-g0
            e[i+1,0] = 0.0
            e[i+1,1] = self.eps
            g1 = self.obj_cost((X + e)[1:-1,:])
            G[i+1,1] = g1-g0
        return G[1:-1].flatten()/self.eps

    def minimize(self, method = None):
        x0 = self.x[1:-1,:].flatten()
        fun = self.cost
        if method is None:
            result = minimize(fun,x0,method='Nelder-Mead')
            self.x[1:-1,:] = result.x.reshape(self.m-2,self.n)
            self.solved = result.success
        elif method == 'NAG':
            result = minimize_nag(fun,x0,jac=self.gradient, stepsize = 1e-5, mom_parameter=0.1)
            self.x[1:-1,:] = result['x']
        print(result)

    def plot(self):
        plt.imshow(self.image,cmap='jet')
        plt.scatter(self.x[0,0],self.x[0,1],color='b')
        plt.scatter(self.x[-1,0],self.x[-1,1],color='b')
        plt.scatter(self.x[1:-1,0],self.x[1:-1,1],color='r')
        plt.plot(self.x[:,0], self.x[:,1])
        plt.show()

    
        

if __name__=='__main__':
    #test_NEB()
    #test_Catenary()
    test_MinMap()

