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
Downhill = np.array([3800, 2500])
#Downhill = np.array([1312, 1371])
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
#{{{ MinMap
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

#}}}    
#{{{ 
def test_ConstantSlope():
    m = 5000
    blend = np.linspace(0, 1, m)
    X = np.outer(blend, MtEvansPeak) + np.outer(1-blend,Downhill)
    N = ConstantSlope(X,filter = .45, k = 10e2, projected=False)
    elevation_change = np.abs(np.diff(N.map(np.vstack([MtEvansPeak, Downhill]))))
    distance = np.linalg.norm(MtEvansPeak - Downhill) * 5
    av_slope = elevation_change/distance
    N.goal_slope = av_slope * 1/4
    N.plot()
    N.minimize(method='NAG')
    N.plot()
    N.plot_slope()

class ConstantSlope(NEB):
    def __init__(self, x, filter=None, k=.01, spring_type = 'quadratic', projected = True,mapfile = 'final.tif'
                 ):
        self.x , self.spring_type, self.k, self.projected = x, spring_type, k, projected
        self.mapfile = mapfile
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.solved = False
        self.magnify = 1000
        #self.goal_slope = .25
        self.eps = 1e-5
        self.image = np.array(Image.open(self.mapfile))
        self.original = np.copy(self.image)
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
        delta_run = np.linalg.norm(X[1:,:] - X[:-1,:],axis=1)
        delta_rise = values[1:] - values[:-1]
        slope = delta_rise/delta_run
        difference = self.goal_slope - slope
        return np.sum(difference**2) * self.magnify

    def plot_slope(self):
        x = None
        if x is None: x = self.x[1:-1,:]
        X = self.x
        X[1:-1,:] = x.reshape(self.m-2,self.n)
        values = self.map(X)
        delta_run = np.linalg.norm(X[1:,:] - X[:-1,:],axis=1)
        delta_rise = values[1:] - values[:-1]
        slope = delta_rise/delta_run
        plt.plot(slope)
        plt.show()
        
    
    def obj_gradient(self, x = None):
        if x is None: x = self.x[1:-1,:].flatten()
        X = self.x
        X[1:-1,:] = x.reshape(self.m-2,self.n)
        step_right = np.vstack([self.eps * np.ones(X.shape[0]),np.zeros(X.shape[0])]).T
        step_up    = np.vstack([np.zeros(X.shape[0]),self.eps * np.ones(X.shape[0])]).T
        center = np.copy(X)
        up = np.copy(X) + step_up
        right = np.copy(X) + step_right
        center_height = self.map(X)
        hill_gradient = np.vstack([self.map(up) - center_height, self.map(right) - center_height]).T * 1./self.eps
        # Compute gradient from sheet:
        backward_point = X[:-2,:]
        forward_point = X[2:,:]
        center_point = X[1:-1,:]
        center_slope = hill_gradient[1:-1,:]
        backward_height = center_height[:-2]
        forward_height = center_height[2:]
        center_height = center_height[1:-1]

        center_to_back_distance = np.linalg.norm(center_point - backward_point,axis=1)
        center_to_back_preamble    = 2 * (center_height - backward_height)/center_to_back_distance - self.goal_slope
        center_to_back_numerator_01 = center_slope * center_to_back_distance[:,None]
        center_to_back_numerator_02 = -1./2*(center_height - backward_height)[:,None] * (center_point - backward_point)/center_to_back_distance[:,None]
        center_to_back_denominator = center_to_back_distance**2
        center_to_back_term = center_to_back_preamble[:,None] * (
                                        center_to_back_numerator_01 + 
                                        center_to_back_numerator_02 ) / center_to_back_denominator[:,None]

        forward_to_center_distance = np.linalg.norm(center_point - forward_point,axis=1)
        forward_to_center_preamble    = 2 * (forward_height - center_height)/forward_to_center_distance - self.goal_slope
        forward_to_center_numerator_01 = -center_slope * forward_to_center_distance[:,None]
        forward_to_center_numerator_02 = 1./2*(forward_height - center_height)[:,None] * (forward_point - center_point)/forward_to_center_distance[:,None]
        forward_to_center_denominator = forward_to_center_distance**2
        forward_to_center_term = forward_to_center_preamble[:,None] * (
                                        forward_to_center_numerator_01 + 
                                        forward_to_center_numerator_02 ) / forward_to_center_denominator[:,None]

        return (forward_to_center_term + center_to_back_term).flatten() * self.magnify

    def minimize(self, method = None):
        x0 = self.x[1:-1,:].flatten()
        fun = self.cost
        if method is None:
            result = minimize(fun,x0,method='Nelder-Mead')
            self.x[1:-1,:] = result.x.reshape(self.m-2,self.n)
            self.solved = result.success
        elif method == 'NAG':
            result = minimize_nag(fun,x0,jac=self.gradient, stepsize = 1e-5, mom_parameter=0.1,maxiter = 1000)
            self.x[1:-1,:] = result['x']
        print(result)

    def plot(self):
        plt.imshow(self.original,cmap='jet')
        plt.scatter(self.x[0,0],self.x[0,1],color='b')
        plt.scatter(self.x[-1,0],self.x[-1,1],color='b')
        plt.scatter(self.x[1:-1,0],self.x[1:-1,1],color='r')
        plt.plot(self.x[:,0], self.x[:,1])
        plt.show()

#}}}    


        

if __name__=='__main__':
    #test_NEB()
    #test_Catenary()
    #test_MinMap()
    test_ConstantSlope()

