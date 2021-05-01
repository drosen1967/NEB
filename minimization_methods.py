import numpy as np

def minimize_nag(fun, x0, args=(), jac = None, stepsize = 1e-5, 
                 mom_parameter=0.1, callback=None, ftol=1e-5, norm=np.Inf, 
                 maxiter=None, disp=False, return_all=False):
    
    """
    Minimization of scalar function of one or more variables using Nesterov's accelerated gradient algorithm.
    
    Parameters
    -----------
    fun : callable
        The objective function to be minimized.
        ``fun(x, *args) -> float``
        where `x` is an 1-D array with shape (n,) and `args`
        is a tuple of the fixed parameters needed to completely
        specify the function.
        
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,), 
        where 'n' is the number of independent variables.
        
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (fun and jac functions).
    
    jac : callable
        The function that returns the gradient vector:
            ``jac(x, *args) -> array_like, shape (n,)``
        where ``x`` is an array with shape (n,) and ``args`` is a tuple with
        the fixed parameters.
    
    stepsize: {float, 'linesearch'}
        If it is a float number, the number specifies the constant stepsize.
        If 'linesearch', the stepsize is determined by a the line search method.
    
    callback : callable, optional
        Called after each iteration:
        ``callable(xk)``
        where ``xk`` is the current iterate.
             
    Options
    -------
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
        
    norm : float
        Order of norm (Inf is max, -Inf is min).
    
    maxiter : int
        Maximum number of iterations to perform.
        
    disp : bool
        Set to True to print convergence messages.
    
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
        
    Returns
    -------
    res : dict
        A dictionary whose keys are: ``x`` the solution array, `, ``fun`` the function value at 
        the solution, ``gfk`` the gradient at the solution, and ``nik`` the total number of iterations.
        If ``return_all`` is true, it also has the field ``allvecs`` that contains all ``xk`` along the 
        iteration trajectory."""
        
    x0 = np.asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0)*200
    
    alpha = stepsize
    beta = mom_parameter
    
    # Modify the following codes to initialize the algorithm
    k = 1
    xk = x0
    xk_1 = x0
    yk = x0
    gfk = jac(xk.reshape((xk.shape[0]//2, 2)), *args)
    gnorm = np.dot(gfk, gfk)**0.5
    f_dif = 100
    # callback(xk.reshape((xk.shape[0]//2, 2)), part=True)

    # -------------------------
    
    if return_all:
        allvecs = [xk]
    
    while (f_dif > ftol) and (k < maxiter):
        
        # Modify the following codes to update xk_1, xk, yk, gfk, gnorm
        xre = xk.reshape((xk.shape[0]//2, 2))
        xl = xre[:-1,:]
        xr = xre[1:,:]
        dists = np.linalg.norm(xl - xr, axis=1)
        alpha = 0.75 * np.min(np.abs(xl - xr))

        xk_1 = xk
        xk = yk - alpha * jac(yk.reshape((xk.shape[0]//2, 2)), *args)
        fk = fun(xk.reshape((xk.shape[0]//2, 2)), *args)
        fk_1 = fun(xk_1.reshape((xk.shape[0]//2, 2)), *args),
        f_dif = abs(fk-fk_1)
        yk = xk + beta * (xk - xk_1)
        gfk = jac(xk.reshape((xk.shape[0]//2, 2)), *args)
        gnorm = np.dot(gfk, gfk)**0.5
    
        print(f_dif)
        
        
        # ----------------------
        if return_all:
            allvecs.append(xk)
        if callback is not None:
            callback(xk)
            
        
        step = 10**(np.ceil(np.max((np.log10(maxiter)-2.0, 0))))  # show at most 100 lines of information  
        if disp and (k % step == 0):
            print('iter = {:4d}, feval = {:6.3f}, gnorm = {:8.6f}, alpha = {:8.6f}'.format(k, fun(xk, *args), gnorm, alpha))
            
        k += 1
    
    result = {'fun':fk, 'jac':gfk, 'x':xk.reshape((xk.shape[0]//2, 2)), 'nit':k}
    print(maxiter)
    
    if return_all:
        result['allvecs'] = allvecs
    
    return result

    