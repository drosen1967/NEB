import numpy as np

#{{{ NAG
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
    
        # print(f_dif)
        
        
        # ----------------------
        if return_all:
            allvecs.append(xk)
        if callback is not None:
            callback(xk.reshape((xk.shape[0]//2, 2)))
            
        
        step = 10**(np.ceil(np.max((np.log10(maxiter)-2.0, 0))))  # show at most 100 lines of information  
        if disp and (k % step == 0):
            print('iter = {:4d}, feval = {:6.3f}, gnorm = {:8.6f}, alpha = {:8.6f}'.format(k, fun(xk, *args), gnorm, alpha))
            
        k += 1
    
    result = {'fun':fk, 'jac':gfk, 'x':xk.reshape((xk.shape[0]//2, 2)), 'nit':k}
    print(maxiter)
    
    if return_all:
        result['allvecs'] = allvecs
    
    return result
#}}}
#{{{ BF
def minimize_lbfgs(fun, x0, args=(), jac = None, stepsize = None, 
                 alpha = .1, m = 10,
                 callback=None, ftol=1e-5, norm=np.Inf, 
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
    
    # Modify the following codes to initialize the algorithm
    k = 1
    x = x0
    q = jac(x)
    sk = np.zeros([len(x),m])
    yk = np.zeros([len(x),m])
    pk = np.zeros([m])
    old_grad = np.copy(q) * .9
    old_x = np.zeros_like(x)
    f_dif = 100
    gr = (np.sqrt(5) + 1)/2
    # callback(xk.reshape((xk.shape[0]//2, 2)), part=True)

    # -------------------------
    if return_all:
        allvecs = [xk]
    
    while (f_dif > ftol) and (k < maxiter):
        
        # Modify the following codes to update xk_1, xk, yk, gfk, gnorm

        q = jac(x)
        grad = jac(x)
        f_dif = np.linalg.norm(old_grad - grad)
        yk = np.hstack([yk[:,1:], (grad - old_grad).reshape(-1,1)])
        sk = np.hstack([sk[:,1:], (x - old_x).reshape(-1,1)])
        pk = np.nan_to_num(1./(np.sum(yk * sk,axis=0) + 1e-3) * (np.linspace(m-1,0,m) <= k))
        a_i = pk * (sk.T @ q)
        q -= np.sum(a_i[None,:] *  yk,axis=1)
        gamma = sk[:,-1]@yk[:,-1] / (yk[:,-1] @ yk[:,-1] + 1e-5)
        HOk = np.eye(len(x)) * gamma
        r = HOk @ q
        b_i = (yk.T @ r) * pk
        r += np.sum(sk * (a_i - b_i)[None,:],axis=1)
        old_grad = np.copy(grad)
        old_x = np.copy(x)
        # Golden Section Search
        if stepsize is None:
            starting_step_size = 1e-1
            a = x; b = x - starting_step_size * r
            c = b - (b-a)/gr; d = a + (b-a)/gr
            for i in range(40): # step precision now on the order of 4 * (2**-10)
                if fun(c) < fun(d): b = d
                else:           a = c
                c = b - (b-a)/gr; d = a + (b-a)/gr
            x = (b + a)/2 # Optimal line search step
        else: x -= stepsize * r

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
    
    result = {'fun':fun(x), 'jac':jac(x), 'x':x.reshape((x.shape[0]//2, 2)), 'nit':k}
    print(maxiter)
    
    if return_all:
        result['allvecs'] = allvecs
    
    return result
#}}}
