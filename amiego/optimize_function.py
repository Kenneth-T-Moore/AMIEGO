"""
Optimize a python function or method with pyoptsparse.
"""
import numpy as np

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.mpi import FakeComm

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


def snopt_opt(objfun, desvar, lb, ub, ncon=None, title=None, options=None,
              sens='FD', jac=None):
    """
    Find optimal values using SNOPT from pyoptsparse.

    If SNOPT is not available, use other avaiilable optimizer. The desvar is an array of variables
    named 'x', and the objective is named 'obj'. All constraints are in a vector called 'con'.

    Parameters
    ----------
    objfun : function
        Callback function for objective/constraint evaluation.
    desvar : ndarray
        Initial values for design variables.
    lb : ndarray
        Lower bounds for design variables.
    ub : ndarray
        Upper bounds for design variables.
    ncon : float or None, optional
        Number of constraints. Constrants must be linear with pre-computed jacobian.
    title : str
        Title for pyoptsparse output.
    options : dict, optional
        Dictionary of specific options.
    sens : function or 'FD', optional
        Derivative of objective and constraints, set to 'FD' for pyoptsparse finite difference.
    jac : ndarray, optioinal
        Precalculated jacobian of the constraints.

    Returns
    -------
    ndarray
        Solution vector.
    float
        Objective value.
    bool
        Success flag, True if optimization was successful.
    int
        Return (error) code from SNOPT.
    """
    if OPTIMIZER:
        from pyoptsparse import Optimization
    else:
        raise(RuntimeError, 'Need pyoptsparse to run the SNOPT sub optimizer.')

    opt_prob = Optimization(title, objfun, comm=None)

    ndv = len(desvar)

    opt_prob.addVarGroup('x', ndv, type='c', value=desvar.flatten(), lower=lb.flatten(),
                         upper=ub.flatten())
    if ncon is not None:
        opt_prob.addConGroup('con', ncon, upper=np.zeros((ncon)))#, linear=True, wrt='x',
                             #jac={'x': jac})
    opt_prob.addObj('obj')

    # Fall back on SLSQP or COBYLA if SNOPT isn't there
    _tmp = __import__('pyoptsparse', globals(), locals(), [OPTIMIZER], 0)
    opt = getattr(_tmp, OPTIMIZER)()

    if OPTIMIZER == 'SNOPT':

        for name, value in options.items():
            opt.setOption(name, value)

        opt.setOption('Major iterations limit', 100)
        opt.setOption('Verify level', -1)
        opt.setOption('iSumm', 0)
        opt.setOption('iPrint', 0)

        # Use SNOPT fd instead of pyoptsparse
        if sens == 'FD':
            sens = None

    elif OPTIMIZER == 'SLSQP':
        opt.setOption('MAXIT', 100)
    elif OPTIMIZER == 'CONMIN':
        opt.setOption('ITMAX', 100)
    elif OPTIMIZER == 'COBYLA':
        for name, value in options.items():
            opt.setOption(name, value)


    sol = opt(opt_prob, sens=sens, sensStep=1.0e-6)
    # print(sol)

    x = sol.getDVs()['x']
    f = sol.objectives['obj'].value
    try:
        success_flag = sol.optInform['value'] < 2
        msg = sol.optInform['text']

    except KeyError:
        # optimizers other than pySNOPT may not populate this dict
        success_flag = True
        msg = 'No Status Returned'

    return x, f, success_flag, msg
