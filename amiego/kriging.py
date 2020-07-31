"""
Surrogate model based on Kriging.

In AMIEGO, optimization over the integer design variables are done on this surrogate.
"""
import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize
from pyDOE import lhs

from openmdao.utils.concurrent import concurrent_eval_lb, concurrent_eval

from amiego.optimize_function import snopt_opt

MACHINE_EPSILON = np.finfo(np.double).eps


class AMIEGOKrigingSurrogate(object):
    """
    Surrogate Modeling method based on the simple Kriging interpolation.

    Predictions are returned as a tuple of mean and RMSE. Based on Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams. (see also: scikit-learn).

    Attributes
    ----------
    c_r : ndarray
        Reduced likelyhood parameter c_r.
    comm : MPI communicator or None
        The MPI communicator from parent solver's containing group.
    eval_rmse : bool
        When true, calculate the root mean square prediction error.
    n_dims : int
        Number of independents in the surrogate
    n_samples : int
        Number of training points.
    nugget : double or ndarray, optional
        Nugget smoothing parameter for smoothing noisy data. Represents the variance
        of the input values. If nugget is an ndarray, it must be of the same length
        as the number of training points. Default: 10. * Machine Epsilon
    pcom : int
        Internally calculated optimal number of hyperparameters.
    SigmaSqr : ndarray
        Reduced likelyhood parameter: sigma squared
    thetas : ndarray
        Kriging hyperparameters.
    trained : bool
        True when surrogate has been trained.
    use_snopt : bool
        Set to True to use pyOptSparse and SNOPT.
    Wstar : ndarray
        The weights for KPLS.
    X : ndarray
        Training input values, normalized.
    X_mean : ndarray
        Mean of training input values, normalized.
    X_std : ndarray
        Standard deviation of training input values, normalized.
    Y : ndarray
        Training model response values, normalized.
    Y_mean : ndarray
        Mean of training model response values, normalized.
    Y_std : ndarray
        Standard deviation of training model response values, normalized.
    """

    def __init__(self, nugget=10. * MACHINE_EPSILON, eval_rmse=False):
        """
        Initialize the Amiego Kriging surrogate.

        Parameters
        ----------
        nugget : double or ndarray, optional
            Nugget arameter for smoothing noisy data. Represents the variance of the input
            values. If nugget is an ndarray, it must be of the same length as the number of training
            points. Default: 10. * Machine Epsilon
        eval_rmse : bool
            Flag indicating whether the Root Mean Squared Error (RMSE) should be computed.
            Set to False by default.
        """
        self.n_dims = 0       # number of independent
        self.n_samples = 0       # number of training points
        self.thetas = np.zeros(0)
        self.nugget = nugget

        self.c_r = np.zeros(0)
        self.SigmaSqr = np.zeros(0)
        self.trained = False

        # Normalized Training Values
        self.X = np.zeros(0)
        self.Y = np.zeros(0)
        self.X_mean = np.zeros(0)
        self.X_std = np.zeros(0)
        self.Y_mean = np.zeros(0)
        self.Y_std = np.zeros(0)

        self.use_snopt = False
        self.eval_rmse = eval_rmse

        self.Wstar = np.identity(0)
        self.pcom = 0

        # Put the comm here
        self.comm = None

    def train(self, x, y, KPLS=False, norm_data=False):
        """
        Train the surrogate model with the given set of inputs and outputs.

        Parameters
        ----------
        x : array-like
            Training input locations
        y : array-like
            Model responses at given inputs.
        KPLS : Boolean
            False when KPLS is not added to Kriging (default)
            True Adds KPLS method to Kriging to reduce the number of hyper-parameters
        norm_data : bool
            Set to True if the incoming training data has already been normalized.
        """
        self.trained = True

        x, y = np.atleast_2d(x, y)

        self.n_samples, self.n_dims = x.shape

        if self.n_samples <= 1:
            raise ValueError('KrigingSurrogate require at least 2 training points.')

        if not norm_data:
            # Normalize the data
            X_mean = np.mean(x, axis=0)
            X_std = np.std(x, axis=0)
            Y_mean = np.mean(y, axis=0)
            Y_std = np.std(y, axis=0)

            X_std[X_std == 0.] = 1.0
            Y_std[Y_std == 0.] = 1.0

            X = (x - X_mean) / X_std
            Y = (y - Y_mean) / Y_std

            self.X = X
            self.Y = Y
            self.X_mean, self.X_std = X_mean, X_std
            self.Y_mean, self.Y_std = Y_mean, Y_std

        comm = self.comm
        num_pts = max([30, 3*comm.size])
        if KPLS:
            # Maximum number of hyper-parameters we want to afford
            pcom_max = 3

            # TODO Use some criteria to find optimal number of hyper-parameters.
            self.pcom = min([pcom_max, self.n_dims])

            self.Wstar = self.KPLS_reg()
            if self.pcom >= 3:
                start_point = lhs(3, num_pts)
            else:
                start_point = lhs(self.n_dims, 30)
        else:
            self.Wstar = np.identity(self.n_dims)
            self.pcom = self.n_dims
            start_point = lhs(self.n_dims, num_pts)

        # Multi-start approach (starting from 10*pcom_max different locations)
        if comm is not None and comm.size < 2:
            comm = None

        cases = [([pt], None) for pt in start_point]
        results = concurrent_eval_lb(self._calculate_thetas, cases,
                                     comm, broadcast=True)
        # results = concurrent_eval(self._calculate_thetas, cases,
        #                           comm, allgather=True)

        # Print the traceback if it fails
        for result in results:
            if not result[0]:
                print(result[1])

        thetas = [item[0][0] for item in results if item[0] is not None]
        fval = [item[0][1] for item in results if item[0] is not None]

        idx = fval.index(min(fval))
        self.thetas = np.dot((self.Wstar**2), thetas[idx].T).flatten()

        print("BestLogLike: ", fval[idx])

        _, params = self._calculate_reduced_likelihood_params()
        self.c_r = params['c_r']
        self.S_inv = params['S_inv']
        self.Vh = params['Vh']
        self.mu = params['mu']
        self.SigmaSqr = params['SigmaSqr']
        self.R_inv = params['R_inv']

    def _calculate_thetas(self, point):
        """
        Solve optimization problem for hyperparameters.

        This has been parallelized so that the best value can be found from a set of
        optimization starting points.

        Parameters
        ----------
        point : list
            Starting point for opt.

        Returns
        -------
        ndarray
            Optimal Hyperparameters.
        float
            Objective value from optimizing the hyperparameters.
        """
        x0 = -3.0 * np.ones((self.pcom, )) + point * (5.0 * np.ones((self.pcom, )))

        # Use SNOPT (or fallback on other pyoptsparse optimizer.)
        if self.use_snopt:
            def _calcll(dv_dict):
                """
                Evaluate objective for pyoptsparse.
                """
                thetas = dv_dict['x']
                x = np.dot((self.Wstar**2), (10.0**thetas).T)
                loglike = self._calculate_reduced_likelihood_params(x)[0]

                # Objective
                func_dict = {}
                func_dict['obj'] = -loglike

                return func_dict, 0

            low = -3.0 * np.ones([self.pcom, 1])
            high = 2.0 * np.ones([self.pcom, 1])
            opt_x, opt_f, success, msg = snopt_opt(_calcll, x0, low, high, title='kriging',
                                                   options={'Major optimality tolerance': 1.0e-6})

            if not success:
                print("SNOPT failed to converge.", msg)
                opt_f = 1.0

            thetas = np.asarray(10.0**opt_x)
            fval = opt_f

        # Use Scipy COBYLA.
        else:

            def _calcll(thetas):
                """
                Evaluate objective for Scipy Cobyla.
                """
                x = np.dot((self.Wstar**2), (10.0**thetas).T).flatten()
                loglike = self._calculate_reduced_likelihood_params(x)[0]
                return -loglike

            bounds = [(-3.0, 2.0) for _ in range(self.pcom)]
            optResult = minimize(_calcll, x0, method='cobyla',
                                 options={'ftol': 1e-6},
                                 bounds=bounds)

            if not optResult.success:
                print("Cobyla failed to converge", optResult.success)
                optResult.fun = 1.0

            thetas = 10.0**optResult.x.flatten()
            fval = optResult.fun

        return thetas, fval

    def _calculate_reduced_likelihood_params(self, thetas=None):
        """
        Compute quantity with the same maximum location as the log-likelihood for a given theta.

        Parameters
        ----------
        thetas : ndarray, optional
            Given input correlation coefficients. If none given, uses self.thetas from training.

        Returns
        -------
        float
            Calculated reduced likelihood.
        dict
            Dictionary of reduced likelyhood parameters.
        """
        if thetas is None:
            thetas = self.thetas

        X, Y = self.X, self.Y
        params = {}

        # Correlation Matrix
        distances = np.zeros((self.n_samples, self.n_dims, self.n_samples))
        for i in range(self.n_samples):
            distances[i, :, i + 1:] = np.abs(X[i, ...] - X[i + 1:, ...]).T
            distances[i + 1:, :, i] = distances[i, :, i + 1:].T

        R = np.exp(-thetas.dot(np.square(distances)))
        diag = np.arange(self.n_samples)
        R[diag, diag] = 1. + self.nugget

        [U, S, Vh] = linalg.svd(R)

        # Penrose-Moore Pseudo-Inverse:
        # Given A = USV^* and Ax=b, the least-squares solution is
        # x = V S^-1 U^* b.
        # Tikhonov regularization is used to make the solution significantly more robust.
        h = 1e-8 * S[0]
        inv_factors = S / (S ** 2. + h ** 2.)

        # Using the approach suggested on 1. EGO by D.R.Jones et.al and
        # 2. Engineering Deisgn via Surrogate Modeling-A practical guide
        # by Alexander Forrester, Dr. Andras Sobester, Andy Keane
        one = np.ones([self.n_samples, 1])
        R_inv = Vh.T.dot(np.einsum('i,ij->ij', inv_factors, U.T))
        mu = np.dot(one.T, np.dot(R_inv, Y)) / np.dot(one.T, np.dot(R_inv, one))
        c_r = Vh.T.dot(np.einsum('j,kj,kl->jl', inv_factors, U, (Y - mu * one)))
        logdet = -np.sum(np.log(inv_factors))
        SigmaSqr = np.dot((Y - mu * one).T, c_r).sum(axis=0) / self.n_samples
        reduced_likelihood = -(np.log(np.sum(SigmaSqr)) + logdet / self.n_samples)

        params['c_r'] = c_r
        params['S_inv'] = inv_factors
        params['U'] = U
        params['Vh'] = Vh
        params['R_inv'] = R_inv
        params['mu'] = mu
        params['SigmaSqr'] = SigmaSqr  # This is wrt normalized y

        return reduced_likelihood, params

    def predict(self, x):
        """
        Predict value at new point.

        Calculates a predicted value of the response based on the current
        trained model for the supplied list of inputs.

        Parameters
        ----------
        x : array-like
            Point at which the surrogate is evaluated.

        Returns
        -------
        float
            New predicted value
        """
        if not self.trained:
            msg = "{0} has not been trained, so no prediction can be made."\
                .format(type(self).__name__)
            raise RuntimeError(msg)

        X, Y = self.X, self.Y
        thetas = self.thetas
        if isinstance(x, list):
            x = np.array(x)
        x = np.atleast_2d(x)
        n_eval = x.shape[0]

        x_n = (x - self.X_mean) / self.X_std

        r = np.zeros((n_eval, self.n_samples), dtype=x.dtype)
        for r_i, x_i in zip(r, x_n):
            r_i[:] = np.exp(-thetas.dot(np.square((x_i - X).T)))

        if r.shape[1] > 1:  # Ensure r is always a column vector
            r = r.T

        # Predictor
        y_t = self.mu + np.dot(r.T, self.c_r)
        y = self.Y_mean + self.Y_std * y_t

        if self.eval_rmse:
            one = np.ones([self.n_samples, 1])
            R_inv = self.R_inv
            mse = self.SigmaSqr * (1.0 - np.dot(r.T, np.dot(R_inv, r)) +
                                   ((1.0 - np.dot(one.T, np.dot(R_inv, r)))**2 /
                                    np.dot(one.T, np.dot(R_inv, one))))

            # Forcing negative RMSE to zero if negative due to machine precision
            mse[mse < 0.] = 0.
            return y, np.sqrt(mse)

        return y

    def linearize(self, x):
        """
        Calculate the jacobian of the Kriging surface at the requested point.

        Parameters
        ----------
        x : array-like
            Point at which the surrogate Jacobian is evaluated.

        Returns
        -------
        ndarray
            Jacobian of modeled outputs with respect to inputs.
        """
        thetas = self.thetas

        # Normalize Input
        x_n = (x - self.X_mean) / self.X_std

        r = np.exp(-thetas.dot(np.square((x_n - self.X).T)))

        # Z = einsum('i,ij->ij', X, Y) is equivalent to, but much faster and
        # memory efficient than, diag(X).dot(Y) for vector X and 2D array Y.
        # I.e. Z[i,j] = X[i]*Y[i,j]
        gradr = r * -2 * np.einsum('i,ij->ij', thetas, (x_n - self.X).T)
        jac = np.einsum('i,j,ij->ij', self.Y_std, 1.0 / self.X_std, gradr.dot(self.c_r).T)
        return jac

    def KPLS_reg(self):
        """
        Compute the KLPS weights.

        Returns
        -------
        ndarray
            Wstar, the KPLS weights.
        """
        def power_iter(X, y):
            A = np.dot(np.dot(X.T, y), np.dot(y.T, X))
            qk = np.zeros([A.shape[0], 1])
            qk[0] = 1.0
            kk = 0
            delta = 1.0
            qk_prev = qk
            while delta > 1.0e-6:
                kk += 1
                zk = np.dot(A, qk)
                qk = zk / np.linalg.norm(zk)
                delta = np.linalg.norm(qk - qk_prev)
                qk_prev = qk
            return qk

        Xl = self.X
        yl = self.Y
        k = self.n_dims
        W = np.empty((k, self.pcom))
        P = np.empty((k, self.pcom))
        for l in range(self.pcom):
            wl = power_iter(Xl, yl)
            tl = np.dot(Xl, wl)
            tl_hat = tl / (np.dot(tl.T, tl))
            pl = (np.dot(Xl.T, tl_hat)).T
            cl = np.dot(yl.T, tl_hat)
            W[:, l] = wl[:, 0]
            P[:, l] = pl[0, :]
            Xl = Xl - np.dot(tl, pl)
            yl = yl - cl * tl
        # TODO: See if there are better ways to do inverse
        Wstar = np.dot(W, np.linalg.inv(np.dot(P.T, W)))
        return Wstar
