"""
Class definition for the MIMOS psuedo-driver.

Implements Multiple Infills via a Multi-Objective Strategy (MIMOS) as a plugin for
the minlp slot on the AMIEGO driver.

Developed by Satadru Roy
School of Aeronautics & Astronautics
Purdue University, West Lafayette, IN 47906
2018~9
Implemented in OpenMDAO, May 2019, Kenneth T. Moore
"""
from math import factorial
import os

import numpy as np
from scipy.cluster.vq import kmeans2, whiten
from scipy.stats import norm

from openmdao.core.driver import Driver
from openmdao.drivers.genetic_algorithm_driver import GeneticAlgorithm


class MIMOS(Driver):
    """
    Class definition for the MIMOS psuedo-driver.

    Implements Multiple Infills via a Multi-Objective Strategy (MIMOS) as a plugin for
    the minlp slot on the AMIEGO driver.

    This will not work as a standalone driver.

    Attributes
    ----------
    _concurrent_pop_size : int
        Number of points to run concurrently when model is a parallel one.
    _concurrent_color : int
        Color of current rank when running a parallel model.
    dvs : list
        Cache of integer design variable names.
    eflag_MINLPBB : bool
        This is set to True when we find a local minimum.
    fopt : ndarray
        Objective value with the maximum expected improvement.
    xI_lb : ndarray
        Lower bound of the integer design variables.
    xI_ub : ndarray
        Upper bound of the integer design variables.
    xopt : ndarray
        List of new infill points from optimizing exploitation and exploration.
    _randomstate : np.random.RandomState, int
         Random state (or seed-number) which controls the seed and random draws.
    """

    def __init__(self):
        """
        Initialize the MIMOS driver.
        """
        super(MIMOS, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['integer_design_vars'] = True
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['active_set'] = False
        self.supports['linear_constraints'] = False
        self.supports['gradients'] = False

        # Options
        self.options.declare('required_samples', 3,
                             desc='Number of infill points.')
        self.options.declare('bits', default={}, types=(dict),
                             desc='Number of bits of resolution. Default is an empty dict, where '
                             'every unspecified variable is assumed to be integer, and the number '
                             'of bits is calculated automatically. If you have a continuous var, '
                             'you should set a bits value as a key in this dictionary.')
        self.options.declare('disp', True,
                             desc='Set to False to prevent printing of iteration '
                             'messages.')
        self.options.declare('max_gen', default=100,
                             desc='Number of generations before termination.')
        self.options.declare('pop_size', default=0,
                             desc='Number of points in the GA. Set to 0 and it will be computed '
                             'as four times the number of bits.')
        self.options.declare('run_parallel', types=bool, default=False,
                             desc='Set to True to execute the points in a generation in parallel.')
        self.options.declare('Pc', default=0.5, lower=0., upper=1.,
                             desc='Crossover rate.')
        self.options.declare('Pm',
                             desc='Mutation rate.', default=None, lower=0., upper=1.,
                             allow_none=True)

        self.dvs = []
        self.i_idx_cache = {}
        self._ga = None

        # We will set this to True if we have found a minimum.
        self.eflag_MINLPBB = False

        # random state can be set for predictability during testing
        if 'SimpleGADriver_seed' in os.environ:
            self._randomstate = int(os.environ['SimpleGADriver_seed'])
        else:
            self._randomstate = None

        # Support for Parallel models.
        self._concurrent_pop_size = 0
        self._concurrent_color = 0

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super(MIMOS, self)._setup_driver(problem)

        model_mpi = None
        comm = self._problem().comm
        if self._concurrent_pop_size > 0:
            model_mpi = (self._concurrent_pop_size, self._concurrent_color)
        elif not self.options['run_parallel']:
            comm = None

        self._ga = GeneticAlgorithm(self.objective_callback, comm=comm, model_mpi=model_mpi)

    def run(self):
        """
        Execute the MIMOS method.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False if successful.
        """
        self.eflag_MINLPBB = False

        x_nd, y_nd = self.find_nondominated_set()

        if len(x_nd) > 0:
            ei_min = min(y_nd[:, 0])
            num_nd = y_nd.shape[0]

            actual_pt2sam = min(self.options['required_samples'], num_nd)

            # Create and select samples from clusters.
            x_new, eflag = self.create_cluster(x_nd, y_nd, actual_pt2sam)

        else:
            x_new = []
            ei_min = None
            eflag = True

        # Save the new infill points for AMIEGO to retrieve.
        self.xopt = x_new
        self.fopt = ei_min
        self.eflag_MINLPBB = eflag

        return False

    def find_nondominated_set(self):
        """
        Compute a non-dominated set of candidate points.

        This is computed via a genetic algorithm that returns a set of points that are pareto
        optimal for maximum expected improvement (exploration) and maximum distance from existing
        points.

        Returns
        -------
        ndarray
            Non-dominated design points.
        ndarray
            Objective values at non dominated points.
        """
        model = self._problem().model
        ga = self._ga
        ga.nobj = 3

        pop_size = self.options['pop_size']
        max_gen = self.options['max_gen']
        user_bits = self.options['bits']
        Pm = self.options['Pm']  # if None, it will be calculated in execute_ga()
        Pc = self.options['Pc']

        count = self.xI_lb.shape[0]
        lower_bound = self.xI_lb.copy()
        upper_bound = self.xI_ub.copy()
        outer_bound = np.full((count, ), np.inf)

        bits = np.empty((count, ), dtype=np.int)

        # Figure out initial design vars.
        desvars = self._designvars
        desvar_vals = self.get_design_var_values()
        x0 = np.empty(count)
        for name, meta in desvars.items():
            i, j = self.i_idx_cache[name]
            x0[i:j] = desvar_vals[name]

        # Bits of resolution
        abs2prom = model._var_abs2prom['output']

        for name, meta in desvars.items():
            i, j = self.i_idx_cache[name]

            if name in self._designvars_discrete:
                prom_name = name
            else:
                prom_name = abs2prom[name]

            if name in user_bits:
                val = user_bits[name]

            elif prom_name in user_bits:
                val = user_bits[prom_name]

            else:
                # If the user does not declare a bits for this variable, we assume they want it to
                # be encoded as an integer. Encoding requires a power of 2 in the range, so we need
                # to pad additional values above the upper range, and adjust accordingly. Design
                # points with values above the upper bound will be discarded by the GA.
                log_range = np.log2(upper_bound[i:j] - lower_bound[i:j] + 1)
                val = log_range  # default case -- no padding required
                mask = log_range % 2 > 0  # mask for vars requiring padding
                val[mask] = np.ceil(log_range[mask])
                outer_bound[i:j][mask] = upper_bound[i:j][mask]
                upper_bound[i:j][mask] = 2**np.ceil(log_range[mask]) - 1 + lower_bound[i:j][mask]

            bits[i:j] = val

        # Automatic population size.
        if pop_size == 0:
            pop_size = 4 * np.sum(bits)

        desvar_new, opt, nfit = ga.execute_ga(x0, lower_bound, upper_bound, outer_bound,
                                              bits, pop_size, max_gen,
                                              self._randomstate, Pm, Pc)

        # Remove any duplicates with the surrogate set.
        idx = []
        for j, var in enumerate(desvar_new):
            if np.all((var == self.obj_surrogate.x_org).all(1) == False):
                idx.append(j)

        return desvar_new[idx], opt[idx]

    def objective_callback(self, x, icase):
        """
        Evaluate problem objectives at the requested point.

        Parameters
        ----------
        x : ndarray
            Value of design variables.
        icase : int
            Case number, used for identification when run in parallel.

        Returns
        -------
        ndarray
            Objective values
        bool
            Success flag, True if successful
        int
            Case number, used for identification when run in parallel.
        """
        obj_surrogate = self.obj_surrogate

        # Objective 1: Expected Improvement
        # (Normalized as per the convention in openmdao_Alpha:Kriging.)
        xval = (x - obj_surrogate.X_mean) / obj_surrogate.X_std

        negEI1 = calc_genEI_norm(xval, obj_surrogate, 1)
        negEI0 = calc_genEI_norm(xval, obj_surrogate, 0)

        # Objective 2: Maximize distance from any existing points.
        x_data = obj_surrogate.x_org
        distance = np.min(np.sqrt(np.sum((x_data - x)**2, axis=1)))

        return np.array([negEI1, -distance, negEI0]), True, icase

    def create_cluster(self, x_nd, y_nd, actual_pt2sam):
        """
        Create cluster of infill points.

        Attributes
        ----------
        x_nd : ndarray
            Non dominated design points.
        y_nd : ndarray
            Objective values at non dominated points.
        actual_pt2sam : float
            Number of new samples.

        Returns
        -------
        ndarray
            New sample points for amiego.
        eflag : bool
            This is set to True unless no new samples are found.
        """
        obj_surrogate = self.obj_surrogate
        exist_pt_x = obj_surrogate.x_org
        exist_pt_y = obj_surrogate.y_org

        num_nd, num_desvar = x_nd.shape
        num_obj = y_nd.shape[1]

        if num_nd <= 1:
            distance = np.sqrt(np.sum((exist_pt_x - x_nd)**2, axis=1))

            if x_nd in exist_pt_x:
                print('No new sample found!')
                x_new = []
                eflag = False
            else:
                eflag = True
                x_new = x_nd

        elif num_nd == actual_pt2sam:
            x_new = x_nd
            eflag = True

        else:
            eflag = True
            ideal_pt = np.min(y_nd, axis=0)
            worst_pt = np.max(y_nd, axis=0)

            norm_nd = (y_nd - ideal_pt) / (worst_pt - ideal_pt)

            cluster_centroids, labels = kmeans2(norm_nd, actual_pt2sam, minit='++')
            clus_sorted_idx = np.argsort(cluster_centroids, axis=0)

            cluster_stack = list(clus_sorted_idx.flatten())

            # Centroid ids are sorted by objective. If we flatten this, it is basically a stack
            # with the entries in the correct order. When we use an entry, we purge all duplicates
            # from the rankings in the other objectives and take the first one again.
            # This will always give us at one from each cluster.
            x_new = np.zeros((actual_pt2sam, num_desvar))
            bad_idx = []
            for ii in range(actual_pt2sam):
                picked = cluster_stack[0]
                centroid_idx = np.where(labels == picked)[0]
                x_clus = x_nd[centroid_idx, :]
                y_clus = y_nd[centroid_idx, :]

                obj_picked = ii % num_obj + 1
                x_new[ii, :], fail = self.pick_from_cluster(x_clus, y_clus, obj_picked, exist_pt_x,
                                                            exist_pt_y)

                cluster_stack = [val for val in cluster_stack if val != picked]

                if fail:
                    bad_idx.append(ii)

            x_new[bad_idx, :] = 0.0

        return x_new, eflag

    def pick_from_cluster(self, x_clus, y_clus, obj_picked, exist_pt_x, exist_pt_y):
        """
        Pick a point from a given cluster.

        Attributes
        ----------
        x_clus : ndarray
            Cluster of non-dominated design points.
        y_clus : ndarray
            Objective values at cluster of non dominated points.
        obj_picked : int
            Id number of objective from 0, 1, 2.
        exist_pt_x : ndarray
            Design points in our set of evaluated desigs.
        exist_pt_y : ndarray
            Objectives for our evaluated designs.

        Returns
        -------
        ndarray
            New sample points for amiego.
        fail : bool
            This is set to True unless no new samples are found.
        """
        fail = False

        if x_clus.shape[0] == 1:
            x_new = x_clus[0]
            return x_new, fail

        distance = np.min(np.sqrt(np.sum((exist_pt_x[:, np.newaxis, :] - x_clus[np.newaxis, :, :])**2,
                                         axis=2)), axis=0)
        pos_dist_idx = np.where(distance > 0.)[0]

        if obj_picked == 1:
            # Expected Improvement (Strategy: Balance)
            # Picks a point that satisfies: [min(EI) and dis>0]
            idx = np.argmin(y_clus[pos_dist_idx, 0], axis=0)
            x_new = x_clus[pos_dist_idx[idx], :]

        elif obj_picked == 2:
            # Reduce void spaces (Strategy: pure exploration)
            # Picks a point that satisfies: max(dis)
            pos_dist = distance[pos_dist_idx]
            idx = np.argmax(pos_dist)
            x_new = x_clus[pos_dist_idx[idx], :]

        else:
            # Search around the best solution (Strategy: pure exploitation)
            # Picks a point that satisfies: closest to the best existing point
            idx = np.argmin(exist_pt_y)
            distance2 = np.sqrt(np.sum((exist_pt_x[idx, :] - x_clus)**2, axis=1))
            pos_dist2_idx = np.where(distance2 > 0)[0]
            pos_dist2 = distance[pos_dist2_idx]
            idx = np.argmin(pos_dist2)
            x_new = x_clus[pos_dist2_idx[idx], :]

        if len(x_new) < 1:
            fail = True
            x_new = 0.0 * x_clus

        return x_new, fail


def calc_genEI_norm(xval, obj_surrogate, gg):
    """
    Compute the generalized expected improvement.

    Parameters
    ----------
    xval : ndarray
        Value of the current integer design variables.
    obj_surrogate : <AMIEGOKrigingSurrogate>
        Surrogate model of optimized objective with respect to integer design variables.
    gg : float
        Parameter used in generalized expected improvement to shift between local or global search.
        Higher values emphasize global search.

    Returns
    -------
    float
        The generalized expected improvement evaluated at xval.
    """
    y_min = obj_surrogate.best_obj_norm

    X = obj_surrogate.X
    Y = obj_surrogate.Y
    c_r = obj_surrogate.c_r
    thetas = obj_surrogate.thetas
    SigmaSqr = obj_surrogate.SigmaSqr
    R_inv = obj_surrogate.R_inv
    mu = obj_surrogate.mu
    p = 2

    r = np.exp(-np.einsum("ij->i", thetas.T * (xval - X)**p))

    # Calculate prediction and error.
    y_hat = mu + np.dot(r, c_r)
    term0 = np.dot(R_inv, r)

    SSqr = SigmaSqr * (1.0 - r.dot(term0) +
                       (1.0 - np.einsum('i->', term0))**2 / np.einsum('ij->', R_inv))

    if SSqr <= 1.0e-30:
        neg_genEI = -0.0

    else:
        # Calculate the generalized expected improvement function
        sqrt_SSqr = np.sqrt(SSqr)
        z = (y_min - y_hat) / sqrt_SSqr
        z = np.asscalar(z)
        sg = sqrt_SSqr ** gg

        phi_s = norm.pdf(z)
        phi_C = norm.cdf(z)

        #%     phi_C_check = (0.5 + 0.5*erf((1/sqrt(2))*((y_min - y_hat)/sqrt(abs(SSqr)))));
        #%     phi_s_check = (1/sqrt(2*pi))*exp(-(1/2)*((y_min - y_hat)^2/abs(SSqr)));
        #%     if abs(phi_C - phi_C_check)>1e-10 || abs(phi_s - phi_s_check)>1e-10
        #%         disp('Values do not match!')
        #%         keyboard
        #%     end

        T_k = np.array([phi_C, -phi_s])
        SS = 0;
        for kk in range(gg + 1):
            if kk >= 1:
                # NOTE: This branch is never reached with our current choice of gg, but it has
                # been kept for future investigation.
                T_k[kk] = -phi_s * z**(kk - 1) + (kk - 1)*T_k[kk - 2]

            SS += ((-1)**kk) * (factorial(gg) / (factorial(kk) * factorial(gg - kk))) * \
                  (z**(gg - kk)) * T_k[kk]

        neg_genEI = -sg[0] * SS

    return neg_genEI
