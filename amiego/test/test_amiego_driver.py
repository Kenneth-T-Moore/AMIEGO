""" Unit tests for the AMIEGO driver."""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.branin import Branin, BraninDiscrete
from openmdao.test_suite.components.three_bar_truss import ThreeBarTruss, ThreeBarTrussVector
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import set_pyoptsparse_opt

from amiego.amiego_driver import AMIEGO_Driver

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class TestAMIEGOdriver(unittest.TestCase):

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is not 'SNOPT':
            raise unittest.SkipTest("SNOPT is needed to run this test")

    def test_simple_branin_opt(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Branin(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = AMIEGO_Driver()
        prob.driver.options['disp'] = True

        prob.driver.cont_opt = om.pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'
        prob.driver.minlp.options['trace_iter'] = 3
        prob.driver.minlp.options['trace_iter_max'] = 5

        prob.driver.sampling = {'xI' : np.array([[-5.0], [0.0], [5.0]])}

        prob.setup()

        prob.set_val('xC', 7.5)
        prob.set_val('xI', 0.0)

        prob.run_driver()

        # Optimal solution
        assert_near_equal(prob['comp.f'], 0.49398, 1e-5)
        self.assertTrue(int(prob['xI']) in [3, -3])

    def test_simple_branin_opt_manual_ivc(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', om.IndepVarComp('xI', 0.0))
        model.add_subsystem('comp', Branin())

        model.connect('p2.xI', 'comp.x0')
        model.connect('p1.xC', 'comp.x1')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = AMIEGO_Driver()
        prob.driver.options['disp'] = True

        prob.driver.cont_opt = om.pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'
        prob.driver.minlp.options['trace_iter'] = 3
        prob.driver.minlp.options['trace_iter_max'] = 5

        prob.driver.sampling = {'p2.xI' : np.array([[-5.0], [0.0], [5.0]])}

        prob.setup()
        prob.run_driver()

        # Optimal solution
        assert_near_equal(prob['comp.f'], 0.49398, 1e-5)
        self.assertTrue(int(prob['p2.xI']) in [3, -3])

    def test_simple_branin_opt_discrete(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', BraninDiscrete(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5, upper=10)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = AMIEGO_Driver()
        prob.driver.options['disp'] = False

        prob.driver.cont_opt = om.pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'
        prob.driver.minlp.options['trace_iter'] = 3
        prob.driver.minlp.options['trace_iter_max'] = 5

        prob.driver.sampling = {'xI' : np.array([[-5.0], [0.0], [5.0]])}

        prob.setup()

        prob.set_val('xC', 7.5)
        prob.set_val('xI', 0.0)

        prob.run_driver()

        # Optimal solution
        assert_near_equal(prob['comp.f'], 0.49398, 1e-5)
        self.assertTrue(int(prob['xI']) in [3, -3])

    def test_simple_branin_opt_mimos(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', BraninDiscrete(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5, upper=10)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = AMIEGO_Driver()
        prob.driver.options['disp'] = True
        prob.driver.options['multiple_infill'] = True

        prob.driver.cont_opt = om.pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'

        prob.driver.sampling = {'xI' : np.array([[-5.0], [0.0], [5.0]])}

        prob.setup()

        prob.set_val('xC', 7.5)
        prob.set_val('xI', 0.0)

        prob.run_driver()

        # Optimal solution
        assert_near_equal(prob['comp.f'], 0.49398, 1e-5)
        self.assertTrue(int(prob['xI']) in [3, -3])

    def test_three_bar_truss(self):
        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('area1', 5.0, units='cm**2')
        ivc.add_output('area2', 5.0, units='cm**2')
        ivc.add_output('area3', 5.0, units='cm**2')
        ivc.add_output('mat1', 1)
        ivc.add_output('mat2', 1)
        ivc.add_output('mat3', 1)

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem('comp', ThreeBarTruss(), promotes=['*'])

        prob.driver = AMIEGO_Driver()
        prob.driver.cont_opt = om.pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'

        prob.driver.minlp.options['trace_iter'] = 3
        prob.driver.minlp.options['trace_iter_max'] = 5
        prob.driver.minlp._randomstate = 1

        prob.driver.options['max_infill_points'] = 4

        model.add_design_var('area1', lower=0.0005, upper=10.0)
        model.add_design_var('area2', lower=0.0005, upper=10.0)
        model.add_design_var('area3', lower=0.0005, upper=10.0)
        model.add_design_var('mat1', lower=1, upper=4)
        model.add_design_var('mat2', lower=1, upper=4)
        model.add_design_var('mat3', lower=1, upper=4)
        model.add_objective('mass')
        model.add_constraint('stress', upper=1.0)

        npt = 5
        samples = np.array([[4, 2, 3],
                            [1, 3, 1],
                            [3, 1, 2],
                            [3, 4, 2],
                            [1, 1, 4]])

        prob.driver.sampling = {'mat1' : samples[:, 0].reshape((npt, 1)),
                                'mat2' : samples[:, 1].reshape((npt, 1)),
                                'mat3' : samples[:, 2].reshape((npt, 1))}

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob['mass'], 5.287, 1e-3)
        assert_near_equal(prob['mat1'], 3, 1e-5)
        assert_near_equal(prob['mat2'], 3, 1e-5)
        #Material 3 can be anything

    def test_three_bar_truss_preopt(self):
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('area', np.array([5.0, 5.0, 5.0]), units='cm**2')
        model.set_input_defaults('mat', np.array([1, 1, 1]))

        model.add_subsystem('comp', ThreeBarTrussVector(), promotes=['*'])

        prob.driver = AMIEGO_Driver()
        prob.driver.cont_opt = om.pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'
        prob.driver.options['disp'] = True

        prob.driver.minlp.options['trace_iter'] = 3
        prob.driver.minlp.options['trace_iter_max'] = 5
        prob.driver.minlp._randomstate = 1

        prob.driver.options['max_infill_points'] = 4

        model.add_design_var('area', lower=0.0005, upper=10.0)
        model.add_design_var('mat', lower=1, upper=4)
        model.add_objective('mass')
        model.add_constraint('stress', upper=1.0)

        npt = 5
        samples = [np.array([ 4.,  2.,  3.]),
                   np.array([ 1.,  3.,  1.]),
                   np.array([ 3.,  1.,  2.]),
                   np.array([ 3.,  4.,  2.]),
                   np.array([ 1.,  1.,  4.])]

        obj_samples = [np.array([ 20.33476318]),
                       np.array([ 15.70506926]),
                       np.array([ 11.400119]),
                       np.array([ 13.86862845]),
                       np.array([ 7.82279865])]

        con_samples = [np.array([1.21567329, 0.41459045, 0.11071787]),
                       np.array([1.00000066, 0.37435425, 0.35066965]),
                       np.array([ 0.49384804,  1.        ,  0.09501274]),
                       np.array([ 1.00000004,  1.        ,  0.91702808]),
                       np.array([1.29927534, 1.05250939, 0.69434504])]

        prob.driver.sampling = {'mat' : samples}
        prob.driver.obj_sampling = {'mass' : obj_samples}
        prob.driver.con_sampling = {'stress' : con_samples}
        prob.driver.int_con = ['stress']

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob['mass'], 5.287, 1e-3)
        assert_near_equal(prob['mat'][0], 3, 1e-5)
        assert_near_equal(prob['mat'][1], 3, 1e-5)
        # Material 3 can be anything

    def test_three_bar_truss_preopt_mimos(self):
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('area', np.array([5.0, 5.0, 5.0]), units='cm**2')
        model.set_input_defaults('mat', np.array([1, 1, 1]))

        model.add_subsystem('comp', ThreeBarTrussVector(), promotes=['*'])

        prob.driver = AMIEGO_Driver()
        prob.driver.cont_opt = om.pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'
        prob.driver.options['disp'] = True
        prob.driver.options['multiple_infill'] = True

        prob.driver.minlp.options['trace_iter'] = 3
        prob.driver.minlp.options['trace_iter_max'] = 5
        prob.driver.minlp._randomstate = 1

        prob.driver.options['max_infill_points'] = 5

        model.add_design_var('area', lower=0.0005, upper=10.0)
        model.add_design_var('mat', lower=1, upper=4)
        model.add_objective('mass')
        model.add_constraint('stress', upper=1.0)

        npt = 5
        samples = [np.array([ 4.,  2.,  3.]),
                   np.array([ 1.,  3.,  1.]),
                   np.array([ 3.,  1.,  2.]),
                   np.array([ 3.,  4.,  2.]),
                   np.array([ 1.,  1.,  4.])]

        obj_samples = [np.array([ 20.33476318]),
                       np.array([ 15.70506926]),
                       np.array([ 11.400119]),
                       np.array([ 13.86862845]),
                       np.array([ 7.82279865])]

        con_samples = [np.array([1.21567329, 0.41459045, 0.11071787]),
                       np.array([1.00000066, 0.37435425, 0.35066965]),
                       np.array([ 0.49384804,  1.        ,  0.09501274]),
                       np.array([ 1.00000004,  1.        ,  0.91702808]),
                       np.array([1.29927534, 1.05250939, 0.69434504])]

        prob.driver.sampling = {'mat' : samples}
        prob.driver.obj_sampling = {'mass' : obj_samples}
        prob.driver.con_sampling = {'stress' : con_samples}
        prob.driver.int_con = ['stress']

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob['mass'], 5.287, 1e-3)
        assert_near_equal(prob['mat'][0], 3, 1e-5)
        assert_near_equal(prob['mat'][1], 3, 1e-5)


if __name__ == "__main__":
    unittest.main()
