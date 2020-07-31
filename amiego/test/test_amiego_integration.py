"""
Integration tests for the AMIEGO driver.

Run a single test with

    testflo -vs -m amiego_greiwank

Run all tests with:

    testflo -vs -m amiego*

The -vs flag shows the output as you go. You can omit it if you just want to see the test pass results.

"""
import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp
from openmdao.drivers.amiego_driver import AMIEGO_driver
from openmdao.test_suite.components.greiwank import Greiwank
from openmdao.utils.assert_utils import assert_rel_error


class TestAMIEGOintegration(unittest.TestCase):

    def amiego_greiwank(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('xC', np.array([0.0, 0.0, 0.0])), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('xI', np.array([0, 0, 0])), promotes=['*'])
        model.add_subsystem('comp', Greiwank(num_cont=3, num_int=3), promotes=['*'])

        prob.driver = AMIEGO_driver()
        prob.driver.cont_opt.options['tol'] = 1e-12
        prob.driver.options['disp'] = True

        model.add_design_var('xI', lower=-5, upper=5)
        model.add_design_var('xC', lower=-5.0, upper=5.0)

        model.add_objective('f')
        samples = np.array([[1.0, 0.25, 0.75],
                            [0.0, 0.75, 0.0],
                            [0.75, 0.0, 0.25],
                            [0.75, 1.0, 0.5],
                            [0.25, 0.5, 1.0]])
        # prob.driver.sampling = {'xI' : np.array([[0.0], [.76], [1.0]])}
        prob.driver.sampling = {'xI' : samples}

        prob.setup(check=False)
        prob.run_driver()

        # Optimal solution
        assert_rel_error(self, prob['f'], 0.0, 1e-5)
        assert_rel_error(self, prob['xI'][0], 0.0, 1e-5)
        assert_rel_error(self, prob['xI'][1], 0.0, 1e-5)
        assert_rel_error(self, prob['xI'][2], 0.0, 1e-5)