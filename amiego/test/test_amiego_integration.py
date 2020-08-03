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

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from amiego.amiego_driver import AMIEGO_Driver
from amiego.test.greiwank import Greiwank


class TestAMIEGOintegration(unittest.TestCase):

    def amiego_greiwank(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Greiwank(num_cont=3, num_int=3), promotes=['*'])

        prob.driver = AMIEGO_Driver()
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

        prob.setup()

        prob.set_val('xI', np.array([0, 0, 0]))
        prob.set_val('xC', np.array([0.0, 0.0, 0.0]))

        prob.run_driver()

        # Optimal solution
        assert_near_equal(prob['f'], 0.0, 1e-5)
        assert_near_equal(prob['xI'][0], 0.0, 1e-5)
        assert_near_equal(prob['xI'][1], 0.0, 1e-5)
        assert_near_equal(prob['xI'][2], 0.0, 1e-5)


if __name__ == "__main__":
    unittest.main()
