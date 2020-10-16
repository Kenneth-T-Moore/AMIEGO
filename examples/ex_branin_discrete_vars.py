"""
AMIEGO Example

Problem: Branin
- Continuous Variables: xC
- Integer Variables:    xI -- These are implemented in the model as "Discrete Variables"

Continuous Opt: pyOptSparse
MINLP Opt: MIMOS

Pre-Opt: None
"""
import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.branin import BraninDiscrete

from amiego.amiego_driver import AMIEGO_Driver


prob = om.Problem()
model = prob.model

model.add_subsystem('comp', BraninDiscrete(),
                    promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

model.add_design_var('xI', lower=-5, upper=10)
model.add_design_var('xC', lower=0.0, upper=15.0)
model.add_objective('comp.f')

prob.driver = AMIEGO_Driver()

prob.driver.cont_opt = om.pyOptSparseDriver()

# You can use SNOPT here if you have it.
prob.driver.cont_opt.options['optimizer'] = 'SLSQP'

# We recommened to use MIMOS instead of Branch and Bound.
prob.driver.options['multiple_infill'] = True

prob.driver.sampling = {'xI' : np.array([[-5.0], [0.0], [5.0]])}

prob.setup()

prob.set_val('xC', 7.5)
prob.set_val('xI', 0)

prob.run_driver()

print('Solution')
print('xI:', prob.get_val('xI'))
print('xC:', prob.get_val('xC'))
print('Obj:', prob.get_val('comp.f'))