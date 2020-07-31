"""
Greiwank function with N continuous desgin variables and M integer desgin
variables. The Griewank function is a function widely used to test the
convergence of optimization functions.
"""
import numpy as np

import openmdao.api as om


class Greiwank(om.ExplicitComponent):
    """
    Greiwank function with N continuous desgin variables and M integer desgin variables.

    Parameters
    ----------
    num_int : int(2)
        Number of integer design variables.
    num_cont : int(2)
        Number of continuous design variables.
    """

    def initialize(self):
        self.options.declare('num_int', 2, types=int)
        self.options.declare('num_cont', 2, types=int)

    def setup(self):
        num_int = self.options['num_int']
        num_cont = self.options['num_cont']

        # Inputs
        self.add_input('xI', np.zeros((num_int, )))
        self.add_input('xC', np.zeros((num_cont, )))

        # Outputs
        self.add_output('f', val=0.0)

        self.declare_partials(of='f', wrt=['xC'])

    def compute(self, inputs, outputs):
        """
        Define the function f(xI, xC)
        Here xI is integer and xC is continuous.
        """
        xI = inputs['xI']
        xC = inputs['xC']

        f1I = np.sum((xI**2 / 4000.0))
        f1C = np.sum((xC**2 / 4000.0))


        f2C = 1.0
        for ii in range(len(xC)):
            f2C *= np.cos(xC[ii] / np.sqrt(ii + 1.))

        f2I = 1.0
        for ii in range(len(xI)):
            f2I *= np.cos(xI[ii] / np.sqrt(ii + 1.))

        outputs['f'] = f1C + f1I - (f2C * f2I) + 1.0

    def compute_partials(self, inputs, partials):
        """
        Provide the Jacobian of just the continuous vars.
        """
        xI = inputs['xI']
        xC = inputs['xC']
        nC = len(xC)

        df1C = xC / 2000.0

        df2C = np.empty((nC, ))
        for ii in range(nC):
            fact = 1.0/np.sqrt(ii + 1.)
            df2C[ii] = -fact * np.sin(xC[ii] * fact)

        f2I = 1.0
        for ii in range(len(xI)):
            f2I *= np.cos(xI[ii] / np.sqrt(ii + 1.))

        partials['f', 'xC'] = df1C + df2C * f2I