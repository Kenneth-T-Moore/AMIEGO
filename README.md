# AMIEGO

OpenMDAO Driver plugin for AMIEGO (A Mixed Integer Efficient Global Optimization)

AMIEGO is an experimental OpenMDAO driver for solving mixed-integer optimization problems. The
approach combines gradient-based optimization, to optimize the continuous design space, together
with an efficient global optimization (EGO), for exploration in the integer design space. More
details on the theory and the algorithm implementation can be found in the references below.

## Installation

After cloning the repository, enter the AMIEGO directory while in an activated OpenMDAO environment and type

    pip install -e .


## Documentation

See the examples directory for several examples of how to use this plugin.


## References

* [Monolithic Approach for Next-Generation Aircraft Design Considering Airline Operations and Economics](http://openmdao.org/pubs/roy_amiego_2019.pdf)

* [A Mixed Integer Efficient Global Optimization Algorithm with Multiple Infill Strategy - Applied to a Wing Topology Optimization Problem](http://openmdao.org/pubs/roy_scitech_2019_submitted.pdf)

* [Next Generation Aircraft Design Considering Airline Operations and Economics](http://openmdao.org/pubs/roy_amiego_2018.pdf)
