from distutils.core import setup

setup(name='amiego',
      version='1.0.0',
      packages=[
          'amiego',
          'amiego/test',
      ],

      install_requires=[
        'openmdao>=3.2.0',
      ],

      entry_points={
          "openmdao_driver": [
              "amiegodriver = amiego.amiego_driver:AMIEGO_Driver",
          ],
      },
)
