from distutils.core import setup

setup(name='amiego',
      version='1.0.0',
      packages=[
          'amiego',
          'amiego/test',
      ],

      install_requires=[
        'openmdao>=3.2.0',
      ]
)
