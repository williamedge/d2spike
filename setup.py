from setuptools import setup

setup(
    name='d2spike',
    version='0.0.1',    
    description='A Python package for de-spiking velocity data.',
    url='https://github.com/williamedge/d2spike',
    author='William Edge',
    author_email='william.edge@uwa.edu.au',
    license='BSD 3-clause',
    packages=['d2spike'],
    install_requires=['numpy',
                      'xarray',
                      'matplotlib',
                      'seaborn',
                      'afloat',
                      'wutils'],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)
