from setuptools import setup

setup(name='Distutils',
      version='1.0',
      description='cobras testing',
      url='https://www.python.org/sigs/distutils-sig/',
      py_modules=['before_clustering',
                  'evaluate_clusterings',
                  'experiments',
                  'generate_clusterings',
                  'present_results',
                  'run_locally',
                  'run_through_ssh',
                  'run_with_dask',
                  'single_tests'],
     )