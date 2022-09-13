from setuptools import setup, find_packages

"""
To upload: 
conda activate py38 && python setup.py bdist_wheel && conda deactivate && twine upload dist/*.whl
"""

setup(name='temo',
      version='0.0.2',
      description='teqp-based model optimization',
      author='Ian H. Bell',
      author_email='ian.bell@nist.gov',
      packages = find_packages(include=['temo*']),
      zip_safe = False
     )