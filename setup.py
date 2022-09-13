from distutils.core import setup
import setuptools

"""
To upload: 
conda activate py38 && python setup.py bdist_wheel && conda deactivate && twine upload dist/*.whl
"""

setup(name='temo',
      version='0.0.1',
      description='teqp-based model optimization',
      author='Ian H. Bell',
      author_email='ian.bell@nist.gov',
      packages = ['temo'],
      zip_safe = False
     )

