from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

# ---- C/C++ EXTENSIONS ---- #
cython_modules = ['fg2015/feature/cython/gradient.pyx',
                  'fg2015/image/cython/extract_patches.pyx']
cython_exts = cythonize(cython_modules, quiet=True)
include_dirs = [np.get_include()]

requirements = ['menpo>=0.3.0',
                'scikit-image>=0.10.1']

setup(name='fg2015',
      version='0.0',
      description='AAM repository',
      author='Joan Alabort-i-Medina',
      author_email='joan.alabort@gmail.com',
      include_dirs=include_dirs,
      ext_modules=cython_exts,
      packages=find_packages(),
      install_requires=requirements)
