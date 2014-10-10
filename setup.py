from setuptools import setup, find_packages

project_name = 'ijcv_2004_aam'

requirements = ['menpo>=0.3.0',
                'scikit-image>=0.10.1']

setup(name=project_name,
      version='0.0',
      description='AAM repository',
      author='Joan Alabort-i-Medina',
      author_email='joan.alabort@gmail.com',
      packages=find_packages(),
      install_requires=requirements)
