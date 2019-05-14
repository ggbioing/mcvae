from setuptools import setup, find_packages

packages = {"" : "src"}
for package in find_packages("src"):
    packages[package] = "src"

setup(
    packages = packages.keys(),
    package_dir = {"" : "src"},
    name = 'mcvae',
    version = '1.0.0',
    author = 'Luigi Antelmi',
    author_email = 'luigi.antelmi@inria.fr',
    description = 'Multi-Channel Variational Autoencoder',
    long_description = 'TODO',
    license = 'Inria',
    )
