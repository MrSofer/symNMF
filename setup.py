from setuptools import setup, Extension

# Define the C extension module
symnmf_module = Extension(
    'symnmfmodule',  # The name of the extension module
    sources=['symnmfmodule.c', 'symnmf.c'],  # Source files for the extension
)

setup(
    name='symnmfmodule',  # The name of your package
    version='0.1.0', # The version of your package
    description='Symmetric Non-negative Matrix Factorization clustering', # A short description
    ext_modules=[symnmf_module], # List of extension modules to build
)
