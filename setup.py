from setuptools import setup

from utils.strategies import __version__

setup(
    name='strategies',
    version=__version__,

    url='https://github.com/Henrywzh/finance_utils',
    author='Henry Wu',
    author_email='hernywzh88@gmail.com',

    py_modules=['strategies'],
)