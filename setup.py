from setuptools import setup, find_packages

setup(
    name='strategies',
    version='1.0',

    url='https://github.com/Henrywzh/finance_utils',
    author='Henry Wu',
    author_email='hernywzh88@gmail.com',

    install_requires=[
        'numpy', 'pandas', 'matplotlib', 'seaborn'
    ],
    packages=find_packages()
)