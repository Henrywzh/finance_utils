from setuptools import setup

setup(
    name='finance_utils',
    version='0.1.3',

    url='https://github.com/Henrywzh/finance_utils',
    author='Henry Wu',
    author_email='hernywzh88@gmail.com',

    install_requires=[
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'yfinance'
    ],
    packages=['finance_utils'],

    license='IC',
    description='My first python package',
    long_description=open('README.md').read()
)

"""
    # Needed to silence warnings (and to be a worthwhile package)
    name='Measurements',
    url='https://github.com/jladan/package_demo',
    author='John Ladan',
    author_email='jladan@uwaterloo.ca',
    # Needed to actually package something
    packages=['measure'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
"""