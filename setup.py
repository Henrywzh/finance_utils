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

setup(name='py_qbee_tst',
    version='0.1',
    description='test package to run on qbee.io',
    author='qbee AS',
    author_email='author@somemail.com',
    license='MIT',
    packages=['py_qbee_tst'],
    scripts=['bin/qbee_tst.py'],
    zip_safe=False)