from setuptools import setup, find_packages

setup(
    name='EMG Group 2',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        # Add any other required packages here
    ],
    entry_points={
        'console_scripts': [
            'my_project=my_project.main:main',
        ],
    },
    description='Using EMG and machine learning project to classify finger activity',
    author='',
    author_email='',
    url='',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)
