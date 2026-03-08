from setuptools import setup, find_packages

setup(name='xgbexcel',
      version='0.1.0',
      description='Convert XGBRegressor or XGBClassifier models to Excel formula expressions for spreadsheet deployment',
      author='Kalin Nonchev',
      license='MIT License',
      long_description_content_type='text/markdown',
      long_description=open('README.md').read(),
      url="https://github.com/KalinNonchev/xgbexcel",
      packages=find_packages(),
      include_package_data=True,
      install_requires=['xgboost'],
      python_requires='>=3.8',
      keywords=[
          'xgboost', 'excel', 'model-deployment', 'machine-learning',
          'model-export', 'interpretability', 'gradient-boosting',
      ],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      )
