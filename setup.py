from setuptools import setup, find_packages

setup(name='xgbexcel',
      version='0.0.1',
      description='Convert an XGBRegressor model to an Excel formula expression.',
      author='KalinNonchev',
      author_email='boo@foo.com',
      license='MIT License',
      long_description_content_type='text/markdown',
      long_description=open('README.md').read(),
      url="https://github.com/KalinNonchev/xgbexcel",
      packages=find_packages(),  # find packages
      #      package_data={
      #          "boo": ["pkgdata/*"],   # include pkgdata into package
      #      },
      include_package_data=True,
      # external packages as dependencies,
      # install_requires=[],
      python_requires='>=3.6'
      )
