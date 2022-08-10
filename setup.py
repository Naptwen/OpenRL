from setuptools import setup

setup(
  name = 'usgAI',
  version = '2.2.4',
  description='usgAI pip installation',
  url='https://github.com/Naptwen/Open_pyAI.git',
  autho='Useop Gim',
  license='GNU AFFERO (c) 2022 Useop Gim',
  packages=['usgAI'],
  zip_safe=False,
  install_requires=[
    'numpy==1.23.0',
    'colorama==0.4.5',
    'sympy==1.10.1'
  ]
)
