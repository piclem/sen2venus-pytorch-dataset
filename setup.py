import os
import sys
import subprocess
import logging
from setuptools import setup, find_packages
import re


def get_version():
    VERSIONFILE = os.path.join('sen2venus', '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r'^__version__ = [\"\']*([\d.]+)[\"\']'
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger()

setup(name='sen2venus-torch-dataset',
      version=get_version(),
      description="Unofficial pytorch dataloader for the Sen2Venµs Super-Resolution dataset.",
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      classifiers=[],
      author=u"Clément Peyrard",
      author_email='clement.peyrard.phd@gmail.com',
      url='https://github.com/piclem/sen2venus-torch-dataset',
      license='',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      zip_safe=False,
      include_package_data=True,
      install_requires=[
        'torch', 'torchvision', 'rioxarray', 'py7zr', 'geopandas', 'fire', 'py7zr'
      ]
      )
