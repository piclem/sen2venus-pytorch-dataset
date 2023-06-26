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

setup(name='sen2venus',
      version=get_version(),
      description="",
      classifiers=[],
      author=u"piclem",
      author_email='clement.peyrard.phd@gmail.com',
      url='https://github.com/piclem/pysen2venus',
      license='',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      zip_safe=False,
      include_package_data=True,
      )
