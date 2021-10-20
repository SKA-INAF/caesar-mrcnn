#! /usr/bin/env python
"""
Setup for mrcnn
"""
import os
import sys
from setuptools import setup, find_packages


def read(fname):
	"""Read a file"""
	return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
	""" Get the package version number """
	import mrcnn
	return mrcnn.__version__



PY_MAJOR_VERSION=sys.version_info.major
PY_MINOR_VERSION=sys.version_info.minor
print("PY VERSION: maj=%s, min=%s" % (PY_MAJOR_VERSION,PY_MINOR_VERSION))

reqs= []
reqs.append('numpy>=1.10')
reqs.append('astropy>=2.0, <3')


if PY_MAJOR_VERSION<=2:
	print("PYTHON 2 detected")
	reqs.append('future')
	reqs.append('scipy<=1.2.1')
	reqs.append('scikit-learn>=0.20')
	reqs.append('pyparsing>=2.0.1')
	reqs.append('matplotlib<=2.2.4')
else:
	print("PYTHON 3 detected")
	
	#reqs.append('scipy<=1.2.1')
	reqs.append('scikit-image<=0.15.0')
	#reqs.append('scikit-learn>=0.20')
	reqs.append('scikit-learn')
	reqs.append('scipy')
	reqs.append('pyparsing')
	reqs.append('matplotlib')

reqs.append('keras>=2.0')
reqs.append('tensorflow-gpu>=1.13')
reqs.append('opencv-python')
reqs.append('h5py<=2.10.0')
reqs.append('imgaug')
reqs.append('Pillow')
reqs.append('cython')
reqs.append('numpyencoder')


data_dir = 'data'

setup(
	name="mrcnn",
	version=get_version(),
	author="Simone Riggi",
	author_email="simone.riggi@gmail.com",
	description="Tool to detect radio sources from astronomical FITS images using Mask R-CNN",
	license = "GPL3",
	url="https://github.com/SKA-INAF/mrcnn",
	long_description=read('README.md'),
	#packages=['mrcnn'],
	packages=find_packages(),
	data_files=[("data",["data/galaxy0002.fits", "data/sidelobe0001.fits"])],
	include_package_data=True,
	install_requires=reqs,
	scripts=['scripts/run.py','scripts/train_all.py','scripts/train_all_gpu.py','scripts/sclassifier.py','scripts/run_mrcnn.sh','scripts/draw_img.py'],
)
