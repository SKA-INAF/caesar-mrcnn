############################################################
#        MODULE IMPORT
############################################################

import os
import sys
import json
import time
import argparse
import datetime
import random
import numpy as np


from mrcnn import logger
from mrcnn import utils

import matplotlib.pyplot as plt

############################################################
#        PARSE/VALIDATE ARGS
############################################################

def parse_args():
	""" Parse command line arguments """  
  
	# - Parse command line arguments
	parser = argparse.ArgumentParser(description='Draw input image')

	parser.add_argument('--grayimg', dest='grayimg', action='store_true')	
	parser.set_defaults(grayimg=False)
	parser.add_argument('--no_uint8', dest='to_uint8', action='store_false')	
	parser.set_defaults(to_uint8=True)
	parser.add_argument('--no_zscale', dest='zscale', action='store_false')	
	parser.set_defaults(zscale=True)
	parser.add_argument('--zscale_contrasts', dest='zscale_contrasts', required=False, type=str, default='0.25,0.25,0.25',help='zscale contrasts applied to all channels') 
		
	parser.add_argument('--biascontrast', dest='biascontrast', action='store_true')	
	parser.set_defaults(biascontrast=False)
	parser.add_argument('--bias', dest='bias', required=False, type=float, default=0.5,help='Bias value (default=0.5)') 
	parser.add_argument('--contrast', dest='contrast', required=False, type=float, default=1.0,help='Contrast value (default=1)') 

	parser.add_argument('--image',required=False,metavar="Input image",type=str,help='Input image in FITS format to apply the model (used in detect task)')

	args = parser.parse_args()

	return args


############################################################
#       MAIN
############################################################
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Parsing script args ...")
	try:
		args= parse_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	if args.grayimg:
		convert_to_rgb= False
	else:
		convert_to_rgb= True	

	zscale_contrasts= [float(x) for x in args.zscale_contrasts.split(',')]

	#===========================
	#==   READ IMAGE
	#===========================
	data, header= utils.read_fits(
		args.image,
		stretch=args.zscale,
		zscale_contrasts=zscale_contrasts,
		normalize=True,
		convertToRGB=convert_to_rgb,
		to_uint8=args.to_uint8,
		stretch_biascontrast=args.biascontrast,
		bias=args.bias,
		contrast=args.contrast
	)

	#===========================
	#==   DRAW
	#===========================
	print("image shape")
	print(data.shape)

	if convert_to_rgb:
		fig, (ax1, ax2, ax3) = plt.subplots(1,3)

		ax1.imshow(data[:,:,0])
		ax2.imshow(data[:,:,1])
		ax3.imshow(data[:,:,2])
		
	else:
		plt.imshow(data)

	plt.show()

	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())


