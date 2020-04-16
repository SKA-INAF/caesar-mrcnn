############################################################
#              MODULE IMPORTS
############################################################
# - Standard modules
import os
import sys
import json
import time
import datetime
import numpy as np

# - MRCNN modules
from mrcnn import __version__, __date__
from mrcnn import logger

#from mrcnn import model as modellib, utils
from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn.classifier import SClassifier



# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
ROOT_DIR = os.getcwd()
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#             CONFIG OPTIONS
############################################################

class SClassifierConfig(Config):
    
	""" Configuration for training on the toy  dataset.
			Derives from the base Config class and overrides some values.
	"""
	# Give the configuration a recognizable name	
	NAME = "sources"

	# We use a GPU with 12GB memory, which can fit two images.
	# Adjust down if you use a smaller GPU.
	IMAGES_PER_GPU = 1
	GPU_COUNT = 1

	# Number of classes (including background)
	NUM_CLASSES = 1 + 4  # Background + Objects (sidelobes, sources, galaxy_C2, galaxy_C3)
	CLASS_NAMES= ["bkg","sidelobe","source","galaxy_C2","galaxy_C3"]

	# Number of training steps per epoch
	STEPS_PER_EPOCH = 1

	# Don't exclude based on confidence. Since we have two classes
	# then 0.5 is the minimum anyway as it picks between source and BG
	DETECTION_MIN_CONFIDENCE = 0 # default=0.9 (skip detections with <90% confidence)

	# Non-maximum suppression threshold for detection
	DETECTION_NMS_THRESHOLD = 0.3

	# Length of square anchor side in pixels
	RPN_ANCHOR_SCALES = (4,8,16,32,64)

	# Maximum number of ground truth instances to use in one image
	MAX_GT_INSTANCES = 300 # default=100

	# Use a smaller backbone
	BACKBONE = "resnet101"

	# The strides of each layer of the FPN Pyramid. These values
	# are based on a Resnet101 backbone.
	BACKBONE_STRIDES = [4, 8, 16, 32, 64]
	
	# Input image resizing
	# Generally, use the "square" resizing mode for training and predicting
	# and it should work well in most cases. In this mode, images are scaled
	# up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
	# scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
	# padded with zeros to make it a square so multiple images can be put
	# in one batch.
	# Available resizing modes:
	# none:   No resizing or padding. Return the image unchanged.
	# square: Resize and pad with zeros to get a square image
	#         of size [max_dim, max_dim].
	# pad64:  Pads width and height with zeros to make them multiples of 64.
	#         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
	#         up before padding. IMAGE_MAX_DIM is ignored in this mode.
	#         The multiple of 64 is needed to ensure smooth scaling of feature
	#         maps up and down the 6 levels of the FPN pyramid (2**6=64).
	# crop:   Picks random crops from the image. First, scales the image based
	#         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
	#         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
	#         IMAGE_MAX_DIM is not used in this mode.
	IMAGE_RESIZE_MODE = "square"
	IMAGE_MIN_DIM = 256
	IMAGE_MAX_DIM = 256
	
	# Image mean (RGB)
	#MEAN_PIXEL = np.array([112,112,112])
	# Image mean (RGB) - consider setting these to zero, and do per image mean/std normalization
	MEAN_PIXEL = np.array([0, 0, 0])

	# Non-max suppression threshold to filter RPN proposals.
	# You can increase this during training to generate more propsals.
	RPN_NMS_THRESHOLD = 0.9 # default=0.7

	# How many anchors per image to use for RPN training
	RPN_TRAIN_ANCHORS_PER_IMAGE = 512  # default=128

	# Number of ROIs per image to feed to classifier/mask heads	
	# The Mask RCNN paper uses 512 but often the RPN doesn't generate
	# enough positive proposals to fill this and keep a positive:negative
	# ratio of 1:3. You can increase the number of proposals by adjusting
	# the RPN NMS threshold.
	TRAIN_ROIS_PER_IMAGE = 512


	# Ratios of anchors at each cell (width/height)
	# A value of 1 represents a square anchor, and 0.5 is a wide anchor
	RPN_ANCHOR_RATIOS = [0.5, 1, 2]

	## Learning rate and momentum
	## The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
	## weights to explode. Likely due to differences in optimizer
	## implementation.
	LEARNING_RATE = 0.0005
	# LEARNING_MOMENTUM = 0.9
	OPTIMIZER = "ADAM" # default is SGD

	# If enabled, resizes instance masks to a smaller size to reduce
	# memory load. Recommended when using high-resolution images.
	USE_MINI_MASK = False



############################################################
#           SOURCE CLASSIFIER
############################################################

#if __name__ == '__main__': 

def main():
	"""Main function"""

	# =================================
	# ==       CONFIG OPTIONS
	# =================================   
	# - Define options
	import argparse

	parser = argparse.ArgumentParser(description='Use Mask R-CNN to classify radio sources')
	parser.add_argument('--image', required=True,type=str,metavar="path to image",help='Image to apply the color splash effect on')
	parser.add_argument('--scatalog', required=True,type=str,metavar="/path/to/scatalog.root",help='Path to Caesar source catalog file (.root)')
	parser.add_argument('--weights', required=True,type=str,metavar="/path/to/weights.h5",help="Path to weights .h5 file")

	parser.add_argument('--logs', required=False,default=DEFAULT_LOGS_DIR,metavar="/path/to/logs/",help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--nthreads', required=False,default=1,type=int,metavar="Number of worker threads",help="Number of worker threads")		
	parser.add_argument('--scoreThr', required=False,default=0.7,type=float,metavar="score threshold",help="Object detection score threshold to be used during test")
	parser.add_argument('--iouThr', required=False,default=0.6,type=float,metavar="IOU threshold",help="IOU threshold used to match detected objects with true objects")
	parser.add_argument('--nsources_max', required=False,default=-1,type=int,metavar="Max number of sources to be processed",help="Max number of sources to be processed")
	parser.add_argument('--scutout_size', required=False,default=132,type=int,metavar="Source image size",help="Source cutout image size")
	
	args = parser.parse_args()

	
	# - Validate arguments
	assert args.image, "Provide --image "
	assert args.scatalog, "Provide --scatalog "
	assert args.weights, "Provide --weights "

	# - Get options
	weights_path = args.weights

	print("Image: ", args.image)
	print("Source catalog: ", args.scatalog)
	print("Weights: ", args.weights)
	print("Logs: ", args.logs)
	print("scoreThr: ",args.scoreThr)
	print("iouThr: ",args.iouThr)
	print("nsources_max: ",args.nsources_max)
	print("scutout_size: ",args.scutout_size)

	# - Set configurations
	config = SClassifierConfig()
	config.display()

	# =================================
	# ==       BUILD MODEL
	# =================================  
	# - Create model for inference
	logger.info("Creating model according to given config ...")
	model = modellib.MaskRCNN(mode="inference", config=config,model_dir=args.logs)

	# - Load weights
	logger.info("Loading weights %s ..." % args.weights)
	model.load_weights(args.weights,by_name=True)

	# =================================
	# ==       CLASSIFY SOURCES
	# ================================= 
	# - Create classifier
	logger.info("Creating classifier ...")
	classifier= SClassifier(model,config)
	classifier.n_max_sources= args.nsources_max
	classifier.iou_thr= args.iouThr
	classifier.score_thr= args.scoreThr
	classifier.scutout_size= args.scutout_size

	# - Run classifier
	logger.info("Running source classification ...")	
	status= classifier.run(args.image,args.scatalog)

	if status<0:
		logger.error("Failed to run source classifier!")
		return 1

	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
