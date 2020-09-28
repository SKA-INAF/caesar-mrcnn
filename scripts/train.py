"""
Mask R-CNN
Train on the radio galaxy/source/sidelobe dataset.

Copyright (c) 2020 Simone Riggi - INAF
Licensed under the GPL3 License (see LICENSE for details)

------------------------------------------------------------

Usage: Run from the command line as such:

    # Train a new model starting from pre-trained weights
    python3 train.py train --dataset=/path/to/dataset --weights=...

"""

import os
import sys
import json
import time
import argparse
import datetime
import numpy as np
import skimage.draw
import skimage.measure
import tensorflow as tf
from imgaug import augmenters as iaa
from skimage.measure import find_contours
from imgaug import augmenters as iaa
import uuid

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import logger
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.analyze import ModelTester
from mrcnn.graph import Graph

## Import graphics modules
import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class SDetectorConfig(Config):
    
	""" Configuration for training on the toy  dataset.
			Derives from the base Config class and overrides some values.
	"""
	# Give the configuration a recognizable name	
	NAME = "rg-dataset"

	# NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
	GPU_COUNT = 1

	# We use a GPU with 12GB memory, which can fit two images.
	# Adjust down if you use a smaller GPU.
	IMAGES_PER_GPU = 2

	# Number of classes (including background)
	##NUM_CLASSES = 1 + 5  # Background + Objects (sidelobes, sources, galaxy_C1, galaxy_C2, galaxy_C3)
	NUM_CLASSES = 1 + 3  # Background + Objects (sidelobes, sources, galaxy)

	# Number of training steps per epoch
	#STEPS_PER_EPOCH = 16000
	VALIDATION_STEPS = max(1, 200 // (IMAGES_PER_GPU*GPU_COUNT)) # 200 validation/test images
	STEPS_PER_EPOCH = ((16439 - 200) // (IMAGES_PER_GPU*GPU_COUNT)) #16439 total images
	#STEPS_PER_EPOCH = ((18888 - 200) // (IMAGES_PER_GPU*GPU_COUNT)) #18888 total images
	
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
#       DATASET CLASS
############################################################

class SourceDataset(utils.Dataset):

	""" Define dataset class """

	# ================================================================
	# ==   CONSTRUCTOR
	# ================================================================
	def __init__(self):
		utils.Dataset.__init__(self)
	
		self.class_id_map= {
			'bkg': 0,
			'sidelobe': 1,
			'source': 2,
			'galaxy': 3
		}
		
		# - Add classes
		self.add_class("rg-dataset", 1, "sidelobe")
		self.add_class("rg-dataset", 2, "source")
		self.add_class("rg-dataset", 3, "galaxy")
		

	# ================================================================
	# ==   LOAD DATASET FROM ASCII (row format: file,mask,class_id)
	# ================================================================
	def load_dataset(self, dataset):
		""" Load a subset of the source dataset.
				dataset_dir: Root directory of the dataset.
		"""
		 
		# - Read dataset
		with open(dataset,'r') as f:
		
			for line in f:
				line_split = line.strip().split(',')
				(filename,filename_mask,class_name) = line_split

				filename_base= os.path.basename(filename)
				filename_base_noext= os.path.splitext(filename_base)[0]	
				image_id= str(uuid.uuid1())

				class_id= 0
				if class_name in class_id_map:
					class_id= class_id_map.get(class_name)					

				self.add_image(
        	"rg-dataset",
					#image_id=filename_base_noext,  # use file name as a unique image id
					image_id=image_id,  # use file name as a unique image id
					path=filename,
					path_masks=[filename_mask],
					class_ids=[class_id]
				)

	# ================================================================
	# ==   LOAD DATASET FROM ASCII (row format: jsonfile)
	# ================================================================
	def load_dataset_json(self, dataset):
		""" Load dataset specified in a json filelist """
	
		# - Read json filelist
		with open(dataset,'r') as f:
			for filename in f:
				logger.info("Loading dataset info from file %s ..." % filename)

				# - Read json file
				try:
					json_file = open(filename)
				except IOError:
					logger.error("Failed to open file %s, skip it..." % filename)
					continue
	
				# - Read obj info
				d= json.load(json_file)				
				print(d)
					
				img_path= d['img']
				img_path_base= os.path.basename(img_path)
				img_path_base_noext= os.path.splitext(img_path_base)[0]
				img_id= str(uuid.uuid1())
	
				nobjs= len(d['objs'])
				logger.info("#%d objects present in file %s ..." % filename)
				
				mask_paths= []
				class_ids= []	
				
				for obj_dict in d['objs']:
					mask_path= obj_dict['mask']
					class_name= obj_dict['class']
					class_id= 0
					if class_name in self.class_id_map:
						class_id= self.class_id_map.get(class_name)	
					mask_paths.append(mask_path)
					class_ids.append(class_id)
				
				# - Add image & mask informations in dataset class
				self.add_image(
        	"rg-dataset",
					image_id=img_id,
					path=img_path,
					path_masks=mask_paths,
					class_ids=class_ids
				)
		
	# ================================================================
	# ==   LOAD GT MASKS (multiple objects per image)
	# ================================================================
	def load_gt_masks(self, image_id, binary=True):
		""" Load gt mask """

		# Read filename
		info = self.image_info[image_id]
		filenames= info["path_masks"]
		nobjs= len(filenames)

		# Read mask file and fill binary mask images
		mask = None	
		counter= 0

		for filename in filenames:
			data, header= utils.read_fits(filename,stretch=False,normalize=False,convertToRGB=False)
			height= data.shape[0]
			width= data.shape[1]
			if binary:
				data= data.astype(np.bool)
			if not mask:
				if binary:
					mask = np.zeros([height,width,nobjs],dtype=np.bool)
				else:
					mask = np.zeros([height,width,nobjs],dtype=np.int)
			mask[:,:,counter]= data
			counter+= 1
	
		return mask


	# ================================================================
	# ==   LOAD MASK (multiple objects per image)
	# ================================================================
	def load_mask(self, image_id):
		""" Generate instance masks for an image.
				Returns:
					masks: A bool array of shape [height, width, instance count] with one mask per instance.
					class_ids: a 1D array of class IDs of the instance masks.
		"""

		# - Check	dataset name
		if self.image_info[image_id]["source"] != "rg-dataset":
			return super(self.__class__, self).load_mask(image_id)

		# - Set bitmap mask of shape [height, width, instance_count]
		info = self.image_info[image_id]
		filenames= info["path_masks"]
		class_ids= info["class_ids"]

		# - Read mask files
		mask = None	
		counter= 0

		for filename in filenames:
			data, header= utils.read_fits(filename,stretch=False,normalize=False,convertToRGB=False)
			height= data.shape[0]
			width= data.shape[1]
			data= data.astype(np.bool)
			if not mask:
				mask = np.zeros([height,width,nobjs],dtype=np.bool)
			mask[:,:,counter]= data
			counter+= 1

		instance_counts= np.full([mask.shape[-1]], class_ids, dtype=np.int32)
		
		# - Return mask, and array of class IDs of each instance
		return mask, instance_counts


	# ================================================================
	# ==   LOAD IMAGE
	# ================================================================
	def load_image(self, image_id):
		"""Load the specified image and return a [H,W,3] Numpy array."""
		
		# - Load image
		filename= self.image_info[image_id]['path']
		image, header= utils.read_fits(filename,stretch=True,normalize=True,convertToRGB=True)
				
		return image

	# ================================================================
	# ==   GET IMAGE PATH
	# ================================================================
	def image_reference(self, image_id):
		""" Return the path of the image."""

		if info["source"] == "rg-dataset":
			return info["path"]
		else:
			super(self.__class__).image_reference(self, image_id)

	
############################################################
#             TRAIN
############################################################

def train(model,nepochs=10,nthreads=1):    
	"""Train the model."""
    
	# Training dataset.
	dataset_train = SourceDataset()
	dataset_train.load_dataset(args.dataset)
	dataset_train.prepare()

	# Validation dataset
	dataset_val = SourceDataset()
	dataset_val.load_dataset(args.dataset)
	dataset_val.prepare()

	# Image augmentation
	# http://imgaug.readthedocs.io/en/latest/source/augmenters.html
	augmentation = iaa.SomeOf((0, 2), 
		[
			iaa.Fliplr(1.0),
			iaa.Flipud(1.0),
			iaa.OneOf([iaa.Affine(rotate=90),iaa.Affine(rotate=180),iaa.Affine(rotate=270)])
		]
	)


	# *** This training schedule is an example. Update to your needs ***
	# Since we're using a very small dataset, and starting from
	# COCO trained weights, we don't need to train too long. Also,
	# no need to train all layers, just the heads should do it.
	print("INFO: Training network ...")
	model.train(dataset_train, dataset_val,	
		learning_rate=config.LEARNING_RATE,
		epochs=nepochs,
		augmentation=augmentation,
		#layers='heads',
		layers='all',
		n_worker_threads=nthreads
	)

############################################################
#        TEST
############################################################

def test(model):
	""" Test the model on input dataset """  

	dataset = SourceDataset()
	dataset.load_dataset(args.dataset)
	dataset.prepare()

	tester= ModelTester(model,config,dataset)	
	tester.score_thr= args.scoreThr_test
	tester.iou_thr= args.iouThr_test
	tester.n_max_img= args.nimg_test

	tester.test()


############################################################
#        PARSE/VALIDATE ARGS
############################################################

def parse_args():
	""" Parse command line arguments """  
  
	# - Parse command line arguments
	parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect radio sources.')

	parser.add_argument("command",metavar="<command>",help="'train' or 'test'")
	parser.add_argument('--dataset', required=False,metavar="/path/to/dataset/",help='Directory of the source dataset')
	parser.add_argument('--weights', required=False,metavar="/path/to/weights.h5",help="Path to weights .h5 file")
	parser.add_argument('--logs', required=False,default=DEFAULT_LOGS_DIR,metavar="/path/to/logs/",help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--image', required=False,metavar="path or URL to image",help='Image to apply the color splash effect on')
	parser.add_argument('--nepochs', required=False,default=10,type=int,metavar="Number of training epochs",help='Number of training epochs')
	parser.add_argument('--epoch_length', required=False,default=10,type=int,metavar="Number of data batches per epoch",help='Number of data batches per epoch')
	parser.add_argument('--nvalidation_steps', required=False,default=50,type=int,metavar="Number of validation steps per epoch",help='Number of validation steps per epoch')
	parser.add_argument('--ngpu', required=False,default=1,type=int,metavar="Number of GPUs",help='Number of GPUs')
	parser.add_argument('--nimg_per_gpu', required=False,default=1,type=int,metavar="Number of images per gpu",help='Number of images per gpu')
	parser.add_argument('--weighttype', required=False,default='',metavar="Type of weights",help="Type of weights")
	parser.add_argument('--nthreads', required=False,default=1,type=int,metavar="Number of worker threads",help="Number of worker threads")
	parser.add_argument('--nimg_test', required=False,default=-1,type=int,metavar="Number of images in dataset to inspect during test",help="Number of images in dataset to inspect during test")	
	parser.add_argument('--scoreThr_test', required=False,default=0.7,type=float,metavar="Object detection score threshold to be used during test",help="Object detection score threshold to be used during test")
	parser.add_argument('--iouThr_test', required=False,default=0.6,type=float,metavar="IOU threshold used to match detected objects with true objects",help="IOU threshold used to match detected objects with true objects")

	args = parser.parse_args()

	#return vars(args)
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

	#===========================
	#==   VALIDATE ARGS
	#===========================
	logger.info("Validating script args ...")
	if args.command == "train":
		assert args.dataset, "Argument --dataset is required for training"
	elif args.command == "test":
		assert args.dataset, "Argument --dataset is required for testing"
	else:
		logger.error("Unknown command given (%s), valid commands are {train,test}!" % args.command)
		return 1

	print("Weights: ", args.weights)
	print("Dataset: ", args.dataset)
	print("Logs: ", args.logs)
	print("nEpochs: ",args.nepochs)
	print("epoch_length: ",args.epoch_length)
	print("nvalidation_steps: ",args.nvalidation_steps)
	print("ngpu: ",args.ngpu)
	print("nimg_per_gpu: ",args.nimg_per_gpu)
	print("nimg_test: ",args.nimg_test)
	print("scoreThr_test: ",args.scoreThr_test)

	weights_path = args.weights

	train_from_scratch= False
	if not weights_path or weights_path=='':
		train_from_scratch= True

	#===========================
	#==   CONFIG
	#===========================
	if args.command == "train":
		config = SDetectorConfig()
		config.GPU_COUNT = args.ngpu
		config.IMAGES_PER_GPU = args.nimg_per_gpu
		config.VALIDATION_STEPS = max(1, args.nvalidation_steps // (config.IMAGES_PER_GPU*config.GPU_COUNT)) # 200 validation/test images
		config.STEPS_PER_EPOCH = ((args.epoch_length - args.nvalidation_steps) // (config.IMAGES_PER_GPU*config.GPU_COUNT)) #16439 total images
		#config.STEPS_PER_EPOCH= args.epoch_length
	elif args.command == "test":
		class InferenceConfig(SDetectorConfig):
			# Set batch size to 1 since we'll be running inference on
			# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
			GPU_COUNT = 1
			IMAGES_PER_GPU = 1
		config = InferenceConfig()

	config.display()

	#===========================
	#==   CREATE MODEL
	#===========================
	# - Create model
	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs)
	elif args.command == "test":
		# Device to load the neural network on.
		# Useful if you're training a model on the same 
		# machine, in which case use CPU and leave the
		# GPU for training.
		DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
		with tf.device(DEVICE):
			model = modellib.MaskRCNN(mode="inference", config=config,model_dir=args.logs)

	# - Load weights
	if train_from_scratch:
		logger.info("No weights given, training from scratch ...")
	else:	
		logger.info("Loading weights from file %s ..." % weights_path)
		model.load_weights(weights_path, by_name=True)
	
	#===========================
	#==   TRAIN/TEST
	#===========================
	# - Train or evaluate
	if args.command == "train":
		train(model,args.nepochs,args.nthreads)
	elif args.command == "test":
		test(model)	
	
	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())

