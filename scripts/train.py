"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa


# Root directory of the project
#ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_WEIGHTS_PATH = '/home/riggi/Data/MLData/NNWeights/mask_rcnn_coco.h5'

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class SidelobeConfig(Config):
    
	""" Configuration for training on the toy  dataset.
			Derives from the base Config class and overrides some values.
	"""
	# Give the configuration a recognizable name	
	NAME = "sidelobes"

	# We use a GPU with 12GB memory, which can fit two images.
	# Adjust down if you use a smaller GPU.
	IMAGES_PER_GPU = 2

	# Number of classes (including background)
	NUM_CLASSES = 1 + 1  # Background + sidelobes

	# Number of training steps per epoch
	STEPS_PER_EPOCH = 1000

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
	IMAGE_MIN_DIM = 192
	IMAGE_MAX_DIM = 192
	
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
#  Dataset
############################################################

class SidelobeDataset(utils.Dataset):

	def load_dataset(self, dataset):
		""" Load a subset of the Sidelobe dataset.
				dataset_dir: Root directory of the dataset.
		"""
		# Add classes. We have only one class to add
		class_id= 1
		self.add_class("sidelobe", class_id, "sidelobe")
 
		# Read dataset
		with open(dataset,'r') as f:
		
			for line in f:
				line_split = line.strip().split(',')
				(filename,filename_mask,class_name) = line_split

				filename_base= os.path.basename(filename)
				filename_base_noext= os.path.splitext(filename_base)[0]						

				self.add_image(
        	class_name,
					image_id=filename_base_noext,  # use file name as a unique image id
					path=filename,
					path_mask=filename_mask,
					class_id=class_id
				)


	def load_mask(self, image_id):
		""" Generate instance masks for an image.
				Returns:
					masks: A bool array of shape [height, width, instance count] with one mask per instance.
					class_ids: a 1D array of class IDs of the instance masks.
		"""
		# If not a sidelobe dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "sidelobe":
			return super(self.__class__, self).load_mask(image_id)

		# Set bitmap mask of shape [height, width, instance_count]
		info = self.image_info[image_id]
		filename= info["path_mask"]
		class_id= info["class_id"]

		# Read mask
		data, header= utils.read_fits(filename,stretch=False,normalize=False,convertToRGB=False)
		height= data.shape[0]
		width= data.shape[1]

		data= data.astype(np.bool)

		mask = np.zeros([height,width,1],dtype=np.bool)
		mask[:,:,0]= data

		instance_counts= np.full([mask.shape[-1]], class_id, dtype=np.int32)
		
		# Return mask, and array of class IDs of each instance
		return mask, instance_counts


	def load_image(self, image_id):
		"""Load the specified image and return a [H,W,3] Numpy array."""
		# Load image
		filename= self.image_info[image_id]['path']

		image, header= utils.read_fits(filename,stretch=True,normalize=True,convertToRGB=True)
		
		#image = skimage.io.imread(filename)
        
		# If grayscale. Convert to RGB for consistency.
		#if image.ndim != 3:
		#	image = skimage.color.gray2rgb(image)
		# If has an alpha channel, remove it for consistency
		#if image.shape[-1] == 4:
		#	image = image[..., :3]
		
		return image

	def image_reference(self, image_id):
		""" Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "sidelobe":
			return info["path"]
		else:
			super(self.__class__, self).image_reference(image_id)


	
def train(model,nepochs=10,nthreads=1):    
	"""Train the model."""
    
	# Training dataset.
	dataset_train = SidelobeDataset()
	dataset_train.load_dataset(args.dataset)
	dataset_train.prepare()

	# Validation dataset
	dataset_val = SidelobeDataset()
	dataset_val.load_dataset(args.dataset)
	dataset_val.prepare()

	# Image augmentation
	# http://imgaug.readthedocs.io/en/latest/source/augmenters.html
	augmentation = iaa.SomeOf((0, 2), 
		[
			iaa.Fliplr(0.5),
			iaa.Flipud(0.5),
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


def test(model):
	""" Test the model on input dataset """    
	dataset = SidelobeDataset()
	dataset.load_dataset(args.dataset)
	dataset.prepare()

	for index, image_id in enumerate(dataset.image_ids):
		# - Load image
		image = dataset.load_image(image_id)
		image_path = dataset.image_info[index]['path']
		image_path_base= os.path.basename(image_path)
		image_path_base_noext= os.path.splitext(image_path_base)[0]		

		# - Load mask
		# ...
		# ...		

		# Detect objects
		r = model.detect([image], verbose=0)[0]
		mask= r['masks']
        
		# Create regions from detections
		if mask.shape[-1] > 0:
			print("INFO: Mask found for image %s ..." % image_path_base)

			n_mask_true= 0
			for i in range(0,mask.shape[-1]):
				mask_data = mask[:,:,i]
				counts= np.count_nonzero(mask_data)
				if counts<=0:
					continue

				n_mask_true+= counts
				print("--> Printing mask no. %s " % (str(i+1)))
				print(mask_data)
				
			if n_mask_true>0:
				# Color splash
				splash = color_splash(image, mask)
	
				# Save output
				outfile = 'splash_' + image_path_base_noext + '.png'
				skimage.io.imsave(outfile, splash)

		else:
			print("INFO: No object mask found for image %s ..." % image_path_base)
			continue
		


def color_splash(image, mask):
	""" Apply color splash effect.
			image: RGB image [height, width, 3]
			mask: instance segmentation mask [height, width, instance count]

   		Returns result image.
	"""
	# Make a grayscale copy of the image. The grayscale copy still
	# has 3 RGB channels, though.
	gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
	# Copy color pixels from the original color image where mask is set
	if mask.shape[-1] > 0:
		# We're treating all instances as one, so collapse the mask into one layer
		mask = (np.sum(mask, -1, keepdims=True) >= 1)
		splash = np.where(mask, image, gray).astype(np.uint8)
	else:
		splash = gray.astype(np.uint8)

	return splash


def detect_and_color_splash(model, image_path):

	# Run model detection and generate the color splash effect
	print("Running on {}".format(args.image))
	
	# Read image
	#image = skimage.io.imread(args.image)
	image, header= utils.read_fits(filename=image_path,stretch=True,normalize=True,convertToRGB=True)

	# Detect objects
	r = model.detect([image], verbose=1)[0]

	# Color splash
	splash = color_splash(image, r['masks'])
	
	# Save output
	file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
	skimage.io.imsave(file_name, splash)


############################################################
#  Training
############################################################

if __name__ == '__main__':    
	import argparse

	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect sidelobes.')

	parser.add_argument("command",metavar="<command>",help="'train' or 'splash'")
	parser.add_argument('--dataset', required=False,metavar="/path/to/balloon/dataset/",help='Directory of the Sidelobe dataset')
	parser.add_argument('--weights', required=True,metavar="/path/to/weights.h5",help="Path to weights .h5 file or 'coco'")
	parser.add_argument('--logs', required=False,default=DEFAULT_LOGS_DIR,metavar="/path/to/logs/",help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--image', required=False,metavar="path or URL to image",help='Image to apply the color splash effect on')
	parser.add_argument('--nepochs', required=False,default=10,type=int,metavar="Number of training epochs",help='Number of training epochs')
	parser.add_argument('--weighttype', required=False,default='',metavar="Type of weights",help="Type of weights")
	parser.add_argument('--nthreads', required=False,default=1,type=int,metavar="Number of worker threads",help="Number of worker threads")
	
	args = parser.parse_args()

	# Validate arguments
	if args.command == "train":
		assert args.dataset, "Argument --dataset is required for training"
	elif args.command == "test":
		assert args.dataset, "Argument --dataset is required for testing"
	elif args.command == "splash":
		assert args.image, "Provide --image to apply color splash"

	print("Weights: ", args.weights)
	print("Dataset: ", args.dataset)
	print("Logs: ", args.logs)
	print("nEpochs: ",args.nepochs)

	# Configurations
	if args.command == "train":
		config = SidelobeConfig()
	else:
		class InferenceConfig(SidelobeConfig):
			# Set batch size to 1 since we'll be running inference on
			# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
			GPU_COUNT = 1
			IMAGES_PER_GPU = 1
		config = InferenceConfig()
	config.display()

	# Create model
	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs)
	else:
		model = modellib.MaskRCNN(mode="inference", config=config,model_dir=args.logs)

	# Select weights file to load
	weights_path = args.weights

	#if args.weights.lower() == "coco":
	#	weights_path = COCO_WEIGHTS_PATH
	#	# Download weights file
	#	if not os.path.exists(weights_path):
	#		utils.download_trained_weights(weights_path)
	#elif args.weights.lower() == "last":
	#	# Find last trained weights
	#	weights_path = model.find_last()
	#elif args.weights.lower() == "imagenet":
	#	# Start from ImageNet trained weights
	#	weights_path = model.get_imagenet_weights()
	#else:
	#	weights_path = args.weights

	# Load weights
	print("Loading weights ", weights_path)
	#if args.weights.lower() == "coco":
	if args.weighttype.lower() == "coco":
		# Exclude the last layers because they require a matching
		# number of classes
		model.load_weights(
			weights_path, by_name=True, 
			exclude=[
				"mrcnn_class_logits", "mrcnn_bbox_fc",
				"mrcnn_bbox", "mrcnn_mask"
			]
		)
	else:
		model.load_weights(weights_path, by_name=True)

	# Train or evaluate
	if args.command == "train":
		train(model,args.nepochs,args.nthreads)
	elif args.command == "test":
		test(model)	
	elif args.command == "splash":
		detect_and_color_splash(model, image_path=args.image)
	else:
		print("'{}' is not recognized. "
			"Use 'train' or 'splash'".format(args.command))



