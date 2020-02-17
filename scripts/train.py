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

## ASTRO MODULES
from astropy.io import ascii, fits
from astropy.units import Quantity
from astropy.modeling.parameters import Parameter
from astropy.modeling.core import Fittable2DModel
from astropy import wcs
from astropy import units as u
from astropy.visualization import ZScaleInterval

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
	STEPS_PER_EPOCH = 100

	# Skip detections with < 90% confidence
	DETECTION_MIN_CONFIDENCE = 0.9


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
		data, header= self.read_fits(filename,stretch=False,normalize=False,convertToRGB=False)
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

		stretch= True
		normalize= True
		convertToRGB= True
		image, header= self.read_fits(filename,stretch,normalize=True,convertToRGB=True)
		
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


	def read_fits(self,filename,stretch=True,normalize=True,convertToRGB=True):
		""" Read FITS image """
	
		# - Open file
		try:
			hdu= fits.open(filename,memmap=False)
		except Exception as ex:
			errmsg= 'Cannot read image file: ' + filename
			logger.error(errmsg)
			return None

		# - Read data
		data= hdu[0].data
		data_size= np.shape(data)
		nchan= len(data.shape)
		if nchan==4:
			output_data= data[0,0,:,:]
		elif nchan==2:
			output_data= data	
		else:
			errmsg= 'Invalid/unsupported number of channels found in file ' + filename + ' (nchan=' + str(nchan) + ')!'
			hdu.close()
			logger.error(errmsg)
			return None

		# - Convert data to float 32
		output_data= output_data.astype(np.float32)

		# - Read metadata
		header= hdu[0].header

		# - Close file
		hdu.close()

		# - Replace nan values with min pix value
		img_min= np.nanmin(output_data)
		output_data[np.isnan(output_data)]= img_min	

		# - Stretch data using zscale transform
		if stretch:
			data_stretched= self.stretch_img(output_data)
			output_data= data_stretched
			output_data= output_data.astype(np.float32)

		# - Normalize data to [0,255]
		if normalize:
			data_norm= self.normalize_img(output_data)
			output_data= data_norm
			output_data= output_data.astype(np.float32)

		# - Convert to RGB image
		if convertToRGB:
			if not normalize:
				data_norm= self.normalize_img(output_data)
				output_data= data_norm
			data_rgb= self.gray2rgb(output_data) 
			output_data= data_rgb

		return output_data, header
	
	
	def stretch_img(self,data,contrast=0.25):
		""" Apply z-scale stretch to image """
		
		transform= ZScaleInterval(contrast=contrast)
		data_stretched= transform(data)
	
		return data_stretched

	def normalize_img(self,data):
		""" Normalize image to (0,1) """
	
		data_max= np.max(data)
		data_norm= data/data_max

		return data_norm

	def gray2rgb(self,data_float):
		""" Convert gray image data to rgb """

		# - Convert to uint8
		data_uint8 = np.array( (data_float*255).round(), dtype = np.uint8)
	
		# - Convert to uint8 3D
		data3_uint8 = np.stack((data_uint8,)*3, axis=-1)

		return data3_uint8


def train(model):    
	"""Train the model."""
    
	# Training dataset.
	dataset_train = SidelobeDataset()
	dataset_train.load_dataset(args.dataset)
	dataset_train.prepare()

	# Validation dataset
	dataset_val = SidelobeDataset()
	dataset_val.load_dataset(args.dataset)
	dataset_val.prepare()

	# *** This training schedule is an example. Update to your needs ***
	# Since we're using a very small dataset, and starting from
	# COCO trained weights, we don't need to train too long. Also,
	# no need to train all layers, just the heads should do it.
	print("Training network heads")
	model.train(dataset_train, dataset_val,	
		learning_rate=config.LEARNING_RATE,
		epochs=30,
		layers='heads'
	)


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
	image = skimage.io.imread(args.image)
	
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

	args = parser.parse_args()

	# Validate arguments
	if args.command == "train":
		assert args.dataset, "Argument --dataset is required for training"
	elif args.command == "splash":
		assert args.image or args.video, "Provide --image or --video to apply color splash"

	print("Weights: ", args.weights)
	print("Dataset: ", args.dataset)
	print("Logs: ", args.logs)

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
	if args.weights.lower() == "coco":
		weights_path = COCO_WEIGHTS_PATH
		# Download weights file
		if not os.path.exists(weights_path):
			utils.download_trained_weights(weights_path)
	elif args.weights.lower() == "last":
		# Find last trained weights
		weights_path = model.find_last()
	elif args.weights.lower() == "imagenet":
		# Start from ImageNet trained weights
		weights_path = model.get_imagenet_weights()
	else:
		weights_path = args.weights

	# Load weights
	print("Loading weights ", weights_path)
	if args.weights.lower() == "coco":
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
		train(model)
	elif args.command == "splash":
		detect_and_color_splash(model, image_path=args.image)
	else:
		print("'{}' is not recognized. "
			"Use 'train' or 'splash'".format(args.command))



