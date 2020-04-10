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
import time
import datetime
import numpy as np
import skimage.draw
import skimage.measure
from skimage.measure import find_contours
from imgaug import augmenters as iaa


# Root directory of the project
#ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.analyze import ModelTester
from mrcnn.graph import Graph

## Import graphics modules
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon

# Path to trained weights file
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_WEIGHTS_PATH = '/home/riggi/Data/MLData/NNWeights/mask_rcnn_coco.h5'

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
	NAME = "sources"

	# We use a GPU with 12GB memory, which can fit two images.
	# Adjust down if you use a smaller GPU.
	IMAGES_PER_GPU = 2

	# Number of classes (including background)
	NUM_CLASSES = 1 + 4  # Background + Objects (sidelobes, sources, galaxy_C2, galaxy_C3)

	# Number of training steps per epoch
	STEPS_PER_EPOCH = 16000

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
#  Dataset
############################################################

class SourceDataset(utils.Dataset):

	def load_dataset(self, dataset):
		""" Load a subset of the source dataset.
				dataset_dir: Root directory of the dataset.
		"""
		# Add classes. We have only one class to add
		class_id_map= {
			'bkg': 0,
			'sidelobe': 1,
			'source': 2,
			'galaxy_C2': 3,
			'galaxy_C3': 4,
		}
		self.add_class("sources", 1, "sidelobe")
		self.add_class("sources", 2, "source")
		self.add_class("sources", 3, "galaxy_C2")
		self.add_class("sources", 4, "galaxy_C3")
		 
		

		# Read dataset
		with open(dataset,'r') as f:
		
			for line in f:
				line_split = line.strip().split(',')
				(filename,filename_mask,class_name) = line_split

				filename_base= os.path.basename(filename)
				filename_base_noext= os.path.splitext(filename_base)[0]	

				class_id= 0
				if class_name in class_id_map:
					class_id= class_id_map.get(class_name)					

				self.add_image(
        	#class_name,
					"sources",
					image_id=filename_base_noext,  # use file name as a unique image id
					path=filename,
					path_mask=filename_mask,
					class_id=class_id
				)


	def load_gt_mask(self, image_id):
		""" Load gt mask """

		# Read filename
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
	
		return mask

	def load_gt_mask_nonbinary(self, image_id):
		""" Load gt mask as non binary """

		# Read filename
		info = self.image_info[image_id]
		filename= info["path_mask"]
		class_id= info["class_id"]

		# Read mask
		data, header= utils.read_fits(filename,stretch=False,normalize=False,convertToRGB=False)
		height= data.shape[0]
		width= data.shape[1]
		#data= data.astype(np.bool)
		
		mask = np.zeros([height,width,1],dtype=np.int)
		mask[:,:,0]= data
	
		return mask


	def load_mask(self, image_id):
		""" Generate instance masks for an image.
				Returns:
					masks: A bool array of shape [height, width, instance count] with one mask per instance.
					class_ids: a 1D array of class IDs of the instance masks.
		"""

		# Check	
		if self.image_info[image_id]["source"] != "sources":
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
				
		return image


	def image_reference(self, image_id):
		""" Return the path of the image."""

		#info = self.image_info[image_id]
		#return info["path"]

		if info["source"] == "sources":
			return info["path"]
		else:
			super(self.__class__).image_reference(self, image_id)

	
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

def merge_masks(mask1,mask2):
	""" Merge masks """
	mask= mask1 + mask2
	mask[mask>1]= 1	
	return mask

def extract_mask_connected_components(mask):
	""" Extract mask components """

	labels, ncomponents= skimage.measure.label(mask, background=0, return_num=True, connectivity=1)
	return labels, ncomponents


def are_mask_connected(mask1,mask2):
	""" Check if two masks are connected """

	# - Find how many components are found in both masks
	labels1, ncomponents1= extract_mask_connected_components(mask1)
	labels2, ncomponents2= extract_mask_connected_components(mask2)
	
	# - Merge masks
	mask= merge_masks(mask1,mask2)

	# - Find how many components are found in mask sum
	#   If <ncomp1+ncomp2 the masks are not connected 
	labels, ncomponents= extract_mask_connected_components(mask)

	if ncomponents==ncomponents1+ncomponents2:
		connected= False
	else:
		connected= True

	return connected



def inspect_results(image_id,image_path,model,dataset,score_thr=0.7,iou_thr=0.6):
	""" Inspect results on given image """

	# - Load image
	image = dataset.load_image(image_id)
	image_path_base= os.path.basename(image_path)
	image_path_base_noext= os.path.splitext(image_path_base)[0]		

	# - Get detector result
	r = model.detect([image], verbose=0)[0]
	class_names= dataset.class_names
	masks= r['masks']
	boxes= r['rois']
	##bboxes= utils.extract_bboxes(mask)
	class_ids= r['class_ids']
	scores= r['scores']
	nobjects= masks.shape[-1]
	N = boxes.shape[0]
	if nobjects <= 0:
		print("INFO: No object mask found for image %s ..." % image_path_base)
		return 0

	print("INFO: %d objects (%d boxes) found in this image ..." % (nobjects,N))

	# - Set colors
	class_color_map= {
		'bkg': (0,0,0),# black
		'sidelobe': (1,1,0),# yellow
		'source': (1,0,0),# red
		'galaxy_C2': (0,0,1),# blue
		'galaxy_C3': (0,1,0),# green
	}

	# - Select detected objects with score larger than threshold
	masks_sel= []
	class_ids_sel= []
	scores_sel= []
	nobjects_sel= 0

	for i in range(N):
		mask= masks[:, :, i]
		class_id = class_ids[i]
		score = scores[i]
		label = class_names[class_id]
		caption = "{} {:.3f}".format(label, score)
		if score<score_thr:
			print("INFO: Skipping object %s (id=%d) with score %f<thr=%f ..." % (label,class_id,score,score_thr))
			continue

		print("INFO: Selecting object %s (id=%d) with score %f>thr=%f ..." % (label,class_id,score,score_thr))
		masks_sel.append(mask)
		class_ids_sel.append(class_id)
		scores_sel.append(score)
		nobjects_sel+= 1
		
	print("INFO: %d objects selected in this image ..." % nobjects_sel)

	# - Sort objects by descending scores
	sort_indices= np.argsort(scores_sel)[::-1]

	# - Separate all detected objects which are not connected.
	#   NB: This is done only for sources & sidelobes not for galaxies
	masks_det= []
	class_ids_det= []
	scores_det= []
	nobjects_det= 0

	for index in sort_indices:
		mask= masks_sel[index]	
		class_id= class_ids_sel[index]
		label= class_names[class_id]
		score= scores_sel[index]

		#print("INFO: Mask shape for object no. %d " % index)
		#print(mask.shape)

		# - Skip if class id is galaxy
		if label=='galaxy_C2' or label=='galaxy_C3':
			masks_det.append(mask)
			class_ids_det.append(class_id)
			scores_det.append(score)
			print("INFO: Selecting object %s (id=%d) with score %f>thr=%f ..." % (label,class_id,score,score_thr))
			continue

		# - Extract components masks
		component_labels, ncomponents= extract_mask_connected_components(mask)
		print("INFO: Found %d sub components in mask no. %d ..." % (ncomponents,index))
		
		# - Extract indices of components and create masks for extracted components
		indices = np.indices(mask.shape).T[:,:,[1, 0]]
		for i in range(ncomponents):	
			mask_indices= indices[component_labels==i+1]
			extracted_mask= np.zeros(mask.shape,dtype=mask.dtype)
			extracted_mask[mask_indices[:,0],mask_indices[:,1]]= 1

			masks_det.append(extracted_mask)
			class_ids_det.append(class_id)
			scores_det.append(score)
			print("INFO: Selecting object %s (id=%d) with score %f>thr=%f ..." % (label,class_id,score,score_thr))
			

	print("INFO: Found %d components overall (after non-connected component extraction) in this image ..." % (len(masks_det)))
		
	# - Init undirected graph
	#   Add links between masks that are connected
	N= len(masks_det)
	g= Graph(N)
	for i in range(N):
		for j in range(i+1,N):
			connected= are_mask_connected(masks_det[i],masks_det[j])
			same_class= (class_ids_det[i]==class_ids_det[j])
			mergeable= (connected and same_class)
			if mergeable:
				print("INFO: Mask (%d,%d) have connected components and can be merged..." % (i,j))
				g.addEdge(i,j)

	# - Compute connected masks
	cc = g.connectedComponents()
	print(cc) 

	# - Merge connected masks
	masks_merged= []
	class_ids_merged= []
	scores_merged= []

	for i in range(len(cc)):
		if not cc[i]:
			continue
		
		score_avg= 0
		n_merged= len(cc[i])

		for j in range(n_merged):
			index= cc[i][j]
			mask= masks_det[index]
			class_id= class_ids_det[index]
			score= scores_det[index]
			score_avg+= score

			print("INFO: Merging mask no. %d ..." % index)
			if j==0:
				merged_mask= mask
				merged_score= score
			else:
				merged_mask= merge_masks(merged_mask,mask)
	
		score_avg*= 1./n_merged	
		masks_merged.append(merged_mask)
		class_ids_merged.append(class_id)
		scores_merged.append(score_avg)
		
	print("INFO: #%d masks finally selected..." % len(masks_merged))


	# - Find if there are overlapping masks with different class id
	#   If so retain the one with largest score
	N_final= len(masks_merged)
	g_final= Graph(N_final)
	for i in range(N_final):
		for j in range(i+1,N_final):
			connected= are_mask_connected(masks_merged[i],masks_merged[j])
			same_class= (class_ids_merged[i]==class_ids_merged[j])
			mergeable= connected
			if mergeable:
				print("INFO: Merged mask (%d,%d) have connected components and are selected for final selection..." % (i,j))
				g_final.addEdge(i,j)

	cc_final = g_final.connectedComponents()
	masks_final= []
	class_ids_final= []
	scores_final= []

	for i in range(len(cc_final)):
		if not cc_final[i]:
			continue
		
		score_best= 0
		index_best= -1
		class_id_best= 0
		n_overlapped= len(cc_final[i])

		for j in range(n_overlapped):
			index= cc_final[i][j]
			mask= masks_merged[index]
			class_id= class_ids_merged[index]
			score= scores_merged[index]
			if score>score_best:	
				score_best= score		
				index_best= index
				class_id_best= class_id
			
		print("INFO: Mask with index %s (score=%f, class=%d) selected as the best among all the overlapping masks..." % (index_best,score_best,class_id_best))
		masks_final.append(masks_merged[index_best])
		class_ids_final.append(class_ids_merged[index_best])
		scores_final.append(scores_merged[index_best])
		
	print("INFO: #%d masks finally selected..." % len(masks_final))

	# - Compute bounding boxes & image captions from selected masks
	bboxes= []
	captions= []
	for i in range(len(masks_final)):
		mask= masks_final[i]
		height= mask.shape[0]
		width= mask.shape[1]
		mask_expanded = np.zeros([height,width,1],dtype=np.bool)
		mask_expanded[:,:,0]= mask
		bbox= utils.extract_bboxes(mask_expanded)
		bboxes.append(bbox[0])
	
		label= class_names[class_ids_final[i]]
		score= scores_final[i]
		caption = "{} {:.3f}".format(label, score)
		captions.append(caption)

	# - Inspect ground truth masks
	#masks_gt= dataset.load_gt_mask(image_id)
	masks_gt= dataset.load_gt_mask_nonbinary(image_id)
	class_id_gt = dataset.image_info[image_id]["class_id"]
	label_gt= class_names[class_id_gt]
	color_gt = class_color_map[label_gt]
	caption_gt = label_gt

	masks_gt_det= []
	class_ids_gt_det= []

	for k in range(masks_gt.shape[-1]):
		mask_gt= masks_gt[:,:,k]
		if label_gt=='galaxy_C2' or label_gt=='galaxy_C3':
			masks_gt_det.append(mask_gt)
			class_ids_gt_det.append(class_id_gt)
			continue

		component_labels_gt, ncomponents_gt= extract_mask_connected_components(mask_gt)
		print("INFO: Found %d sub components in gt mask no. %d ..." % (ncomponents_gt,k))
		
		indices = np.indices(mask_gt.shape).T[:,:,[1, 0]]
		for i in range(ncomponents_gt):	
			mask_indices= indices[component_labels_gt==i+1]
			extracted_mask= np.zeros(mask_gt.shape,dtype=mask_gt.dtype)
			extracted_mask[mask_indices[:,0],mask_indices[:,1]]= 1

			# - Extract true object id from gt mask pixel values (1=sidelobes,2=sources,3=...)
			#   Override class_id_gt
			object_classid= mask_gt[mask_indices[0,0],mask_indices[0,1]]
			print("INFO: gt mask no. %d (subcomponent no. %d): object_classid=%d ..." % (k,i,object_classid))

			masks_gt_det.append(extracted_mask)
			#class_ids_gt_det.append(class_id_gt)
			class_ids_gt_det.append(object_classid)
			
	N= len(masks_gt_det)
	g= Graph(N)
	for i in range(N):
		for j in range(i+1,N):
			connected= are_mask_connected(masks_gt_det[i],masks_gt_det[j])
			same_class= (class_ids_gt_det[i]==class_ids_gt_det[j])
			mergeable= (connected and same_class)
			if mergeable:
				print("INFO: GT mask (%d,%d) have connected components and can be merged..." % (i,j))
				g.addEdge(i,j)

	cc = g.connectedComponents()
	print(cc) 

	masks_gt_merged= []
	class_ids_gt_merged= []
	
	for i in range(len(cc)):
		if not cc[i]:
			continue
		
		n_merged= len(cc[i])

		for j in range(n_merged):
			index= cc[i][j]
			mask= masks_gt_det[index]
			class_id= class_ids_gt_det[index]
			
			print("INFO: Merging GT mask no. %d (class_id=%d) ..." % (index,class_id))
			if j==0:
				merged_mask= mask
				merged_score= score
			else:
				merged_mask= merge_masks(merged_mask,mask)
	
		masks_gt_merged.append(merged_mask)
		class_ids_gt_merged.append(class_id)
		
	bboxes_gt= []
	captions_gt= []
	for i in range(len(masks_gt_merged)):
		mask= masks_gt_merged[i]
		height= mask.shape[0]
		width= mask.shape[1]
		mask_expanded = np.zeros([height,width,1],dtype=np.bool)
		mask_expanded[:,:,0]= mask
		bbox= utils.extract_bboxes(mask_expanded)
		bboxes_gt.append(bbox[0])
	
		label= class_names[class_ids_gt_merged[i]]
		caption = label
		captions_gt.append(caption)	
		

	#bboxes_gt= utils.extract_bboxes(mask_gt)
	#class_id_gt = dataset.image_info[image_id]["class_id"]
	
	#################################
	##    COMPUTE DET PERFORMANCES
	#################################
	# - Init data
	n_classes= config.NUM_CLASSES
	confusion_matrix= np.zeros((n_classes,n_classes))
	confusion_matrix_norm= np.zeros((n_classes,n_classes))	
	purity= np.zeros((1,n_classes))
	nobjs_true= np.zeros((1,n_classes))
	nobjs_det= np.zeros((1,n_classes))
	nobjs_det_right= np.zeros((1,n_classes))
	
	# - Loop over gt boxes and find associations to det boxes
	for i in range(len(bboxes_gt)):
		bbox_gt= bboxes_gt[i]
		class_id_gt= class_ids_gt_merged[i]
		nobjs_true[0][class_id_gt]+= 1
		nobjs_gt[0][class_id_gt]+= 1

		# - Find associations between true and detected objects according to largest IOU
		index_best= -1
		iou_best= 0
		for j in range(len(bboxes)):
			class_id= class_ids_final[j]
			bbox= bboxes[j]
			iou= utils.get_iou(bbox, bbox_gt)
			print("DEBUG: IOU(det=%d,true=%d)=%f" % (j,i,iou))
			if iou>iou_thr and iou>=iou_best:
				index_best= j
				iou_best= iou

		# - Update confusion matrix
		if index_best==-1:
			print("INFO: True object no. %d (class_id=%d) not associated to any detected object ..." % (i+1,class_id_gt))
		else:
			class_id_det= class_ids_final[index_best]
			confusion_matrix[class_id_gt][class_id_det]+= 1
			classification_matrix[class_id_gt][class_id_det]+= 1
			print("INFO: True object no. %d (class_id=%d) associated to detected object no. %d (class_id=%d) ..." % (i+1,class_id_gt,index_best,class_id_det))
			

	# - Normalize confusion matrix
	for i in range(n_classes):
		norm= nobjs_true[0][i]
		if norm<=0:
			continue
		for j in range(n_classes):
			C= confusion_matrix[i][j]
			C_norm= C/norm
			confusion_matrix_norm[i][j]= C_norm

	# - Compute purity
	for j in range(len(bboxes)):
		bbox= bboxes[j]
		class_id= class_ids_final[j]
		nobjs_det[0][class_id]+= 1

		# - Find association to true box
		index_best= -1
		iou_best= 0
		for i in range(len(bboxes_gt)):
			bbox_gt= bboxes_gt[i]
			class_id_gt= class_ids_gt_merged[i]	
			iou= utils.get_iou(bbox, bbox_gt)
			if iou>iou_thr and iou>=iou_best:
				index_best= i
				iou_best= iou
		
		# - Check if correctly detected
		if index_best!=-1:
			class_id_det= class_ids_gt_merged[index_best]	
			if class_id==class_id_det:
				nobjs_det_right[0][class_id]+= 1

	
	for j in range(n_classes):
		if nobjs_det[0][j]<=0:
			continue
		p= nobjs_det_right[0][j]/nobjs_det[0][j]
		purity[0][j]= p
	


	# - Print confusion matrix
	print("== SAMPLE NOBJ TRUE ==")
	print(nobjs_true)

	print("== SAMPLE NOBJ DET ==")
	print(nobjs_det)

	print("== SAMPLE NOBJ DET CORRECTLY ==")
	print(nobjs_det_right)

	print("== SAMPLE CLASSIFICATION MATRIX (not normalized) ==")
	print(confusion_matrix)

	print("== SAMPLE CLASSIFICATION MATRIX (normalized) ==")
	print(confusion_matrix_norm)

	print("== SAMPLE PURITY ==")
	print(purity)

	####################
	##    DRAW PLOT
	####################
	# - Create axis
	print("DEBUG: Create axis...")
	height, width = image.shape[:2]
	#figsize=(height,width)
	figsize=(16,16)
	fig, ax = plt.subplots(1, figsize=figsize)
	
	# - Show area outside image boundaries
	print("DEBUG: Show area outside image boundaries...")
	title= image_path_base_noext
	#ax.set_ylim(height + 10, -10)
	#ax.set_xlim(-10, width + 10)
	ax.set_ylim(height + 2, -2)
	ax.set_xlim(-2, width + 2)
	ax.axis('off')
	ax.set_title(title,fontsize=30)
	
	#ax.set_frame_on(False)

	masked_image = image.astype(np.uint32).copy()

	# - Draw true bounding box
	print("DEBUG: Draw true bounding box...")
	for i in range(len(masks_gt_merged)):
		label= class_names[class_ids_gt_merged[i]]
		color_gt = class_color_map[label]
	
		y1, x1, y2, x2 = bboxes_gt[i]
		p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,alpha=0.7, linestyle="dashed",edgecolor=color_gt, facecolor='none')
		ax.add_patch(p)

		#caption = captions_gt[i]
		caption = ""
		ax.text(x1, y1 + 8, caption, color='w', size=13, backgroundcolor="none")


	# - Draw detected objects
	print("DEBUG: Draw detected objects...")
	for i in range(len(masks_final)):
		label= class_names[class_ids_final[i]]
		color = class_color_map[label]
		
		# Bounding box
		y1, x1, y2, x2 = bboxes[i]
		p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,alpha=0.7, linestyle="solid",edgecolor=color, facecolor='none')
		ax.add_patch(p)
	
		# Label
		caption = captions[i]
		ax.text(x1, y1 + 8, caption, color=color, size=20, backgroundcolor="none")

		# Mask
		mask= masks_final[i]
		masked_image = visualize.apply_mask(masked_image, mask, color)
	
		# Mask Polygon
		# Pad to ensure proper polygons for masks that touch image edges.
		padded_mask = np.zeros( (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		contours = find_contours(padded_mask, 0.5)
		for verts in contours:
			# Subtract the padding and flip (y, x) to (x, y)
			verts = np.fliplr(verts) - 1
			p = Polygon(verts, facecolor="none", edgecolor=color)
			ax.add_patch(p)

	ax.imshow(masked_image.astype(np.uint8))


	# - Write to file	
	print("DEBUG: Write to file...")
	outfile =  'out_' + image_path_base_noext + '.png'
	t1 = time.time()
	fig.savefig(outfile)
	#fig.savefig(outfile,bbox_inches='tight')
	t2 = time.time()
	print('savefig: %.2fs' % (t2 - t1))
	plt.close(fig)
	#plt.show()
	

	return 0



def test(model):
	""" Test the model on input dataset """    
	
	# - Create dataset class
	dataset = SourceDataset()
	dataset.load_dataset(args.dataset)
	dataset.prepare()

	nimg= 0
	n_classes= config.NUM_CLASSES
	classification_matrix= np.zeros((n_classes,n_classes))
	classification_matrix_norm= np.zeros((n_classes,n_classes))
	nobjs_gt= np.zeros((1,n_classes))

	for index, image_id in enumerate(dataset.image_ids):
		# - Check if stop inspection
		if args.nimg_test>0 and nimg>=args.nimg_test:
			print("INFO: Max number of images to inspect reached, stop here.")
			break

		# - Load image
		image_path = dataset.image_info[index]['path']
		#image = dataset.load_image(image_id)
		image_path_base= os.path.basename(image_path)
		#image_path_base_noext= os.path.splitext(image_path_base)[0]		

		# - Inspect results	
		print("INFO: Inspecting results for image %s ..." % image_path_base)
		#r = model.detect([image], verbose=0)[0]
		#inspect_results(image_path,r,dataset,score_thr=args.scoreThr_test)
		inspect_results(image_id,image_path,model,dataset,score_thr=args.scoreThr_test)

		nimg+= 1

	# - Compute performance results
	for i in range(n_classes):
		norm= nobjs_gt[0][i]
		if norm<=0:
			continue
		for j in range(n_classes):
			C= classification_matrix[i][j]
			C_norm= C/norm
			classification_matrix_norm[i][j]= C_norm

	print("== NOBJ TRUE ==")
	print(nobjs_gt)

	#print("== NOBJ DET ==")
	#print(nobjs_det)

	#print("== NOBJ DET CORRECTLY ==")
	#print(nobjs_det_right)

	print("== CLASSIFICATION MATRIX (not normalized) ==")
	print(classification_matrix)

	print("== CLASSIFICATION MATRIX (normalized) ==")
	print(classification_matrix_norm)

	#print("== PURITY ==")
	#print(purity)



	return 0



def test3(model):
	""" Test the model on input dataset """    
	dataset = SourceDataset()
	dataset.load_dataset(args.dataset)
	dataset.prepare()

	tester= ModelTester(dataset,model,config)	
	tester.score_thr= args.scoreThr_test
	tester.iou_thr= args.iouThr_test
	tester.n_max_img= args.nimg_test

	tester.test()


def test2(model):
	""" Test the model on input dataset """    
	dataset = SourceDataset()
	dataset.load_dataset(args.dataset)
	dataset.prepare()

	nimg= 0

	for index, image_id in enumerate(dataset.image_ids):
		# - Check if stop inspection
		if args.nimg_test>0 and nimg>=args.nimg_test:
			print("INFO: Max number of images to inspect reached, stop here.")
			break

		# - Load image
		image = dataset.load_image(image_id)
		image_path = dataset.image_info[index]['path']
		image_path_base= os.path.basename(image_path)
		image_path_base_noext= os.path.splitext(image_path_base)[0]		

		# - Load mask
		mask_gt= dataset.load_gt_mask(image_id)

		mask_gt_chan3= np.broadcast_to(mask_gt,image.shape)
		image_masked_gt= np.copy(image)
		image_masked_gt[np.where((mask_gt_chan3==[True,True,True]).all(axis=2))]=[255,255,0]

		outfile = 'gtmask_' + image_path_base_noext + '.png'
		skimage.io.imsave(outfile, image_masked_gt)

		# - Extract true bounding box from true mask		
		bboxes_gt= utils.extract_bboxes(mask_gt)

		# Detect objects
		r = model.detect([image], verbose=0)[0]
		mask= r['masks']
		bboxes= r['rois']
		##bboxes= utils.extract_bboxes(mask)
		class_labels= r['class_ids']
		nobjects= mask.shape[-1]
		if nobjects <= 0:
			print("INFO: No object mask found for image %s ..." % image_path_base)
			continue	
		
		# Save image with masks
		outfile =  'out_' + image_path_base_noext + '.png'	
		visualize.display_instances(
			image, 
			r['rois'], 
			r['masks'], 
			r['class_ids'],
			dataset.class_names, 
			r['scores'],
			show_bbox=True, 
			show_mask=True,
			title="Predictions"
		)
		plt.savefig(outfile)

		# Inspect results
		#print("INFO: Inspecting results for image %s ..." % image_path_base)
		#inspect_results(image,r,dataset,score_thr=args.scoreThr_test)

		nimg+= 1

	return 0


############################################################
#  Training
############################################################

if __name__ == '__main__':    
	import argparse

	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect radio sources.')

	parser.add_argument("command",metavar="<command>",help="'train' or 'splash'")
	parser.add_argument('--dataset', required=False,metavar="/path/to/balloon/dataset/",help='Directory of the source dataset')
	parser.add_argument('--weights', required=True,metavar="/path/to/weights.h5",help="Path to weights .h5 file or 'coco'")
	parser.add_argument('--logs', required=False,default=DEFAULT_LOGS_DIR,metavar="/path/to/logs/",help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--image', required=False,metavar="path or URL to image",help='Image to apply the color splash effect on')
	parser.add_argument('--nepochs', required=False,default=10,type=int,metavar="Number of training epochs",help='Number of training epochs')
	parser.add_argument('--epoch_length', required=False,default=10,type=int,metavar="Number of data batches per epoch",help='Number of data batches per epoch')
	parser.add_argument('--weighttype', required=False,default='',metavar="Type of weights",help="Type of weights")
	parser.add_argument('--nthreads', required=False,default=1,type=int,metavar="Number of worker threads",help="Number of worker threads")
	parser.add_argument('--nimg_test', required=False,default=-1,type=int,metavar="Number of images in dataset to inspect during test",help="Number of images in dataset to inspect during test")	
	parser.add_argument('--scoreThr_test', required=False,default=0.7,type=float,metavar="Object detection score threshold to be used during test",help="Object detection score threshold to be used during test")
	parser.add_argument('--iouThr_test', required=False,default=0.6,type=float,metavar="IOU threshold used to match detected objects with true objects",help="IOU threshold used to match detected objects with true objects")
	
	
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
	print("epoch_length: ",args.epoch_length)
	print("nimg_test: ",args.nimg_test)
	print("scoreThr_test: ",args.scoreThr_test)

	# Configurations
	if args.command == "train":
		config = SDetectorConfig()
	else:
		class InferenceConfig(SDetectorConfig):
			# Set batch size to 1 since we'll be running inference on
			# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
			GPU_COUNT = 1
			IMAGES_PER_GPU = 1
		config = InferenceConfig()

	config.STEPS_PER_EPOCH= args.epoch_length

	config.display()

	# Create model
	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs)
	else:
		model = modellib.MaskRCNN(mode="inference", config=config,model_dir=args.logs)

	# Select weights file to load
	weights_path = args.weights

	# Load weights
	print("Loading weights ", weights_path)
	if args.weighttype.lower() == "coco":
		# Exclude the last layers because they require a matching number of classes
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
		#test(model)	
		test3(model)	
	else:
		print("'{}' is not recognized. "
			"Use 'train' or 'test'".format(args.command))



