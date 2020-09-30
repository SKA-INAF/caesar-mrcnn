# Import standard modules
import os
import sys
import json
import time
import datetime
import logging
import numpy as np

## Get logger
logger = logging.getLogger(__name__)


## Import ROOT 
#try:
#	import ROOT
#	from ROOT import gSystem, gROOT, AddressOf, TTree, TFile
#except:
#	logger.error("Cannot load ROOT modules (hint: check LD_LIBRARY_PATH)!")
#	exit

# Import CAESAR library
#try:
#	gSystem.Load('libCaesar.so')
#	from ROOT import Caesar
#	from ROOT.Caesar import Image, Source
#except:
#	logger.error("Cannot load Caesar modules (hint: check LD_LIBRARY_PATH)!")
#	exit


# Import Mask RCNN
#from mrcnn.config import Config
#from mrcnn import model as modellib, utils
from mrcnn.analyze import Analyzer
from mrcnn import utils

# ====================================
# ==    SOURCE CLASSIFICATION INFO
# ====================================
class SClassInfo(object):
	""" Define a source classification info """

	def __init__(self):
		""" Return a source classification info """
	
		self.class_id= -1
		self.class_name= ''
		self.score= 0
		self.snames= [] # List of sources associated

# ===========================
# ==    SOURCE DATA
# ===========================
class SData(object):
	""" Define source data """
	
	def __init__(self):
		""" Return a source data object """
	
		# - Source info
		self.name= ''
		self.x0= -1
		self.y0= -1
		self.xmin= -1
		self.xmax= -1
		self.ymin= -1
		self.ymax= -1
		self.visited= False

		# - Source classification info
		self.class_info= []

	def add_class_info(self,c):
		""" Add class info to list """
		self.class_info.append(c)


# ===========================
# ==    SOURCE CLASSIFIER
# ===========================
class SClassifier(object):
	""" Define source classifier object """

	def __init__(self,model,config):
		""" Return a source classifier object """

		# - Model
		self.model= model
		self.r= None

		# - Config options
		self.config= config
		self.n_classes= self.config.NUM_CLASSES
		self.class_names= self.config.CLASS_NAMES

		# - Image data	
		self.image_path= ''
		self.img_data= None
		self.img_header= None
		self.nx= -1
		self.ny= -1
			
		# - Source catalog data
		self.scatalog_path= ''
		self.sources= []
		self.n_max_sources= -1
		self.scutout_size= 132

		# - Classifier options
		self.iou_thr= 0.6
		self.score_thr= 0.7
		
	# ===========================
	# ==     RUN CLASSIFIER
	# ===========================
	def run(self,image_path,scatalog_path):
		""" Predict classification for sources listed in the given Caesar source catalog file """

		# - Set variables
		self.image_path= image_path
		self.scatalog_path= scatalog_path

		# - Read input image
		logger.info("Reading input image %s ..." % self.image_path)
		if self.read_img()<0:
			logger.error("Failed to read image %s!" % self.image_path)
			return -1

		# - Read source catalog
		logger.info("Reading input source catalog %s ..." % self.scatalog_path)
		if(self.read_scatalog()<0):
			logger.error("Failed to read source catalog %s!" % self.scatalog_path)
			return -1


		# - Loop over sources
		for i in range(len(self.sources)):
		
			# - Check if stop inspection
			if self.n_max_sources>0 and i>=self.n_max_sources:
				print("INFO: Max number of sources to be processed reached, stop here.")
				break

			# - Classify source
			sname= self.sources[i].name
			logger.info("Classifying source %s ..." % (sname))
			status= self.classify_source(i)
			if status<0:
				logger.warn("Failed to run source classification on source %s!" % (sname))
				continue

		# - Process classification info
		# ...

		return 0


	# ===========================
	# ==     DETECT SOURCE
	# ===========================


	# ===========================
	# ==     CLASSIFY SOURCE
	# ===========================
	def classify_source(self,sindex):	
		""" Classify a source """

		# - Get source pos & bbox
		sname= self.sources[sindex].name
		x0_s= self.sources[sindex].x0
		y0_s= self.sources[sindex].y0
		xmin_s= self.sources[sindex].xmin
		xmax_s= self.sources[sindex].xmax
		ymin_s= self.sources[sindex].ymin	
		ymax_s= self.sources[sindex].ymax		
		dx_s= xmax_s - xmin_s
		dy_s= ymax_s - ymin_s

		# - Check if already visited
		if self.sources[sindex].visited:
			logger.info("Source %s already visited, nothing to be done ..." % sname)
			return 0


		# - Check cutout size wrt source size
		dx= self.scutout_size
		dy= self.scutout_size
		bbox_cut= False
		if dx<=dx_s:
			logger.warn("Cutout size (%d pix) is <= source %d size (%d pix) along x dimension!" % (dx,sname,dx_s))
			bbox_cut= True
		if dy<=dy_s:
			logger.warn("Cutout size (%d pix) is <= source %d size (%d pix) along y dimension!" % (dy,sname,dy_s))
			bbox_cut= True

		# - Compute source bounding box in cutout coordinates
		xmin= int(x0_s-dx/2)
		ymin= int(y0_s-dy/2)
		xmax= int(x0_s+dx/2)
		ymax= int(y0_s+dy/2)

		xmin_s_local= xmin_s - xmin
		xmax_s_local= xmax_s - xmin
		ymin_s_local= ymin_s - ymin
		ymax_s_local= ymax_s - ymin
		bbox_s= [ymin_s_local,xmin_s_local,ymax_s_local,xmax_s_local]
		logger.info("Source bbox [%s,%s,%s,%s], bbox_local [%s,%s,%s,%s]" % (str(xmin_s),str(xmax_s),str(ymin_s),str(ymax_s), str(xmin_s_local),str(xmax_s_local),str(ymin_s_local),str(ymax_s_local) ))

		# - Extract cutout image around given source
		stretch= True
		normalize= True
		convertToRGB= True

		data_crop= utils.crop_img(
			self.img_data,
			x0_s,y0_s,
			dx,dy,
			stretch,normalize,convertToRGB
		)
		logger.info("Crop image for source %s has size: %d x %d ..." % (sname,data_crop.shape[1],data_crop.shape[0]))
		print(data_crop.shape)

		# - Find other sources located in the same cutout
		bboxes_s= [bbox_s]
		is_bbox_cut= [bbox_cut]
		indices_s= [sindex]
		
		for j in range(len(self.sources)):
			sname_j= self.sources[j].name
			x0_j= self.sources[j].x0
			y0_j= self.sources[j].y0
			xmin_j= self.sources[j].xmin
			xmax_j= self.sources[j].xmax
			ymin_j= self.sources[j].ymin	
			ymax_j= self.sources[j].ymax
			if sname==sname_j:
				continue
			is_inside_x= (x0_j>xmin and x0_j<xmax)
			is_inside_y= (y0_j>ymin and y0_j<ymax)
			is_inside= (is_inside_x and is_inside_y)
			if not is_inside:		
				continue

			indices_s.append(j)
			
			# - Append bbox 
			xmin_j_local= xmin_j - xmin
			xmax_j_local= xmax_j - xmin
			ymin_j_local= ymin_j - ymin
			ymax_j_local= ymax_j - ymin
			bbox_j= [ymin_j_local,xmin_j_local,ymax_j_local,xmax_j_local]
			bboxes_s.append(bbox_j)

			# - Check if source bbox is cut
			is_cut_x= (xmin_j<=xmin or xmax_j>=xmax)
			is_cut_y= (ymin_j<=ymin or ymax_j>=ymax)
			is_cut= (is_cut_x or is_cut_y)
			is_bbox_cut.append(is_cut)

			
		# - Detect object on source image cutout
		analyzer= Analyzer(self.model,self.config)
		analyzer.iou_thr= self.iou_thr
		analyzer.score_thr= self.score_thr

		if analyzer.predict(data_crop,sname,bboxes_s)<0:
			logger.error("Failed to run model prediction on source %s!" % sname)
			return -1

		bboxes_det= analyzer.bboxes
		scores_det= analyzer.scores_final	
		classid_det= analyzer.class_ids_final

		# - Return if no object was detected
		if not bboxes_det:
			logger.warn("No object detected for source cutout %s, this source won't be classified..." % sname)	

			# - Mark all sources with bounding box not cut as visited
			for j in range(len(indices_s)):
				index= indices_s[j]
				bbox_cut= is_bbox_cut[j] 
				if not bbox_cut:
					self.sources[index].visited= True		

			return 0


		# - Process detected objects and match with source according to IOU
		association_map= {}
		det_indices= []

		for j in range(len(bboxes_s)):
			index= indices_s[j]
			sname_s= self.sources[index].name
			bbox_cut= is_bbox_cut[j] 
			bbox_s= bboxes_s[j]
			xmin_s= bbox_j[1]
			xmax_s= bbox_j[3]
			ymin_s= bbox_j[0]
			ymax_s= bbox_j[2]

			logger.info("Find if source %s (index=%d) is associated to any of the %d objects detected..." % (sname_s,index,len(bboxes_det)))
			index_best= -1
			iou_best= 0

			for i in range(len(bboxes_det)):
				bbox_det= bboxes_det[i]

				iou= utils.get_iou(bbox_det,bbox_s)
				xmin_o= bbox_det[1]
				xmax_o= bbox_det[3]
				ymin_o= bbox_det[0]
				ymax_o= bbox_det[2]
 
				logger.info("Det bbox no. %d [%s,%s,%s,%s], Source bbox[%s,%s,%s,%s]: IOU=%f" % (i+1,str(xmin_o),str(xmax_o),str(ymin_o),str(ymax_o),str(xmin_s),str(xmax_s),str(ymin_s),str(ymax_s),iou))
				if iou>self.iou_thr and iou>=iou_best:
					index_best= i
					iou_best= iou

			det_indices.append(index_best)

			if index_best!=-1:
				if not index_best in association_map:
					association_map[index_best]= []
				association_map[index_best].append(index)
				

		# - Add classification info
		for j in range(len(bboxes_s)):	
			index= indices_s[j]
			sname_s= self.sources[index].name
			bbox_cut= is_bbox_cut[j] 
			det_index= det_indices[j]

			# - Mark as visited if bbox is not cut
			if not bbox_cut:
				self.sources[index].visited= True

			# - Fill class data
			if det_index==-1:
				logger.info("Source %s was not associated to any detected object and so it won't be classified." % (sname_s))
				continue
			else:
				score= scores_det[det_index]
				class_id= classid_det[det_index]	
				class_name= self.class_names[class_id]
		
				snames= []
				indices_ass= association_map[det_index]
				print(indices_ass)
				for index_ass in indices_ass:
					sname_ass= self.sources[index_ass].name			
					#print("sname=%s, sname_ass=%s" % (sname_s,sname_ass))	
					if sname_ass!=sname_s:
						snames.append(sname_ass)

				logger.info("Source %s associated to class %s (class_id=%d) with score=%f ..." % (sname_s,class_name,class_id,score))
				#print("--> Associated sources ")
				#print(snames)

				c= SClassInfo()	
				c.class_id= class_id
				c.class_name= class_name
				c.snames= snames
				self.sources[index].add_class_info(c)

		
			
		# ...
		# ...		


		return 0


	## =========================
	## ==    READ IMAGE
	## =========================
	def read_img(self):
		""" Read input FITS image """
	
		res= utils.read_fits(
			self.image_path,
			stretch=False,
			normalize=False,
			convertToRGB=False
		)
	
		if not res:
			logger.error("Failed to read image %s!" % self.image_path)
			return -1

		self.img_data= res[0]
		self.img_header= res[1]
		self.nx= self.img_data.shape[1]
		self.ny= self.img_data.shape[0]

		logger.info("Input image %s has size %d x %d..." % (self.image_path,self.nx,self.ny))

		return 0

	## ============================
	## ==    READ SOURCE CATALOG
	## ============================
	def read_scatalog(self):
		""" Read source catalog """	

		# - Read table
		t= utils.read_table(self.scatalog_path)
		if not t:
			logger.error("Failed to read table!")
			return -1

		# - Loop over table and fill source data
		for item in t:
			 
			sdata= SData()
			sdata.name= item[0]
			sdata.x0= item[5]
			sdata.y0= item[6]
			sdata.xmin= item[13]
			sdata.xmax= item[14]
			sdata.ymin=	item[15]
			sdata.ymax= item[16]
			self.sources.append(sdata)

		logger.info("Read #%d sources from file %s ..." % (len(self.sources),self.scatalog_path))

		return 0

	#def read_scatalog(self):
	#	""" Read source catalog """
	#
	#	# - Open file
	#	f= ROOT.TFile(self.scatalog_path,"READ")
	#	if not f:
	#		logger.error("Cannot open file %s!" % self.scatalog_path)
	#		return -1
	#
	#	# - Read TTree
	#	t= f.Get("SourceInfo")
	#	if not t:
	#		logger.error("Failed to retrieve source tree from file %s!" % self.scatalog_path)
	#		return -1
	#
	#	# - Read source collection
	#	source= Caesar.Source()
	#	t.SetBranchAddress("Source",AddressOf(source))
	#	self.sources= []
	#	for i in range(t.GetEntries()):
	#		t.GetEntry(i)
	#		s= Caesar.Source(source)
	#		self.sources.append(s)
	#
	#	logger.info("Read #%d sources from file %s ..." % (len(self.sources),self.scatalog_path))
	#
	#	# - Check sources names
	#	#for s in self.sources:
	#	#	name= str(s.GetName())
	#	#	print("--> Source %s ..." % (name))
	#
	#	return 0

