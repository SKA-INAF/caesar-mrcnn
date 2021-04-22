# mrcnn
Radioastronomical object detector tool based on Mask R-CNN instance segmentation framework.

## **Status**
This software is under development and supported only on python 3 + tensorflow 1x. 

## **Credit**
This software is distributed with GPLv3 license. If you use it for your research, please add repository link or acknowledge authors in your papers.   

## **Installation**  

To build and install the package:    

* Create a local install directory, e.g. ```$INSTALL_DIR```
* Add installation path to your ```PYTHONPATH``` environment variable:   
  ``` export PYTHONPATH=$PYTHONPATH:$INSTALL_DIR/lib/python3.6/site-packages ```
* Build and install package:   
  ``` python3.6 setup.py sdist bdist_wheel```    
  ``` python3.6 setup build```   
  ``` python3.6 setup install --prefix=$INSTALL_DIR```   

All dependencies will be automatically downloaded and installed in ```$INSTALL_DIR```.   
     
To use package scripts:

* Add binary directory to your ```PATH``` environment variable:   
  ``` export PATH=$PATH:$INSTALL_DIR/bin ```    

## **Usage**  

mrcnn can be run in different modes:   

* To train a model: ```python3.6 $INSTALL_DIR/bin/run.py [OPTIONS] train```      
* To test a model: ```python3.6 $INSTALL_DIR/bin/run.py [OPTIONS] test```    
* To detect objects on new data: ```python3.6 $INSTALL_DIR/bin/run.py [OPTIONS] detect```    
* To classify pre-detected objects: ```python3.6 $INSTALL_DIR/bin/run.py [OPTIONS] classify```  (TO BE IMPLEMENTED)    

Supported options are:  

*IMAGE PRE-PROCESSING*     
* `--grayimg`: To pass 1-channel gray-level images to input layer. Default: pass 3-channels RGB images    
* `--no_uint8`: To disable conversion from image float values to uint8. Default: convert to uint8       
* `--no_zscale`: To disable astropy zscale transform applied to input images. Default: apply   
* `--zscale_contrasts=[CONTRASTS]`: zscale contrast values in range [0,1] for the three RGB channels, in string format separated by commas. Default: 0.25,0.25,0.25    
* `--biascontrast`: To apply astropy BiasContrast transform to input images. Default: not applied    
* `--bias=[BIAS]`: Bias parameter value of BiasContrast transform. Default: 0.5   
* `--contrast=[CONTRAST]`: Contrast parameter value of BiasContrast transform. Default: 1   

*DATA LOADER*    

* `--dataloader=[LOADER]`: Train/cross-val data loader type. Valid values: {datalist,datalist_json,datadir_json}. Default: datalist    
* `--datalist=[FILENAME]`: Data filelist with format: `filename_img(.fits),filename_mask(.fits),label` (datalist loader) or `filename(.json)` (datalist_json loader)    
* `--datalist_train=[FILENAME]`: Train data filelist with format: `filename_img(.fits),filename_mask(.fits),label` (datalist loader) or `filename(.json)` (datalist_json loader)    
* `--datalist_val=[FILENAME]`: Cross-validation data filelist with format: `filename_img(.fits),filename_mask(.fits),label` (datalist loader) or `filename(.json)` (datalist_json loader)   
* `--datadir=[PATH]`: Data top directory traversed recursively to search for json dataset files  
* `--validation_data_fract=[FRACTION]`: Fraction of input data used for cross-validation. Default: 0.1    
* `--maxnimgs=[VALUE]`: Max number of images to consider in dataset (-1=all). Default: -1    
* `--classdict=[DICT]`: Class id dictionary in string format used when loading dataset. Default: `{"sidelobe":1,"source":2,"galaxy":3}`   
* `--classdict_model=[DICT]`: Class id dictionary used for the model (if empty, it is set equal to classdict). Default: empty    
* `--remap_classids`: Remap detected object class ids to ground truth object class ids. Default: False   
* `--classid_remap_dict=[DICT]`: Dictionary in string format used to remap detected object class ids to ground truth object class ids. Default: empty   

*MODEL OPTIONS*   

* `--weights=[FILENAME]`: Path to weights .h5 file. If empty, initialize model with random weights. Default: empty    

*RUN OPTIONS*  

* `--logs=[PATH]`: Logs and checkpoints directory. Default: `logs/`   
* `--nthreads=[N]`: Number of worker threads. Default: 1    

*TRAIN OPTIONS*    
     
* `--ngpu=[N]`: Number of GPUs. Default: 1   
* `--nimg_per_gpu=[N]`: Number of images per gpu. Default: 1   
* `--nepochs=[N]`: Number of training epochs. Default: 1   

*TEST OPTIONS*

* `--scoreThr=[SCORE_THRESHOLD]`: Score Threshold to use during evaluation. Default: 0.7 
* `--iouThr=[IOU_THRESHOLD]`: IoU Threshold to use during evaluation. Default: 0.6
* `--consider_sources_near_mixed_sidelobes` or `--no_consider_sources_near_mixed_sidelobes`: Whether to consider
  sources that are flagged to be near or mixed with sidelobes during evaluation. Default: Considered
  