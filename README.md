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

