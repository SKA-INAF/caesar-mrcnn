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

- OPT1   
- OPT2    
- ...   
- OPTN   


