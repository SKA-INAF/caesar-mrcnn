#!/bin/bash

#######################################
##         CHECK ARGS
#######################################
NARGS="$#"
echo "INFO: NARGS= $NARGS"

if [ "$NARGS" -lt 1 ]; then
	echo "ERROR: Invalid number of arguments...see script usage!"
  echo ""
	echo "**************************"
  echo "***     USAGE          ***"
	echo "**************************"
 	echo "$0 [ARGS]"
	echo ""
	echo "=========================="
	echo "==    ARGUMENT LIST     =="
	echo "=========================="
	
	echo "=== DATA LOADER OPTIONS ==="
	echo "--grayimg - Do not convert input image in RGB (default=no)"
	echo "--classdict=[DICT] - Object class name-id dictionary used for data loading (default={\"sidelobe\":1,\"source\":2,\"galaxy\":3})"
	echo "--classdict_model=[DICT] - Object class name-id dictionary used for model (default={\"sidelobe\":1,\"source\":2,\"galaxy\":3})"
	echo "--remap-classids -Remap class data-model dictionaries (default=no)"	
	echo "--classid-remap-dict=[DICT] - Class data-model remap dictionary (default=empty)"	
	echo "--dataloader=[LOADER] - Type of dataloader to be used {datalist,datalist_json,datadir} (default=empty)"	
	echo "--datalist=[FILENAME] - Filename with data list (default=empty)"	
	echo "--datalist-train=[FILENAME] - Filename with train data list (default=empty)"	
	echo "--datalist-val=[FILENAME] - Filename with cross-validation data list (default=empty)"	
	echo "--datadir=[DIR NAME] - Directory to start file search (default=empty)"	
	echo "--validation-data-fract=[VALUE] - Directory to start file search (default=0.1)"	
	echo "--maxnimgs=[VALUE] - Max number of input images to be read (-1=all) (default=-1)"	
		
	echo ""

	echo "=== NETWORK ARCHITECTURE OPTIONS ==="
	echo "--weights=[PATH] - Path to network weights. If empty use random weights (default=empty)"
	echo "--backbone=[VALUE] - Backbone network {resnet50,resnet101,custom} (default=resnet101)"
	echo "--backbone-strides=[VALUE] - Backbone network stride values (default=4,8,16,32,64)"
	
	echo ""

	echo "=== TRAIN OPTIONS ==="
	echo "--nepochs=[VALUE] - Number of epoch to train (default=1)"	
	echo "--epoch-length=[VALUE] - Number of train steps (equal to train data size if empty) (default=empty)"	
	echo "--nvalidation-steps=[VALUE] - Number of cross-val steps (equal to cross-val data size if empty) (default=empty)"	
	echo "--rpn-anchor-scales=[VALUE] - Value of RPN_ANCHOR_SCALES par (default=4,8,16,32,64)"	
	echo "--max-gt-instances=[VALUE] - Value of MAX_GT_INSTANCES par (default=300)"	
	echo "--rpn-nms-threshold=[VALUE] - Value of RPN_NMS_THRESHOLD par (default=0.7)"	
	echo "--rpn-train-anchors-per-image=[VALUE] - Value of RPN_TRAIN_ANCHORS_PER_IMAGE par (default=512)"	
	echo "--train-rois-per-image=[VALUE] - Value of TRAIN_ROIS_PER_IMAGE par (default=512)"	
	echo "--rpn-anchor-ratios=[VALUE] - Value of RPN_ANCHOR_RATIOS par (default=0.5,1,2)"	
	echo "--rpn-class-loss-weight=[VALUE] - RPN class loss weight factor (default=1)"	
	echo "--rpn-bbox-loss-weight=[VALUE] - RPN bounding box loss weight factor (default=1)"	
	echo "--mrcnn-class-loss-weight=[VALUE] - Classification loss weight factor (default=1)"	
	echo "--mrcnn-bbox-loss-weight=[VALUE] - Bounding box loss weight factor (default=1)"	
	echo "--mrcnn-mask-loss-weight=[VALUE] - Mask loss weight factor (default=1)"	
			
	echo ""
	
	echo "=== TEST/DETECT OPTIONS ==="
	echo "--scoreThr=[VALUE] - Detected object score threshold to select as final object (default=0.7)"		
	echo "--iouThr=[VALUE] - IOU threshold between detected and ground truth bboxes to consider the object as detected (default=0.6)"	
	echo "--image=[VALUE] - Path to input image (.fits) to be given to classifier (default=empty)"		
	
	echo ""

	echo "=== RUN OPTIONS ==="
	echo "--logs=[VALUE] - Path where to store log files (default=current dir)"		
	echo "--nthreads=[VALUE] - Number of threads used for training (default=1)"		
	echo "--ngpu=[VALUE] - Number of GPUs used for training (default=1)"		
	echo "--nimg-per-gpu=[VALUE] - Number of images assigned to each GPUs during training (default=1)"		
	echo "--runmode=[VALUE] - Program run mode {train,test,detect} (default=empty)"		
	
	echo "=========================="
  exit 1
fi

#######################################
##         PARSE ARGS
#######################################
export JOB_DIR="$PWD"
export OUTPUT_DIR="$PWD"

WAIT_COPY=false
COPY_WAIT_TIME=30

RUNMODE=""
GRAYIMG_OPTION=""
UINT8_OPTION=""
ZSCALE_OPTION=""
CLASS_DICT_MODEL="{\"sidelobe\":1,\"source\":2,\"galaxy\":3}"
CLASS_DICT="{\"sidelobe\":1,\"source\":2,\"galaxy\":3}"
CLASS_DICT_OPTION=""
CLASS_DICT_MODEL_OPTION=""
REMAP_CLASSIDS_OPTION=""
CLASSID_REMAP_DICT=""
CLASSID_REMAP_DICT_OPTION=""
DATALOADER="datalist"
DATALOADER_OPTION=""
DATALIST=""
DATALIST_OPTION=""
DATALIST_TRAIN=""
DATALIST_TRAIN_OPTION=""
DATALIST_VAL=""
DATALIST_VAL_OPTION=""
DATADIR=""
DATADIR_OPTION=""
VAL_DATA_FRACT=0.1
VAL_DATA_FRACT_OPTION=""
MAXNIMGS=-1
MAXNIMGS_OPTION=""
WEIGHTS=""
WEIGHTS_OPTION=""
LOGDIR="logs/"
LOGDIR_OPTION=""
NTHREADS=1
NGPU=1
IMG_PER_GPU=1
NEPOCHS=1
NEPOCHS_OPTION=""
EPOCH_LENGTH_OPTION=""
NVAL_STEPS_OPTION=""
RPN_ANCHOR_SCALES="4,8,16,32,64"
RPN_ANCHOR_SCALES_OPTION=""
MAX_GT_INSTANCES=300
MAX_GT_INSTANCES_OPTION=""
BACKBONE="resnet101"
BACKBONE_OPTION=""
BACKBONE_STRIDES="4,8,16,32,64"
BACKBONE_STRIDES_OPTION=""
RPN_NMS_THRESHOLD=0.7
RPN_NMS_THRESHOLD_OPTION=""
RPN_TRAIN_ANCHORS_PER_IMAGE=512
RPN_TRAIN_ANCHORS_PER_IMAGE_OPTION=""
RPN_TRAIN_ROIS_PER_IMAGE=512
RPN_TRAIN_ROIS_PER_IMAGE_OPTION=""
RPN_ANCHOR_RATIOS="0.5,1,2"
RPN_ANCHOR_RATIOS_OPTION=""
RPN_CLASS_LOSS_WEIGHT=1
RPN_CLASS_LOSS_WEIGHT_OPTION=""
RPN_BBOX_LOSS_WEIGHT=1
RPN_BBOX_LOSS_WEIGHT_OPTION=""
MRCNN_CLASS_LOSS_WEIGHT=1
MRCNN_CLASS_LOSS_WEIGHT_OPTION=""
MRCNN_BBOX_LOSS_WEIGHT=1
MRCNN_BBOX_LOSS_WEIGHT_OPTION=""
MRCNN_MASK_LOSS_WEIGHT=1
MRCNN_MASK_LOSS_WEIGHT_OPTION=""
SCORE_THR=0.7
SCORE_THR_OPTION=""
IOU_THR=0.6
IOU_THR_OPTION=""
IMAGE=""
IMAGE_OPTION=""
for item in "$@"
do
	case $item in 
		
		# - DATA LOADERS ##	
		--grayimg*)
    	GRAYIMG_OPTION="--grayimg"
    ;;
		--no-uint8*)
    	UINT8_OPTION="--no_uint8"
    ;;
		--zscale-contrasts=*)
    	ZSCALE=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			ZSCALE_OPTION="--zscale_contrasts=$ZSCALE"
    ;;
		--classdict=*)
    	CLASS_DICT=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			CLASS_DICT_OPTION="--classdict=$CLASS_DICT"
    ;;
		--classdict-model=*)
    	CLASS_DICT_MODEL=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			CLASS_DICT_MODEL_OPTION="--classdict_model=$CLASS_DICT_MODEL"
    ;;
		--remap-classids*)
    	REMAP_CLASSIDS_OPTION="--remap_classids"
    ;;
		--classid-remap-dict=*)
    	CLASSID_REMAP_DICT=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			CLASSID_REMAP_DICT_OPTION="--classid_remap_dict=$CLASSID_REMAP_DICT"
    ;;
		--dataloader=*)
    	DATALOADER=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			DATALOADER_OPTION="--dataloader=$DATALOADER"
    ;;
		--datalist=*)
    	DATALIST=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			DATALIST_OPTION="--datalist=$DATALIST"
    ;;
		--datalist-train=*)
    	DATALIST_TRAIN=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			DATALIST_TRAIN_OPTION="--datalist_train=$DATALIST_TRAIN"
    ;;
		--datalist-val=*)
    	DATALIST_VAL=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			DATALIST_VAL_OPTION="--datalist_val=$DATALIST_VAL"
    ;;
		--datadir=*)
    	DATADIR=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			DATADIR_OPTION="--datadir=$DATADIR"
    ;;
		--validation-data-fract=*)
    	VAL_DATA_FRACT=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			VAL_DATA_FRACT_OPTION="--validation_data_fract=$VAL_DATA_FRACT"
    ;;
		--maxnimgs=*)
    	MAXNIMGS=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			MAXNIMGS_OPTION="--maxnimgs=$MAXNIMGS"
    ;;
	
		# - NETWORK ARCHITECTURE
		--weights=*)
    	WEIGHTS=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			WEIGHTS_OPTION="--weights=$WEIGHTS"
    ;;
		--backbone=*)
    	BACKBONE=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			BACKBONE_OPTION="--backbone=$BACKBONE"
    ;;
		--backbone-strides=*)
    	BACKBONE_STRIDES=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			BACKBONE_STRIDES_OPTION="--backbone_strides=$BACKBONE_STRIDES"
    ;;

		# - TRAIN OPTIONS
		--nepochs=*)
    	NEPOCHS=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			NEPOCHS_OPTION="--nepochs=$NEPOCHS"
    ;;
		--epoch-length=*)
    	EPOCH_LENGTH=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			EPOCH_LENGTH_OPTION="--epoch_length=$EPOCH_LENGTH"
    ;;
		--nvalidation-steps=*)
    	NVAL_STEPS=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			NVAL_STEPS_OPTION="--nvalidation_steps=$NVAL_STEPS"
    ;;
		--rpn-anchor-scales=*)
    	RPN_ANCHOR_SCALES=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			RPN_ANCHOR_SCALES_OPTION="--rpn_anchor_scales=$RPN_ANCHOR_SCALES"
    ;;	
		--max-gt-instances=*)
    	MAX_GT_INSTANCES=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			MAX_GT_INSTANCES_OPTION="--max_gt_instances=$MAX_GT_INSTANCES"
    ;;
		
		--rpn-nms-threshold=*)
    	RPN_NMS_THRESHOLD=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			RPN_NMS_THRESHOLD_OPTION="--rpn_nms_threshold=$RPN_NMS_THRESHOLD"
    ;;
		--rpn-train-anchors-per-image=*)
    	RPN_TRAIN_ANCHORS_PER_IMAGE=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			RPN_TRAIN_ANCHORS_PER_IMAGE_OPTION="--rpn_train_anchors_per_image=$RPN_TRAIN_ANCHORS_PER_IMAGE"
    ;;

		--train-rois-per-image=*)
    	RPN_TRAIN_ROIS_PER_IMAGE=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			RPN_TRAIN_ROIS_PER_IMAGE_OPTION="--train_rois_per_image=$RPN_TRAIN_ROIS_PER_IMAGE"
    ;;

		--rpn-anchor-ratios=*)
    	RPN_ANCHOR_RATIOS=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			RPN_ANCHOR_RATIOS_OPTION="--rpn_anchor_ratios=$RPN_ANCHOR_RATIOS"
    ;;

		--rpn-class-loss-weight=*)
    	RPN_CLASS_LOSS_WEIGHT=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`	
			RPN_CLASS_LOSS_WEIGHT_OPTION="--rpn_class_loss_weight=$RPN_CLASS_LOSS_WEIGHT"
    ;;
		--rpn-bbox-loss-weight=*)
    	RPN_BBOX_LOSS_WEIGHT=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			RPN_BBOX_LOSS_WEIGHT_OPTION="--rpn_bbox_loss_weight=$RPN_BBOX_LOSS_WEIGHT"
    ;;
		--mrcnn-class-loss-weight=*)
    	MRCNN_CLASS_LOSS_WEIGHT=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			MRCNN_CLASS_LOSS_WEIGHT_OPTION="--mrcnn_class_loss_weight=$MRCNN_CLASS_LOSS_WEIGHT"
    ;;
		--mrcnn-bbox-loss-weight=*)
    	MRCNN_BBOX_LOSS_WEIGHT=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			MRCNN_BBOX_LOSS_WEIGHT_OPTION="--mrcnn_bbox_loss_weight=$MRCNN_BBOX_LOSS_WEIGHT"
    ;;
		--mrcnn-mask-loss-weight=*)
    	MRCNN_MASK_LOSS_WEIGHT=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			MRCNN_MASK_LOSS_WEIGHT_OPTION="--mrcnn_mask_loss_weight=$MRCNN_MASK_LOSS_WEIGHT"
    ;;
	
		# - TEST OPTIONS
		--scoreThr=*)
    	SCORE_THR=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			SCORE_THR_OPTION="--scoreThr=$SCORE_THR"
    ;;
		--iouThr=*)
    	IOU_THR=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			IOU_THR_OPTION="--iouThr=$IOU_THR"
    ;;

		# - DETECT OPTIONS
		--image=*)
    	IMAGE=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			IMAGE_OPTION="--image=$IMAGE"
    ;;

		# - RUN OPTIONS
		--logs=*)
    	LOG_DIR=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			LOGDIR_OPTION="--logs=$LOG_DIR"
    ;;
		--nthreads=*)
    	NTHREADS=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			NTHREADS_OPTION="--nthreads=$NTHREADS"
    ;;
		--ngpu=*)
    	NGPU=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			NGPU_OPTION="--ngpu=$NGPU"
    ;;
		--nimg-per-gpu=*)
    	NIMG_PER_GPU=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			NIMG_PER_GPU_OPTION="nimg_per_gpu=$NIMG_PER_GPU"
    ;;
		--runmode=*)
    	RUNMODE=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
    ;;
		--outdir=*)
    	OUTPUT_DIR=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
    ;;
		--waitcopy*)
    	WAIT_COPY=true
    ;;
		--copywaittime=*)
    	COPY_WAIT_TIME=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
    ;;
	
		--jobdir=*)
    	JOB_DIR=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
    ;;

    *)
    # Unknown option
    echo "ERROR: Unknown option ($item)...exit!"
    exit 1
    ;;
	esac
done

if [ "$JOB_DIR" = "" ]; then
  echo "WARN: Empty JOB_DIR given, setting it to pwd ($PWD) ..."
	JOB_DIR="$PWD"
fi

if [ "$OUTPUT_DIR" = "" ]; then
  echo "WARN: Empty OUTPUT_DIR given, setting it to pwd ($PWD) ..."
	OUTPUT_DIR="$PWD"
fi


###########################
##    RUN
###########################
# - Set exec options
EXE_ARGS=""
EXE_ARGS="$EXE_ARGS $GRAYIMG_OPTION "
EXE_ARGS="$EXE_ARGS $UINT8_OPTION "
EXE_ARGS="$EXE_ARGS $ZSCALE_OPTION "
EXE_ARGS="$EXE_ARGS $CLASS_DICT_OPTION "
EXE_ARGS="$EXE_ARGS $CLASS_DICT_MODEL_OPTION "
EXE_ARGS="$EXE_ARGS $REMAP_CLASSIDS_OPTION "
EXE_ARGS="$EXE_ARGS $CLASSID_REMAP_DICT_OPTION "
EXE_ARGS="$EXE_ARGS $DATALOADER_OPTION "
EXE_ARGS="$EXE_ARGS $DATALIST_OPTION "
EXE_ARGS="$EXE_ARGS $DATALIST_TRAIN_OPTION "
EXE_ARGS="$EXE_ARGS $DATALIST_VAL_OPTION "
EXE_ARGS="$EXE_ARGS $DATADIR_OPTION "
EXE_ARGS="$EXE_ARGS $VAL_DATA_FRACT_OPTION "
EXE_ARGS="$EXE_ARGS $MAXNIMGS_OPTION "
EXE_ARGS="$EXE_ARGS $WEIGHTS_OPTION "
EXE_ARGS="$EXE_ARGS $BACKBONE_OPTION "
EXE_ARGS="$EXE_ARGS $BACKBONE_STRIDES_OPTION "
EXE_ARGS="$EXE_ARGS $NEPOCHS_OPTION "
EXE_ARGS="$EXE_ARGS $EPOCH_LENGTH_OPTION "
EXE_ARGS="$EXE_ARGS $NVAL_STEPS_OPTION "

EXE_ARGS="$EXE_ARGS $RPN_ANCHOR_SCALES_OPTION "
EXE_ARGS="$EXE_ARGS $MAX_GT_INSTANCES_OPTION "
EXE_ARGS="$EXE_ARGS $RPN_NMS_THRESHOLD_OPTION "
EXE_ARGS="$EXE_ARGS $RPN_TRAIN_ANCHORS_PER_IMAGE_OPTION "
EXE_ARGS="$EXE_ARGS $RPN_TRAIN_ROIS_PER_IMAGE_OPTION "
EXE_ARGS="$EXE_ARGS $RPN_ANCHOR_RATIOS_OPTION "

EXE_ARGS="$EXE_ARGS $RPN_CLASS_LOSS_WEIGHT_OPTION "
EXE_ARGS="$EXE_ARGS $RPN_BBOX_LOSS_WEIGHT_OPTION "
EXE_ARGS="$EXE_ARGS $MRCNN_CLASS_LOSS_WEIGHT_OPTION "
EXE_ARGS="$EXE_ARGS $MRCNN_BBOX_LOSS_WEIGHT_OPTION "
EXE_ARGS="$EXE_ARGS $MRCNN_MASK_LOSS_WEIGHT_OPTION "
EXE_ARGS="$EXE_ARGS $SCORE_THR_OPTION "
EXE_ARGS="$EXE_ARGS $IOU_THR_OPTION "
EXE_ARGS="$EXE_ARGS $IMAGE_OPTION "

EXE_ARGS="$EXE_ARGS $LOGDIR_OPTION "
EXE_ARGS="$EXE_ARGS $NTHREADS_OPTION "
EXE_ARGS="$EXE_ARGS $NGPU_OPTION "
EXE_ARGS="$EXE_ARGS $NIMG_PER_GPU_OPTION "
EXE_ARGS="$EXE_ARGS $RUNMODE "

# - Check if job directory exists
if [ ! -d "$JOB_DIR" ] ; then 
  echo "INFO: Job dir $JOB_DIR not existing, creating it now ..."
	mkdir -p "$JOB_DIR" 
fi

# - Moving to job directory
echo "INFO: Moving to job directory $JOB_DIR ..."
cd $JOB_DIR

# - Run command
echo "INFO: Running mask-rcnn with args: $EXE_ARGS"
python3 $MASKRCNN_DIR/bin/run.py $EXE_ARGS

STATUS=$?
if [ $STATUS -ne 0 ];
then
	echo "ERROR: mask-rcnn run failed with status $STATUS!"
	exit $STATUS
fi

# - Copy output data to output directory
if [ "$JOB_DIR" != "$OUTPUT_DIR" ]; then
	echo "INFO: Copying job outputs in $OUTPUT_DIR ..."
	ls -ltr $JOB_DIR

	# - Copy output plot(s)
	png_count=`ls -1 *.png 2>/dev/null | wc -l`
  if [ $png_count != 0 ] ; then
		echo "INFO: Copying output plot file(s) to $OUTPUT_DIR"
		cp *.png $OUTPUT_DIR
	fi

	# - Copy output jsons
	json_count=`ls -1 *.json 2>/dev/null | wc -l`
	if [ $json_count != 0 ] ; then
		echo "INFO: Copying output json file(s) to $OUTPUT_DIR"
		cp *.json $OUTPUT_DIR
	fi
        
	# - Show output directory
	echo "INFO: Show files in $OUTPUT_DIR ..."
	ls -ltr $OUTPUT_DIR

	# - Wait a bit after copying data
	#   NB: Needed if using rclone inside a container, otherwise nothing is copied
	if [ $WAIT_COPY = true ]; then
		echo "INFO: Sleeping $COPY_WAIT_TIME seconds to allow out file copy ..."
		sleep $COPY_WAIT_TIME
	fi

fi


