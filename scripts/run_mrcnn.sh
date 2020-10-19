#!/bin/bash


## SET PYTHON ENV
#unset PYTHONPATH
#export PYTHONPATH=$PYTHONPATH:$MASKRCNN_DIR/lib/python3.6/site-packages/
#export PATH=$PATH:$MASKRCNN_DIR/bin:$PRMON_DIR/bin

echo "PYTHONPATH=$PYTHONPATH"

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
	echo "*** MANDATORY ARGS ***"
	echo "--filelist=[FILELIST] - Ascii file with list of image files (.fits/.root, full path) to be processed" 
	echo "--inputfile=[FILENAME] - Input file name to be searched (.fits/.root). If the --filelist option is given this option is skipped."
	echo ""

	echo "*** OPTIONAL ARGS ***"
	echo "=== SFINDER OUTPUT OPTIONS ==="
	echo "--save-inputmap - Save input map in output ROOT file (default=no)"
	echo "--save-bkgmap - Save bkg map in output ROOT file (default=no)"
		
	echo "=========================="
  exit 1
fi

#######################################
##         PARSE ARGS
#######################################
RUNMODE=""
GRAYIMG_OPTION=""
CLASS_DICT_MODEL="{\"sidelobe\":1,\"source\":2,\"galaxy\":3}"
CLASS_DICT="{\"sidelobe\":1,\"source\":2,\"galaxy\":3}"
CLASS_DICT_OPTION=""
CLASS_DICT_MODEL_OPTION=""
REMAP_CLASSIDS_OPTION=""
CLASSID_REMAP_DICT=""
CLASSID_REMAP_DICT_OPTION=""
DATALOADER="filelist"
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
		--classdict=*)
    	CLASS_DICT=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			CLASS_DICT_OPTION="--classdict=$CLASS_DICT"
    ;;
		--classdict-model=*)
    	CLASS_DICT_MODEL=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			CLASS_DICT_MODEL_OPTION="--classdict_model=$CLASS_DICT_MODEL"
    ;;
		--remap-classids*)
    	REMAP_CLASSIDS_OPTION="--remap_classids"
    ;;
		--classid-remap-dict=*)
    	CLASSID_REMAP_DICT=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			CLASSID_REMAP_DICT_OPTION="--classid_remap_dict=$CLASSID_REMAP_DICT"
    ;;
		--dataloader=*)
    	DATALOADER=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			DATALOADER_OPTION="--dataloader=$DATALOADER"
    ;;
		--datalist=*)
    	DATALIST=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			DATALIST_OPTION="--datalist=$DATALIST"
    ;;
		--datalist-train=*)
    	DATALIST_TRAIN=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			DATALIST_TRAIN_OPTION="--datalist_train=$DATALIST_TRAIN"
    ;;
		--datalist-val=*)
    	DATALIST_VAL=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			DATALIST_VAL_OPTION="--datalist_val=$DATALIST_VAL"
    ;;
		--datadir=*)
    	DATADIR=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			DATADIR_OPTION="--datadir=$DATADIR"
    ;;
		--validation-data-fract=*)
    	VAL_DATA_FRACT=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			VAL_DATA_FRACT_OPTION="--validation_data_fract=$VAL_DATA_FRACT"
    ;;
		--maxnimgs=*)
    	MAXNIMGS=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			MAXNIMGS_OPTION="--maxnimgs=$MAXNIMGS"
    ;;
	
		# - NETWORK ARCHITECTURE
		--weights=*)
    	WEIGHTS=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			WEIGHTS_OPTION="--weights=$WEIGHTS"
    ;;
		--backbone=*)
    	BACKBONE=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			BACKBONE_OPTION="--backbone=$BACKBONE"
    ;;
		--backbone-strides=*)
    	BACKBONE_STRIDES=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			BACKBONE_STRIDES_OPTION="--backbone_strides=$BACKBONE_STRIDES"
    ;;

		# - TRAIN OPTIONS
		--nepochs=*)
    	NEPOCHS=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			NEPOCHS_OPTION="--nepochs=$NEPOCHS"
    ;;
		--epoch-length=*)
    	EPOCH_LENGTH=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			EPOCH_LENGTH_OPTION="--epoch_length=$EPOCH_LENGTH"
    ;;
		--nvalidation-steps=*)
    	NVAL_STEPS=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			NVAL_STEPS_OPTION="--nvalidation_steps=$NVAL_STEPS"
    ;;
		--rpn-anchor-scales=*)
    	RPN_ANCHOR_SCALES=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			RPN_ANCHOR_SCALES_OPTION="--rpn_anchor_scales=$RPN_ANCHOR_SCALES"
    ;;	
		--max-gt-instances=*)
    	MAX_GT_INSTANCES=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			MAX_GT_INSTANCES_OPTION="--max_gt_instances=$MAX_GT_INSTANCES"
    ;;
		
		--rpn-nms-threshold=*)
    	RPN_NMS_THRESHOLD=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			RPN_NMS_THRESHOLD_OPTION="--rpn_nms_threshold=$RPN_NMS_THRESHOLD"
    ;;
		--rpn-train-anchors-per-image=*)
    	RPN_TRAIN_ANCHORS_PER_IMAGE=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			RPN_TRAIN_ANCHORS_PER_IMAGE_OPTION="--rpn_train_anchors_per_image=$RPN_TRAIN_ANCHORS_PER_IMAGE"
    ;;

		--train-rois-per-image=*)
    	RPN_TRAIN_ROIS_PER_IMAGE=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			RPN_TRAIN_ROIS_PER_IMAGE_OPTION="--train_rois_per_image=$RPN_TRAIN_ROIS_PER_IMAGE"
    ;;

		--rpn-anchor-ratios=*)
    	RPN_ANCHOR_RATIOS=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			RPN_ANCHOR_RATIOS_OPTION="--rpn_anchor_ratios=$RPN_ANCHOR_RATIOS"
    ;;

		--rpn-class-loss-weight=*)
    	RPN_CLASS_LOSS_WEIGHT=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`	
			RPN_CLASS_LOSS_WEIGHT_OPTION="--rpn_class_loss_weight=$RPN_CLASS_LOSS_WEIGHT"
    ;;
		--rpn-bbox-loss-weight=*)
    	RPN_BBOX_LOSS_WEIGHT=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			RPN_BBOX_LOSS_WEIGHT_OPTION="--rpn_bbox_loss_weight=$RPN_BBOX_LOSS_WEIGHT"
    ;;
		--mrcnn-class-loss-weight=*)
    	MRCNN_CLASS_LOSS_WEIGHT=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			MRCNN_CLASS_LOSS_WEIGHT_OPTION="--mrcnn_class_loss_weight=$MRCNN_CLASS_LOSS_WEIGHT"
    ;;
		--mrcnn-bbox-loss-weight=*)
    	MRCNN_BBOX_LOSS_WEIGHT=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			MRCNN_BBOX_LOSS_WEIGHT_OPTION="--mrcnn_bbox_loss_weight=$MRCNN_BBOX_LOSS_WEIGHT"
    ;;
		--mrcnn-mask-loss-weight=*)
    	MRCNN_MASK_LOSS_WEIGHT=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			MRCNN_MASK_LOSS_WEIGHT_OPTION="--mrcnn_mask_loss_weight=$MRCNN_MASK_LOSS_WEIGHT"
    ;;
	
		# - TEST OPTIONS
		--scoreThr=*)
    	SCORE_THR=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			SCORE_THR_OPTION="--scoreThr=$SCORE_THR"
    ;;
		--iouThr=*)
    	IOU_THR=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			IOU_THR_OPTION="--iouThr=$IOU_THR"
    ;;

		# - DETECT OPTIONS
		--image=*)
    	IMAGE=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			IMAGE_OPTION="--image=$IMAGE"
    ;;

		# - RUN OPTIONS
		--logs=*)
    	LOG_DIR=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			LOGDIR_OPTION="--logs=$LOG_DIR"
    ;;
		--nthreads=*)
    	NTHREADS=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			NTHREADS_OPTION="--nthreads=$NTHREADS"
    ;;
		--ngpu=*)
    	NGPU=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			NGPU_OPTION="--ngpu=$NGPU"
    ;;
		--nimg-per-gpu=*)
    	NIMG_PER_GPU=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
			NIMG_PER_GPU_OPTION="nimg_per_gpu=$NIMG_PER_GPU"
    ;;
		--runmode=*)
    	RUNMODE=`echo $item | sed 's/[-a-zA-Z0-9]*=//'`
    ;;

    *)
    # Unknown option
    echo "ERROR: Unknown option ($item)...exit!"
    exit 1
    ;;
	esac
done


###########################
##    RUN
###########################
# - Set exec options
EXE_ARGS=""
EXE_ARGS="$EXE_ARGS $GRAYIMG_OPTION "
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


# - Run command
echo "INFO: Running mask-rcnn with args: $EXE_ARGS"
#exec $CMD &
python3 $MASKRCNN_DIR/bin/run.py $EXE_ARGS

STATUS=$?
if [ $STATUS -ne 0 ];
then
	echo "ERROR: mask-rcnn run failed with status $STATUS!"
	exit $STATUS
fi


