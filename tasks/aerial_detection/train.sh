
python3 ../../train.py \
--name exp_01 \
--img 768 \
--batch-size 12 \
--project aerial_detection \
--data data.yaml \
--cfg yolov5l6.yaml \
--hyp hyps.yaml \
--weights "" \
--save-period 1 \
--resume

#--resume "/storage/train/runs/aerial_detection/02/exp_01/weights/last.pt"

#python3 ../../train.py --name exp_01 \
#--img 768 \
#--batch-size 4 \
#--project aerial_detection \
#--data data.yaml \
#--cfg yolov5l6.yaml \
#--hyp hyps.yaml \
#--weights "/storage/train/weights/coco/yolov5l6.pt" \
#--save-period 1 \
#--resume
