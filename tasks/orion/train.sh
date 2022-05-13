python3 ../../train.py \
--name exp_03 \
--imgsz 1600 \
--rect \
--noautoanchor \
--noaugment \
--batch-size 4 \
--mini-epoch 30000 \
--project orion \
--data data.yaml \
--cfg yolov5l6.yaml \
--hyp hyps.yaml \
--weights "/storage/train/weights/coco/yolov5l6.pt" \
--save-period 5

#--resume "/storage/train/runs/aerial_detection/02/exp_01/weights/last.pt"

