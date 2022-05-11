python ../../detect.py \
--weights aerial_detection/exp_016/weights/epoch26.pt \
--source "/storage/train/data/aerial_detection/01_val/images/Автомобиль легковой/**/*.jpeg" \
--img 768 \
--line-thickness 2 \
--device cpu
