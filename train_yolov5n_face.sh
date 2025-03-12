conda init
conda activate yolov5
nohup python train.py --cfg models/yolov5n.yaml --weights weights/yolov5n-face.pt --epochs 50 &