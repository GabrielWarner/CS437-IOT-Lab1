# shell script to install libs and make tflite models
sh setup.sh 

# command to run object detection
libcamerify python3 detect.py \
  --model efficientdet_lite0.tflite