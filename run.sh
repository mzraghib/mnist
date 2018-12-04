#!/bin/bash

python /tensorflow/models/research/object_detection/train.py --logtostderr --train_dir=/dbc/output --pipeline_config_path=/dbc/code/ssd_mobilenet_v1_coco.config
