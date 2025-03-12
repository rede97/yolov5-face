#!/bin/env sh
conda init
conda create -n yolov5 python=3.10
pip install -r requirements.txt
