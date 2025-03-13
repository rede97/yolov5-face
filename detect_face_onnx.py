# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import sys
import os

import numpy as np
import cv2
import torch
from numpy import random
import copy
import onnxruntime
import numpy as np

model_stride = np.array([8.0, 16.0, 32.0])

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import (
    check_img_size,
    non_max_suppression_face,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def padding_image(img, new_shape=(640, 640)):
    h0, w0 = img.shape[:2]
    assert max(h0, w0) <= 640
    h_pad = (640 - h0) / 2
    w_pad = (640 - w0) / 2
    top, bottom = int(round(h_pad - 0.1)), int(round(h_pad + 0.1))
    left, right = int(round(w_pad - 0.1)), int(round(w_pad + 0.1))
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT
    )  # add border


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def post_proc(idx: int, preds: np.ndarray, anchors: np.ndarray):
    (bs, na, ny, nx, no) = preds.shape
    yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    grid = np.stack((xv, yv), 2)
    grid = np.broadcast_to(grid, (1, na, ny, nx, 2))
    anchor_grid = np.array(anchors[idx] * model_stride[idx]).view().reshape((1, na, 1, 1, 2))
    anchor_grid = np.broadcast_to(anchor_grid, (1, na, ny, nx, 2))

    y = np.copy(preds)
    y[:, :, :, :, 0:5] = sigmoid(preds[:, :, :, :, 0:5])
    y[:, :, :, :, 15:] = sigmoid(preds[:, :, :, :, 15:])
    box_xy = (y[:, :, :, :, 0:2] * 2. - 0.5 + grid) * model_stride[idx]
    box_wh = (y[:, :, :, :, 2:4] * 2) ** 2 * anchor_grid

    landm1 = y[:, :, :, :, 5:7] * anchor_grid + grid * model_stride[idx]  # landmark x1 y1
    landm2 = y[:, :, :, :, 7:9] * anchor_grid + grid * model_stride[idx]  # landmark x2 y2
    landm3 = y[:, :, :, :, 9:11] * anchor_grid + grid * model_stride[idx]  # landmark x3 y3
    landm4 = y[:, :, :, :, 11:13] * anchor_grid + grid * model_stride[idx]  # landmark x4 y4
    landm5 = y[:, :, :, :, 13:15] * anchor_grid + grid * model_stride[idx]  # landmark x5 y5

    y = np.concatenate((box_xy, box_wh, y[:, :, :, :, 4:5], landm1, landm2, landm3, landm4, landm5, y[:, :, :, :, 15:]), 4)
    y = y.view().reshape((bs, na * nx * ny,no))
    return y
    


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def show_results(img, xyxy, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    cv2.rectangle(
        img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA
    )

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(
        img,
        label,
        (x1, y1 - 2),
        0,
        tl / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img


def detect(
    session: onnxruntime.InferenceSession,
    source,
    project,
    name,
    exist_ok,
    save_img,
    view_img,
):
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz = (640, 640)

    input0 = session.get_inputs()[0]
    print(input0.name, input0.shape)
    outputs = session.get_outputs()
    output_names = [o.name for o in outputs]
    print(output_names)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    
    # Dataloader
    if webcam:
        print("loading streams:", source)
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1  # batch_size
    else:
        print("loading images", source)
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    for path, im, im0s, vid_cap in dataset:

        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis=0)
        else:
            orgimg = im.transpose(1, 2, 0)

        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        img = padding_image(img0)
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = img.astype(np.float32) / 255.0  # uint8 to fp16/32
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
        print("input image: ", img.shape)
        # Inference
        outputs = session.run(output_names, {input0.name: img})
        anchors = outputs[3]
        pred = []
        for idx in range(3):
            pred.append(post_proc(idx, outputs[idx], anchors))
        pred = np.concatenate(pred, 1)

        print(pred.shape)

        pred = torch.from_numpy(pred)

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        print(len(pred[0]), "face" if len(pred[0]) == 1 else "faces")

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(Path(save_dir) / p.name)  # im.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(
                    img.shape[2:], det[:, 5:15], im0.shape
                ).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:15].view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()

                    im0 = show_results(im0, xyxy, conf, landmarks, class_num)

            if view_img:
                cv2.imshow("result", im0)
                k = cv2.waitKey(1)

            # Save results (image with detections)
            if save_img:
                print("save: ", save_path)
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(
                            Path(save_path).with_suffix(".mp4")
                        )  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    try:
                        vid_writer[i].write(im0)
                    except Exception as e:
                        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        nargs="+",
        type=str,
        default="weights/yolov5n-face-relu.onnx",
        help="model.onnx path(s)",
    )
    parser.add_argument(
        "--source", type=str, default="0", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--view-img", action="store_true", help="show results")
    opt = parser.parse_args()

    # model = load_model(opt.weights, device)
    session = onnxruntime.InferenceSession(opt.onnx, providers=["CPUExecutionProvider"])

    detect(
        session,
        opt.source,
        opt.project,
        opt.name,
        opt.exist_ok,
        opt.save_img,
        opt.view_img,
    )
