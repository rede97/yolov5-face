import os
import cv2
import numpy as np
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path

def xywh2xxyy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return x1, x2, y1, y2


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def wider2face(label_path: Path, ignore_small=0):
    data = {}
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if '#' in line:
                path = str(label_path.with_name("images").joinpath(line.split()[-1]).resolve())
                img = cv2.imread(path)
                height, width, _ = img.shape
                data[path] = list()
            else:
                box = np.array(line.split()[0:4], dtype=np.float32)  # (x1,y1,w,h)
                if box[2] < ignore_small or box[3] < ignore_small:
                    continue
                box = convert((width, height), xywh2xxyy(box))
                label = '0 {} {} {} {} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1'.format(round(box[0], 4), round(box[1], 4),
                                                                             round(box[2], 4), round(box[3], 4))
                data[path].append(label)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert WIDERFACE validation dataset to YOLO format')
    parser.add_argument('root_path', type=str,
                        help='Path to original WIDERFACE validation folder')
    parser.add_argument('save_path', type=str, nargs='?', default='widerface/val',
                        help='Path to save converted YOLO format data (default: widerface/val)')
    parser.add_argument('-s', '--symlink', action='store_true',
                        help='Use symlinks instead of copying files')
    args = parser.parse_args()

    root_path = Path(args.root_path)
    label_file = root_path / "label.txt"
    if not label_file.exists():
        print(f'Missing label.txt file: {label_file}')
        exit(1)

    save_path = Path(args.save_path)

    save_path.mkdir(parents=True, exist_ok=True)
    images_path = save_path.joinpath("images")
    labels_path = save_path.joinpath("labels")
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    datas = wider2face(label_file)
    for idx, img_path in enumerate(datas.keys()):
        pict_name = os.path.basename(img_path)
        out_img = images_path / f'{idx}.jpg'
        out_txt = labels_path / f'{idx}.txt'
        if args.symlink:
            if os.path.exists(out_img):
                os.remove(out_img)
            os.symlink(img_path.resolve(), out_img)
        else:
            shutil.copyfile(img_path, out_img)
        labels = datas[img_path]
        f = open(out_txt, 'w')
        for label in labels:
            f.write(label + '\n')
        f.close()
