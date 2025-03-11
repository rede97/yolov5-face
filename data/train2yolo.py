import os.path
import argparse
import torch.utils.data as data
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path: Path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = txt_path.with_name("images").joinpath(line[2:])
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert WIDERFACE dataset to YOLO format')
    parser.add_argument('original_path', type=str, 
                        help='Path to original WIDERFACE train folder')
    parser.add_argument('save_path', type=str, nargs='?', default='widerface/train',
                        help='Path to save converted YOLO format data (default: widerface/train)')
    args = parser.parse_args()

    original_path = Path(args.original_path)
    label_file = original_path / "label.txt"
    if not label_file.exists():
        print(f'Missing label.txt file: {label_file}')
        exit(1)

    save_path = Path(args.save_path)

    save_path.mkdir(parents=True, exist_ok=True)
    images_path = save_path.joinpath("images")
    labels_path = save_path.joinpath("labels")
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)


    aa = WiderFaceDetection(label_file)

    for i, img_path in enumerate(tqdm(aa.imgs_path)):
        img = cv2.imread(img_path)
        base_img = Path(img_path)
        base_txt = base_img.stem + ".txt"
        save_img_path = images_path / base_img.name
        save_txt_path = labels_path / base_txt
        with open(save_txt_path, "w") as f:
            height, width, _ = img.shape
            labels = aa.words[i]
            annotations = np.zeros((0, 14))
            if len(labels) == 0:
                continue
            for idx, label in enumerate(labels):
                annotation = np.zeros((1, 14))
                # bbox
                label[0] = max(0, label[0])
                label[1] = max(0, label[1])
                label[2] = min(width - 1, label[2])
                label[3] = min(height - 1, label[3])
                annotation[0, 0] = (label[0] + label[2] / 2) / width  # cx
                annotation[0, 1] = (label[1] + label[3] / 2) / height  # cy
                annotation[0, 2] = label[2] / width  # w
                annotation[0, 3] = label[3] / height  # h
                #if (label[2] -label[0]) < 8 or (label[3] - label[1]) < 8:
                #    img[int(label[1]):int(label[3]), int(label[0]):int(label[2])] = 127
                #    continue
                # landmarks
                annotation[0, 4] = label[4] / width  # l0_x
                annotation[0, 5] = label[5] / height  # l0_y
                annotation[0, 6] = label[7] / width  # l1_x
                annotation[0, 7] = label[8] / height  # l1_y
                annotation[0, 8] = label[10] / width  # l2_x
                annotation[0, 9] = label[11] / height  # l2_y
                annotation[0, 10] = label[13] / width  # l3_x
                annotation[0, 11] = label[14] / height  # l3_y
                annotation[0, 12] = label[16] / width  # l4_x
                annotation[0, 13] = label[17] / height  # l4_y
                str_label = "0 "
                for i in range(len(annotation[0])):
                    str_label = str_label + " " + str(annotation[0][i])
                str_label = str_label.replace('[', '').replace(']', '')
                str_label = str_label.replace(',', '') + '\n'
                f.write(str_label)
        cv2.imwrite(str(save_img_path), img)
