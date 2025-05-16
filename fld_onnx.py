import onnxruntime
import cv2
import numpy as np


def show_results(img: np.ndarray, landmarks: np.ndarray, score: float, tongue: float):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    img = img.copy()

    # clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    nmarks = landmarks.shape[0]
    for i in range(nmarks):
        point_x = int(landmarks[i][0])
        point_y = int(landmarks[i][1])
        cv2.circle(img, (point_x, point_y), tl + 1, (200, 170, 60), -1)

    tf = max(tl - 1, 1)  # font thickness
    label = "score: {} tongue: {}".format(score, tongue)

    cv2.putText(
        img,
        label,
        (0, 0),
        0,
        tl / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img


session = onnxruntime.InferenceSession("fld.onnx", providers=["CPUExecutionProvider"])


inputs = session.get_inputs()
print("inputs: ", ["{}:{}".format(i.name, i.shape[1:]) for i in inputs])
outputs = session.get_outputs()
print("outputs: ", ["{}:{}".format(o.name, o.shape[1:]) for o in outputs])
input0_name = inputs[0].name
output_names = [o.name for o in outputs]


img0 = cv2.imread("KA.NE2.27.png", cv2.IMREAD_COLOR_RGB)
img = np.array(img0, dtype=np.float32) / 255.0
img = np.expand_dims(img, axis=0)
print(img.shape)

output_result = session.run(output_names, {input0_name : img})
marks = output_result[0].reshape(-1, 3)
score = output_result[1][0, 0, 0, 0]
tongue = output_result[2][0, 0]

show_img = show_results(img0, marks,  score, tongue)
cv2.imwrite("output.jpg", show_img)
