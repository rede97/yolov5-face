from pathlib import Path
from torchinfo import summary
import onnx
from utils.general import set_logging, check_img_size
from utils.activations import Hardswish, SiLU
from models.experimental import attempt_load
import models
import torch.nn as nn
import torch
import argparse
import sys
import time
import numpy as np

sys.path.append("./")  # to run '$ python *.py' files in subdirectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="./weights/yolov5n-face-relu.pt",
        help="weights path",
    )  # from yolov5/models/
    parser.add_argument(
        "--img_size", nargs="+", type=int, default=[640, 640], help="image size"
    )  # height, width
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--onnx_infer", action="store_true", default=True, help="onnx infer test"
    )
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(
        opt.weights, map_location=torch.device("cpu")
    )  # load FP32 model
    delattr(model.model[-1], "anchor_grid")
    model.model[-1].anchor_grid = [
        torch.zeros(1)
    ] * 3  # nl=3 number of detection layers
    model.model[-1].export_cat = True
    model.model[-1].export_x = True
    model.eval()

    anchors_path = Path(opt.weights).with_name("anchor.txt")
    anchors = model.model[-1].anchors.numpy().reshape((3, 6))
    np.savetxt(anchors_path, anchors, delimiter=",", newline=",")
    print("Save Anchors: ", anchors_path)

    summary(model)

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [
        check_img_size(x, gs) for x in opt.img_size
    ]  # verify img_size are gs-multiples

    # # Input
    img = torch.rand(
        opt.batch_size, 3, *opt.img_size
    )  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
        if isinstance(m, models.common.ShuffleV2Block):  # shufflenet block nn.SiLU
            for i in range(len(m.branch1)):
                if isinstance(m.branch1[i], nn.SiLU):
                    m.branch1[i] = SiLU()
            for i in range(len(m.branch2)):
                if isinstance(m.branch2[i], nn.SiLU):
                    m.branch2[i] = SiLU()
    y = model(img)  # dry run

    # ONNX export
    print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
    f = opt.weights.replace(".pt", ".onnx")  # filename
    # model.fuse()  # only for ONNX
    input_names = ["input"]
    output_names = ["out80", "out40", "out20"]
    dynamic_axes = {'input': {0: 'batch'},
                      'out80': {0: 'batch'},
                      'out40': {0: 'batch'},
                      'out20': {0: 'batch'},
                      }
    torch.onnx.export(
        model,
        img,
        f,
        verbose=False,
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print("ONNX export success, saved as %s" % f)
    # Finish
    print(
        "\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron."
        % (time.time() - t)
    )

    # onnx infer
    if opt.onnx_infer:
        import onnxruntime
        import numpy as np

        providers = ["CPUExecutionProvider"]
        session = onnxruntime.InferenceSession(f, providers=providers)
        im = img.cpu().numpy().astype(np.float32)  # torch to numpy
        output_names = [o.name for o in session.get_outputs()]
        y_onnx = session.run(output_names, {session.get_inputs()[0].name: im})
        results = y_onnx[0:3]
        for i in range(len(results)):
            print(
                f"name: {output_names[i]}, pred's shape[{i}] is {results[i].shape}, origin shape is {y[i].shape}")
            print(
                "max(|torch_pred - onnx_pred|) =",
                abs(y[i].cpu().numpy() - results[i]).max(),
            )
