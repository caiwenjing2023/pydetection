import numpy as np
import cv2
import torch
import json
import os
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel
import time
import datetime


def getColor(x, y, w, h, frame):
    cropped_img = frame[y:h, x:w]
    now = datetime.datetime.now()
    # cv2.imwrite(
    #     now.strftime("%H:%M:%S") + "cropped_img.jpg",
    #     cropped_img)
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    top = max(gray_hist)
    for x, y in enumerate(gray_hist):
        if gray_hist[x] == top:
            index = x
            break
    if index < 150:
        return False
    else:
        return True


def video_detection_action(path, model):
    path = path
    weights = "last.pt"
    person_count = 0  # 检测到的次数
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    device = 0  # If you want to run with cpu -> device = "cpu"
    device = select_device(str(device))

    imgsz = 640
    half = True

    conf_thres = 0.5#修改1
    iou_thres = 0.5

    if half:
        model.half()  # to FP16

    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once for warmup

    test = 31
    count_f = 0
    count = 0
    to_add = 0

    while True:
        flag, im0 = cap.read()
        count_f += 1
        if flag: # 视频末尾帧处理
            width = im0.shape[1]
            im0 = im0[:, int(width * 0.5):, :]  # 截取主驾驶位
            img = letterbox(im0, imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            # img = np.ascontiguousarray(img)
            img = torch.from_numpy(img.copy()).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t1 = time.time()
            with torch.no_grad():
                pred = model(img, augment=False)[0]

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None,
                                           agnostic=False)  # for filtering change classes
            flag_phone = False
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    x1, y1, x2, y2 = det[:, :4].cpu().numpy()[0]
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]}'  # 检测类型
                        if label == "phone":#修改2
                            flag_phone = True
            if flag_phone:
                count = to_add + count + 1
                to_add = 0
            else:
                if count != 0:
                    to_add += 1
        else:
            break
    if count >= fps * 3:  # 假设fps为视频的帧率和返回类型，修改3
        test = 30  # 30为检测到手机
    else:
        test = 31  # 31为未检测到手机
    return test


# 输出处理
if __name__ == '__main__':
    video_path = './data/video/'
    all_cnt = 0
    true_cnt = 0
    weights = "last.pt"
    device = 0  # If you want to run with cpu -> device = "cpu"
    device = select_device(str(device))
    model = attempt_load(weights, map_location=device)
    for each in os.listdir(video_path):
        video = video_path + each
        start = time.time()
        reg = video_detection_action(video, model)
        end = time.time()
        t = end - start
        rlt = {"result": {"category": reg, "duration": t}}
        res = json.dumps(rlt)#结果的json
        all_cnt += 1
        if str(reg) == str(each.split('_')[-2]):
            true_cnt += 1
        print('all:', all_cnt, '; acc:', true_cnt / all_cnt, "; duration:", t)
        # print(res)
        # print(each)
