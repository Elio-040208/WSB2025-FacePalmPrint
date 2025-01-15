import torch
from models.mobileNetV1 import MobileNetV1
import torchvision.transforms as transforms
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
from PIL import Image
import dlib
import cv2

EAR_THRESH = 0.3
EYE_close = 2
MAR_THRESH = 0.5
FACE_THRESH = 2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

count_eye = 0
total_blinks = 0
count_mouth = 0
total_mouth = 0
distance_left = 0
distance_right = 0
total_face = 0

# 计算眼睛的纵横比（EAR）
def EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# 计算嘴巴的纵横比（MAR）
def MAR(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])
    B = np.linalg.norm(mouth[4] - mouth[7])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

# 计算鼻子到左右脸边界的欧式距离
def nose_jaw_distance(nose, jaw):
    face_left1 = dist.euclidean(nose[0], jaw[0])
    face_right1 = dist.euclidean(nose[0], jaw[16])
    face_left2 = dist.euclidean(nose[3], jaw[2])
    face_right2 = dist.euclidean(nose[3], jaw[14])
    return (face_left1, face_right1, face_left2, face_right2)


def recognize_palm(image, model_path, num_classes=293):
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = MobileNetV1(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 替换为你的模型输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 替换为你的数据归一化参数
    ])

    # 预处理图像
    pil_image = Image.fromarray(image)
    input_image = transform(pil_image).unsqueeze(0).to(device)  # 增加 batch 维度

    # 预测
    with torch.no_grad():
        output = model(input_image)
        _, predicted_label = torch.max(output, dim=1)  # 获取预测类别
        predicted_label = predicted_label.item() + 1  # 加1以匹配文件夹编号

    print(f"Predicted Label: {predicted_label}")
    return predicted_label

def recognize_face_landmarks(image):
    image = imutils.resize(image, width=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    results = []

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        jaw = shape[jStart:jEnd]

        ear_left = EAR(leftEye)
        ear_right = EAR(rightEye)
        ear = (ear_left + ear_right) / 2.0

        mar = MAR(mouth)

        face_distance = nose_jaw_distance(shape[27:31], shape[0:17])

        counters = (count_eye, total_blinks, count_mouth, total_mouth, distance_left, distance_right, total_face)
        thresholds = (EAR_THRESH, EYE_close, MAR_THRESH, FACE_THRESH)

        results.append((ear, mar, face_distance, thresholds, counters))
    
    return results