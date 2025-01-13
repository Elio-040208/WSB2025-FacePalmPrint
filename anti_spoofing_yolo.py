import cv2
from ultralytics import YOLO


def predImage(image):
    # Load a model
    model = YOLO('yolo_v8.pt')  # load a custom model
    # Predict with the model
    results = model(image, verbose=False)  # predict on an image
    for result in results:
        probs = result.probs  # Probs object for classification outputs
        if probs.top1 == 1:
            return "spoof"
        elif probs.top1 == 0:
            return "real"

def facedetect(windowname, camera_id):
    cv2.namedWindow(windowname)
    cap = cv2.VideoCapture(camera_id)
    classfier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    color = (0, 225, 0)  # 人脸框的颜色

    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据，ok表示摄像头读取状态，frame表示摄像头读取的图像矩阵mat类型
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 图像灰度化
        faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        # 利用分类器检测灰度图像中的人脸矩阵数，1.2和3分别为图片缩放比例和需要检测的有效点数

        if len(faceRects) > 0:

            for faceRect in faceRects:
                x, y, w, h = faceRect

                prediction = predImage(frame)
                if prediction:
                    if prediction == "spoof":
                        color = (0, 0, 255)
                    # Put the prediction text above the face rectangle
                    cv2.putText(frame, prediction, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.rectangle(frame, (x - 10, y - 10), (x + w - 10, y + h - 10), color, 2)
        cv2.imshow(windowname, frame)  # 显示图像

        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):  # 按q退出
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    print('face detecting... ')
    facedetect('facedetect', 0)

