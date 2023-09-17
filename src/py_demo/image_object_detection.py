import cv2
from yolov8 import YOLOv8

home = "/Users/wave/PycharmProjects/ImgDetectDemo"

# # Initialize video
input_file = home + "/data/img/"
output_file = home + "/data/img/py_detect/"
model_path = home + "/models/yolov8m.onnx"
img_size = 5

# Initialize yolov8 object detector
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Read image
for i in range(img_size):
    img = input_file + str(i) + '.jpg'
    print("detect img: ", img)
    img = cv2.imread(img)

    boxes, scores, class_ids = yolov8_detector(img)
    combined_img = yolov8_detector.draw_detections(img)
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Objects", combined_img)
    cv2.imwrite(output_file + str(i) + "_detect.jpg", combined_img)
    cv2.waitKey(0)

print("detect finished.")

