import cv2
# from imread_from_url import imread_from_url

from yolov8 import YOLOv8

home = "/Users/wave/PycharmProjects/ImgDetectDemo"

# # Initialize video
input_file = home + "/data/img/"
output_file = home + "/data/img/py_detect/"
model_path = home + "/models/yolov8m.onnx"
img_size = 5
# Initialize yolov8 object detector
# model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)


# detection = Yolov8(args.model, args.img, args.conf_thres, args.iou_thres)
#
#     # Perform object detection and obtain the output image
#     output_image = detection.main()
#
#     # Display the output image in a window
#     cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
#     cv2.imshow('Output', output_image)
#
#     # Wait for a key press to exit
#     cv2.waitKey(0)


# Read image
for i in range(img_size):
    img = input_file + str(i) + '.jpg'
    print("detect img: ", img)
    img = cv2.imread(img)
    # img = cv2.imread(self.input_image)

    boxes, scores, class_ids = yolov8_detector(img)
    combined_img = yolov8_detector.draw_detections(img)
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Objects", combined_img)
    cv2.imwrite(output_file + str(i) + "_detect.jpg", combined_img)
    cv2.waitKey(0)

print("detect finished.")

# img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
# img = imread_from_url(img_url)

# Detect Objects


# Draw detections

