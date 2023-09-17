import cv2
from yolov8 import YOLOv8

home = "/Users/wave/PycharmProjects/ImgDetectDemo"

# # Initialize video
input_file = home + "/data/video/test_demo.mp4"
output_file = home + "/data/video/test_demo_detect.mp4"
model_path = home + "/models/yolov8m.onnx"

cap = cv2.VideoCapture(input_file)

start_time = 5 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (3840, 2160))

# Initialize YOLOv7 models
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)
    out.write(combined_img)

out.release()
