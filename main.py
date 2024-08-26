import torch
import cv2
from cv2 import resize


model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/deneme.onnx')


cap = cv2.VideoCapture('video/640640.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Render the results on the frame
    frame = results.render()[0]


    # Display the frame
    cv2.imshow('YOLOv5 Video', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()