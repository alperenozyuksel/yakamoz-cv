import torch
import cv2
import numpy as np

# YOLOv5 modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/deneme.onnx')

# Video kaynağını aç
cap = cv2.VideoCapture('video/640640.mp4')

class Dogfight:

    def __init__(self):
        self.yolo()

    def draw_center_square(self, frame, square_size=50, color=(0, 255, 0), thickness=2):
        # frame'i kopyala, böylece üzerine yazılabilir
        frame_copy = np.copy(frame)

        # Görüntünün boyutlarını al
        height, width, _ = frame_copy.shape

        # Karenin sol üst ve sağ alt köşe koordinatlarını hesapla
        top_left = (width // 2 - square_size // 2, height // 2 - square_size // 2)
        bottom_right = (width // 2 + square_size // 2, height // 2 + square_size // 2)

        # Kareyi çiz
        cv2.rectangle(frame_copy, top_left, bottom_right, color, thickness)

        return frame_copy

    def yolo(self):
        while cap.isOpened():
            self.ret, self.frame = cap.read()
            if not self.ret:
                break

            # YOLOv5 modeli ile tahmin yap
            self.results = model(self.frame)

            # Sonuçları çerçeveye çiz
            self.frame = self.results.render()[0]

            # Ekranın ortasına kare çiz
            self.frame = self.draw_center_square(self.frame, square_size=300, color=(255, 0, 0), thickness=3)

            # Çerçeveyi göster
            cv2.imshow('YOLOv5 Video', self.frame)

            # 'q' tuşuna basıldığında çık
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Dogfight sınıfını başlat
Dogfight()
