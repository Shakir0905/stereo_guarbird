# import cv2
# import numpy as np
# from time import sleep
# from threading import Lock
# from ultralytics import YOLO
# import pyzed.sl as sl

# from utilities.image_overlay import ImageOverlay

# class Detector:
#     def __init__(self, weights, img_size, conf_thres, iou_thres=0.45, svo=None):
#         self.svo = svo
#         self.weights = weights
#         self.img_size = img_size
#         self.iou_thres = iou_thres  # Initialize iou_thres here
#         self.conf_thres = conf_thres
        
#         self.detections = []
#         self.image_net = None
#         self.run_signal = False
#         self.exit_signal = False
        
#         self.lock = Lock()
#         self.model = YOLO(weights)
#         self.image_overlay = ImageOverlay()

#     def detections_to_custom_boxes(self, detections):
#         custom_boxes = []
        
#         # Пример статических координат для класса 1
#         static_detections = np.array([
#             [876, 900, 100, 200],
#             [1712, 600, 100, 200],
#             [1925, 900, 100, 200],
#             [560, 600, 100, 200]
#         ])
        
#         # Переменная для отслеживания индекса в static_detections для класса 1
#         static_index = 0
        
#         for det in static_detections:
#             if True:#det.cls[0] == 1:
#                 # Используем статические координаты для класса 1
#                 if static_index < len(static_detections):
#                     xywh = static_detections[static_index]
#                     static_index += 1  # Увеличиваем индекс для следующего использования
#                 else:
#                     # Если индекс выходит за пределы, пропускаем обработку
#                     continue
#             else:
#                 # Для класса 0 используем координаты из det.xywh
#                 xywh = det.xywh[0]

#             x_min = xywh[0] - 0.5 * xywh[2]
#             x_max = xywh[0] + 0.5 * xywh[2]
#             y_min = xywh[1] - 0.5 * xywh[3]
#             y_max = xywh[1] + 0.5 * xywh[3]
        
#             bounding_box_2d = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
            
#             obj = sl.CustomBoxObjectData()
#             obj.bounding_box_2d = bounding_box_2d
#             obj.label = det.cls[0]
#             obj.probability = det.conf[0]
#             obj.is_grounded = False
#             custom_boxes.append(obj)
        
#         return custom_boxes


#     def torch_thread(self):
#         """
#         Поток для выполнения инференса модели YOLO.
#         """
#         print("YOLO network initialized.")
#         print(self.conf_thres)
#         while not self.exit_signal:
#             if self.run_signal:
#                 print("Processing frame...")
                
#                 with self.lock:
#                     img = cv2.cvtColor(self.image_net, cv2.COLOR_BGRA2RGB)
#                     det = self.model.predict(img, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres)[0].cpu().numpy().boxes
#                     self.detections =self.detections_to_custom_boxes(det)
#                     self.run_signal = False
#             sleep(0.01)
    
#     def setup_object_detection(self, zed):
#         obj_param = sl.ObjectDetectionParameters()
#         obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
#         obj_param.enable_tracking = True

#         if obj_param.enable_tracking:
#             zed.enable_positional_tracking()
#         zed.enable_object_detection(obj_param)

#         return obj_param



import cv2
import numpy as np
from time import sleep
from threading import Lock
from ultralytics import YOLO
import pyzed.sl as sl

from utilities.image_overlay import ImageOverlay

class Detector:
    def __init__(self, weights, img_size, conf_thres, iou_thres=0.45, svo=None):
        self.svo = svo
        self.weights = weights
        self.img_size = img_size
        self.iou_thres = iou_thres  # Initialize iou_thres here
        self.conf_thres = conf_thres
        
        self.detections = []
        self.image_net = None
        self.run_signal = False
        self.exit_signal = False
        
        self.lock = Lock()
        self.model = YOLO(weights)
        self.image_overlay = ImageOverlay()

    def detections_to_custom_boxes(self, detections):
        """
        Преобразует обнаружения из формата YOLO в формат, совместимый с ZED SDK, 
        преобразуя координаты bounding box из (x, z, width, height) в [(A, B), (C, B), (C, D), (A, D)].
        """

        custom_boxes = []
        
        for det in detections:
            xywh = det.xywh[0]
            x_min = xywh[0] - 0.5 * xywh[2]
            x_max = xywh[0] + 0.5 * xywh[2]
            y_min = xywh[1] - 0.5 * xywh[3]
            y_max = xywh[1] + 0.5 * xywh[3]
            bounding_box_2d = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

            obj = sl.CustomBoxObjectData()
            obj.bounding_box_2d = bounding_box_2d
            obj.label = det.cls[0]
            # if obj.label == 1:
                # print("bounding_box_2d", bounding_box_2d)
            obj.probability = det.conf[0]
            obj.is_grounded = False
            custom_boxes.append(obj)
            
        return custom_boxes

    def torch_thread(self):
        """
        Поток для выполнения инференса модели YOLO.
        """
        print("YOLO network initialized.")
        print(self.conf_thres)
        while not self.exit_signal:
            if self.run_signal:
                # print("Processing frame...")
                
                with self.lock:
                    img = cv2.cvtColor(self.image_net, cv2.COLOR_BGRA2RGB)
                    det = self.model.predict(img, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres)[0].cpu().numpy().boxes
                    self.detections =self.detections_to_custom_boxes(det)
                    self.run_signal = False
            sleep(0.01)
    
    def setup_object_detection(self, zed):
        obj_param = sl.ObjectDetectionParameters()
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True

        if obj_param.enable_tracking:
            zed.enable_positional_tracking()
        zed.enable_object_detection(obj_param)

        return obj_param
