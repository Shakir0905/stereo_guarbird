import cv2
import numpy as np
from time import sleep
from threading import Lock
import pyzed.sl as sl
from threads.inference_trt import TrtYOLO
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
    
        self.model = None     
        self.flag_once = True

        self.image_overlay = ImageOverlay()

    def detections_to_custom_boxes(self, detections):
        """
        Преобразует обнаружения из формата YOLO в формат, совместимый с ZED SDK, 
        преобразуя координаты bounding box из (x, z, width, height) в [(A, B), (C, B), (C, D), (A, D)].
        """

        custom_boxes = []
        
        for det in detections:
            xywh = det
            x_min = xywh[0] - 0.5 * xywh[2]
            x_max = xywh[0] + 0.5 * xywh[2]
            y_min = xywh[1] - 0.5 * xywh[3]
            y_max = xywh[1] + 0.5 * xywh[3]
            bounding_box_2d = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    
            obj = sl.CustomBoxObjectData()
            obj.bounding_box_2d = bounding_box_2d
            obj.label = int(det[-1])
            obj.probability = det[-2]
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
                print("Processing frame...")
                
                with self.lock:
                    img = cv2.cvtColor(self.image_net, cv2.COLOR_BGRA2RGB)
                    if self.flag_once:
                        from pycuda import autoinit
                        self.model = TrtYOLO(input_shape=(640, 640), engine_path="/home/ramazanov/stereo-guardbird/weights/new_best_640.engine", conf=0.6)         
                        self.flag_once = False

                    det = self.model(img)
                    try:
                        if len(det) > 0:
                            self.detections =self.detections_to_custom_boxes(det)
                    except Exception as e:
                        print(e)
                    self.run_signal = False
            sleep(0.01)
    
    def setup_object_detection(self, zed):
        obj_param = sl.ObjectDetectionParameters()
        obj_param.filtering_mode.NMS3D#_PER_CLASS 
        # obj_param.prediction_timeout_s = 1
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True

        if obj_param.enable_tracking:
            zed.enable_positional_tracking()
        zed.enable_object_detection(obj_param)

        return obj_param
