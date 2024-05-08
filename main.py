import cv2
import torch
import numpy as np
from time import time
import pyzed.sl as sl
from datetime import datetime

from configs import config
from threads.thread_manager import start_threads
from threads.input_listener import InputListener
from utilities.mode_controller import ModeController
from src.components import initialize_components, initialize_objects

class StereoVisionSystem:
    def __init__(self):
        self.initialize_system()
        self.extra_frames = 0
        self.last_frame_time = 0
        self.initiate_extra_recording = False
        self.previous_all_persons_inside = False  
        self.previous_polygon_dynamic = True
        self.log_image = None

    def initialize_system(self):
        with torch.no_grad():
            # Инициализация компонентов системы
            self.detector, self.zed, self.init_params, self.obj_param, self.obj_runtime_param, \
            self.viewer2d, self.viewer3d, self.video_logger, self.briefchecker, self.audio_player = initialize_components()

            # Инициализация объектов для обработки кадров
            self.objects, self.image_left, self.image_left_tmp, self.display_resolution, \
            self.image_scale, self.image_left_ocv, self.track_view_generator, self.image_track, \
            self.cam_w_pose, self.runtime_parameters = \
            initialize_objects(self.zed, self.obj_param, self.viewer3d, self.init_params)  

            # Установка зависимостей и начало работы потоков
            self.mode_controller = ModeController(self.track_view_generator, self.viewer2d, self.viewer3d)
            self.input_listener = InputListener(self.detector, self.mode_controller)
            self.viewer3d.set_dependencies(self.viewer2d, self.track_view_generator, self.mode_controller)
            start_threads(self.detector, self.audio_player, self.input_listener)            
                       
            
    def main_loop(self):
        while self.viewer3d.is_available() and not self.detector.exit_signal:
            self.fps = 1 / (time() - self.last_frame_time)
            self.last_frame_time = time()
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.process_frame()
                if config.enable_gui:
                    self.display_gui()
    
                self.display_status_info()

                self.audio_player.all_persons_inside = self.viewer3d.all_persons_inside
                self.audio_player.is_polygon_dynamic = self.viewer3d.is_polygon_dynamic
                self.audio_player.count_conuses = len(self.viewer3d.current_polygon)
                     
                current_all_persons_inside = self.viewer3d.all_persons_inside
                if current_all_persons_inside and not self.previous_all_persons_inside:
                    self.initiate_extra_recording = True
                    add_seconds = 5
                    extra_frames = int(self.fps * add_seconds)  

                if not current_all_persons_inside or (self.initiate_extra_recording and extra_frames > 0):
                    self.video_logger.record_video(self.log_image)
                    
                if self.initiate_extra_recording:
                    extra_frames -= 1
                    if extra_frames <= 0:
                        self.initiate_extra_recording = False
                        extra_frames = 0  

                self.previous_all_persons_inside = current_all_persons_inside
            else:
                self.detector.exit_signal = True
        self.cleanup()

    def process_frame(self):
        with self.detector.lock:
            self.zed.retrieve_image(self.image_left_tmp, sl.VIEW.LEFT)
            self.detector.image_net = self.image_left_tmp.get_data()
            self.detector.image_net = self.detector.image_overlay.overlay_logo_and_text(self.detector.image_net, self.viewer3d)
            self.detector.run_signal = True

        self.zed.ingest_custom_box_objects(self.detector.detections)
        self.zed.retrieve_objects(self.objects, self.obj_runtime_param)
        self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution)
        self.zed.get_position(self.cam_w_pose, sl.REFERENCE_FRAME.WORLD)

        self.image_left_ocv = self.image_left.get_data()
        self.image_left_ocv = self.detector.image_overlay.overlay_logo_and_text(self.image_left_ocv, self.viewer3d)
    
        self.viewer3d.update_view(self.image_left_tmp, self.objects)
        self.viewer2d.render_2D(self.image_left_ocv, self.image_scale, self.objects, self.objects.is_tracked, self.viewer3d)                
        self.track_view_generator.generate_view(self.objects, self.cam_w_pose, self.image_track, self.objects.is_tracked)

        self.log_image = cv2.hconcat([self.image_left_ocv, self.image_track])[:, :, :3]
        timestamp_image = np.zeros((50, self.log_image.shape[1], 3), dtype=np.uint8)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(timestamp_image, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        self.log_image = np.vstack([timestamp_image, self.log_image])

    def display_gui(self):
        image_opngl = self.viewer3d.capture_opengl_window()
        
        image_track_cv = cv2.cvtColor(self.image_track, cv2.COLOR_BGRA2BGR)
        image_opngl_cv = cv2.cvtColor(image_opngl, cv2.COLOR_BGR2RGB)
                        
        image_track_resized = cv2.resize(image_track_cv, (int(image_track_cv.shape[1] * 1016 / image_track_cv.shape[0]), 1016))
        image_opngl_resized = cv2.resize(image_opngl_cv, (int(image_opngl_cv.shape[1] * 1016 / image_opngl_cv.shape[0]), 1016))
        
        concatenated_image = cv2.hconcat([image_opngl_resized, image_track_resized])
        concatenated_image = cv2.resize(concatenated_image, (1920, 800))
        # cv2.imshow("ZED | 2D View and Birds View", concatenated_image)
        # cv2.imshow("ZED | 2D View and Birds View", self.log_image)
        

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):  
            self.detector.exit_signal = True
        elif key == ord('1'):  
            self.mode_controller.toggle_all_modes()

    def display_status_info(self):
        print(
            f"\n{'[IS_OBSCURED]' if self.briefchecker.step(self.image_left_ocv) else '[NOT_OBSCURED]'}\n"
            f"[STATUS ID]: {self.viewer3d.person_status}\n"
            f"{'[NORM]' if self.viewer3d.all_persons_inside else '[ALARM]'}\n"
            f"{'[POLYGON_DYNAMIC]' if self.viewer3d.is_polygon_dynamic or not self.viewer3d.current_polygon else '[POLYGON_STATIC]'}\n"
            f"[CONES IN THE POLYGON]: {len(self.viewer3d.current_polygon)} OF "
            f"{sum(obj.raw_label == 1 for obj in self.objects.object_list)}\n"
            f"[FPS]: {self.fps:.2f}"
        )

    def cleanup(self):
        cv2.destroyAllWindows()
        self.viewer3d.exit()
        self.detector.exit_signal = True
        self.zed.close()

if __name__ == '__main__':
    system = StereoVisionSystem()
    system.main_loop()
