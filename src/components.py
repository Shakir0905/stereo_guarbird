import pyzed.sl as sl
from configs import config
import ogl_viewer.viewer3d as gl

from threads.detector import Detector
from logs.logging_scripts.viewer2d import CVViewer
from threads.audio_manager import AudioPlayer
from utilities.brief_checker import BRIEFChecker
from logs.logging_scripts.video_logger import VideoLogger
import math
import numpy as np
from src.tracking_viewer import TrackingViewer

def get_distance(init_params):
    zed = sl.Camera()
    if zed.open(init_params) == sl.ERROR_CODE.SUCCESS and zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
        image = sl.Mat()
        zed.retrieve_image(image, sl.VIEW.LEFT)
        x, y = round(image.get_width() / 2), round(image.get_height() / 2)
        zed.retrieve_measure(image, sl.MEASURE.XYZRGBA)
        err, point_cloud_value = image.get_value(x, y)
        if math.isfinite(point_cloud_value[2]):
            return math.sqrt(sum(val ** 2 for val in point_cloud_value[:3]))
    return 20

def initialize_camera():
    zed = sl.Camera()
    input_type = sl.InputType()

    if config.svo is not None:
        input_type.set_from_svo_file(config.svo)
    
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD2K  
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = get_distance(init_params) + 5#config.depth_distance
    print("Depth_distance", round(init_params.depth_maximum_distance, 1))
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open Error: {status}")
        exit()

    print("Initialized Camera")
    return zed, init_params

def initialize_components():
    """
    Инициализация всех компонентов системы и создание потоков.
    """
    detector = Detector(config.weights, config.img_size, config.conf_thres, config.iou_thres, config.svo)
    audio_player = AudioPlayer()
    zed, init_params = initialize_camera()
    obj_param = detector.setup_object_detection(zed)
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    briefchecker = BRIEFChecker()
    video_logger = VideoLogger()
    viewer3d = gl.GLViewer()
    viewer2d = CVViewer()

    video_logger.initialize_video_writer()
    
    return detector, zed, init_params, obj_param, obj_runtime_param, viewer2d, viewer3d, video_logger, briefchecker, audio_player



def initialize_objects(zed, obj_param, viewer3d, init_params):  # Add init_params as an argument
    objects = sl.Objects()

    image_left = sl.Mat()
    image_left_tmp = sl.Mat()
        
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    viewer3d.init(camera_infos.camera_configuration.calibration_parameters.left_cam, obj_param.enable_tracking)

    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = TrackingViewer(viewer3d, tracks_resolution, camera_config.fps, init_params.depth_maximum_distance*1000, duration=2)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    
    cam_w_pose = sl.Pose()
    runtime_parameters = sl.RuntimeParameters()

    return objects, image_left, image_left_tmp, display_resolution, image_scale, image_left_ocv, track_view_generator, image_track, cam_w_pose, runtime_parameters
