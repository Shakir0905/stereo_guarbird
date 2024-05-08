import cv2
import numpy as np
import math
import pyzed.sl as sl
from configs import config
# ----------------------------------------------------------------------
#       2D TRACKING VIEW
# ----------------------------------------------------------------------


class TrackPoint:
    """ Класс для представления точки трекинга """
    def __init__(self, position, tracking_state, timestamp):
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        self.tracking_state = tracking_state
        self.timestamp = timestamp
        
    def get_xyz(self):
        return 1000*np.array([self.x, self.y, self.z])


class Tracklet:
    """ Класс для представления треклета (последовательности точек трекинга) """
    def __init__(self, obj, label, timestamp):
        self.id = obj.id
        self.label = label
        self.object_type = obj.label
        self.is_alive = True
        self.last_detected_timestamp = timestamp
        self.recovery_cpt = 0
        self.recovery_length = 5
        self.positions = [TrackPoint(obj.position, 0, timestamp)]
        self.positions_to_draw = [TrackPoint(obj.position, 0, timestamp)]
        self.tracking_state = obj.tracking_state

    def addDetectedPoint(self, obj, timestamp, smoothing_window_size):
        if self.positions[-1].tracking_state == 1 or self.recovery_cpt < self.recovery_length:
            if self.positions[-1].tracking_state == 1:
                self.recovery_cpt = 0
            else:
                self.recovery_cpt += 1
        self.positions.append(TrackPoint(obj.position, 0, timestamp))
        self.tracking_state = obj.tracking_state
        self.last_detected_timestamp = timestamp
        self.positions_to_draw.append(TrackPoint(obj.position, 0, timestamp))


class TrackingViewer:
    def __init__(self, viewer3d, res, fps_, D_max, duration):
        self.window_width = res.width
        self.window_height = res.height
        self.camera_offset = 50
        self.min_length_to_draw = 3
        self.fov = -1.0
        self.background_color = (248, 48, 48, 255)
        self.has_background_ready = False
        self.background = np.full((self.window_height, self.window_width, 4), (128, 128, 128, 155), dtype=np.uint8)
        self.fov_color = (255, 117, 44, 255)
        self.do_smooth = False
        self.history_duration = duration * 1000 * 1000 * 1000
        self.smoothing_window_size = int(math.ceil(0.08 * fps_) + 0.5)
        self.z_min = -D_max
        self.x_min = self.z_min / 2.0
        self.x_max = -self.x_min
        self.x_step = (self.x_max - self.x_min) / self.window_width
        self.z_step = abs(self.z_min) / (self.window_height - self.camera_offset)
        self.tracklets = [] 
    
        self.viewer3d = viewer3d
        self.saved_polygon = []
        self.saved_polygon_state = self.viewer3d.saved_polygon_state

    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Здесь логика обработки нажатия кнопки мыши
            print(f"Mouse Click at ({x}, {y})")
    
    def toggle_polygon_mode(self):
        if self.viewer3d.is_polygon_dynamic:
            if config.polygon:
                self.saved_polygon_state = config.polygon_cv
            else:    
                self.saved_polygon_state = [self.to_cv_point(point, self.current_camera_pose) for point in self.saved_polygon]            
        else:
            self.saved_polygon_state = None

    def generate_view(self, objects, current_camera_pose, tracking_view, tracking_enabled):
        for obj in objects.object_list:
            pos = sl.Translation(obj.position)
            temp = (pos * current_camera_pose.get_orientation()).get()
            current_translation = current_camera_pose.get_translation().get()
            obj.position = 1000*np.array([temp[0]+current_translation[0], temp[1]+current_translation[1], temp[2]+current_translation[2]])

        if not self.has_background_ready:
            self.generateBackground()
        tracking_view[:] = self.background
        # self.drawScale(tracking_view)
        
        self.draw_track_lines(tracking_view, current_camera_pose, objects)

        if tracking_enabled:
            current_timestamp = objects.timestamp.get_nanoseconds()
            self.addToTracklets(objects)
            self.detectUnchangedTrack(current_timestamp)
            self.pruneOldPoints(current_timestamp)
            self.drawTracklets(tracking_view, current_camera_pose, objects)

            self.draw_track_lines(tracking_view, current_camera_pose, objects)
        else:
            pass
            # self.drawPosition(objects, tracking_view, current_camera_pose)

    def draw_track_lines(self, tracking_view, current_camera_pose, objects):
        self.current_camera_pose = current_camera_pose  # Обновляем текущее положение камеры

        # Определение цвета линии на основе условия
        color = (0, 255, 0) if self.viewer3d.all_persons_inside else (0, 0, 255)

        # Рисование полигона для статического режима
        if not self.viewer3d.is_polygon_dynamic and self.saved_polygon_state:
            for start, end in zip(self.saved_polygon_state, self.saved_polygon_state[1:] + [self.saved_polygon_state[0]]):
                cv2.line(tracking_view, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, 2)

        # Рисование полигона для динамического режима
        if self.viewer3d.is_polygon_dynamic:
            self.saved_polygon.clear()  # Очистка предыдущего состояния полигона
            # Фильтрация и сортировка точек
            filtered_last_points = self.filter_and_sort_points(objects)
            if len(filtered_last_points) >= 3:
                cv_points = list(map(lambda point: self.to_cv_point(point, current_camera_pose), filtered_last_points))
                for start, end in zip(cv_points, cv_points[1:] + [cv_points[0]]):
                    cv2.line(tracking_view, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, 2)
                self.saved_polygon = filtered_last_points  # Сохранение текущего состояния полигона

    def filter_and_sort_points(self, objects):
        # Фильтрация последних точек треков, соответствующих определенным объектам
        filtered_last_points = [track.positions[-1].get_xyz() for track in self.tracklets 
                                if track.positions and any(obj.id == track.id and obj.raw_label == 1 for obj in objects.object_list)]
        # Вычисление центра и сортировка точек по углу относительно центра
        if filtered_last_points:
            center = np.mean(filtered_last_points, axis=0)
            return sorted(filtered_last_points, key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))
        return []
    def drawTracklets(self, tracking_view, current_camera_pose, objects):
        
        for track in self.tracklets:
            corresponding_obj = next((obj for obj in objects.object_list if obj.id == track.id), None)
            if corresponding_obj:
                # Определение цвета в зависимости от положения человека и raw_label
                if corresponding_obj.raw_label == 0:
                    is_inside = self.viewer3d.person_status.get(corresponding_obj.id, False)
                    clr = (102, 205, 105, 255) if is_inside else (0, 0, 255, 255)
                else:
                    clr = (0, 255, 255, 255) if self.viewer3d.is_polygon_dynamic or not self.saved_polygon_state else (255, 0, 0, 255)

                # Отрисовка точек и линий
                try:
                    cv_start_point = self.to_cv_point(track.positions[0].get_xyz(), current_camera_pose)
                except IndexError as e:
                    print(f"Error drawing tracklet: {e}")

                for position in track.positions[1:]:
                    cv_end_point = self.to_cv_point(position.get_xyz(), current_camera_pose)
                    cv2.line(tracking_view, (int(cv_start_point[0]), int(cv_start_point[1])), 
                            (int(cv_end_point[0]), int(cv_end_point[1])), clr, 3)
                    cv_start_point = cv_end_point   
                cv2.circle(tracking_view, (int(cv_start_point[0]), int(cv_start_point[1])), 6, clr, -1)

        
    def to_cv_point(self, x, z):
        out = []
        if isinstance(x, float) and isinstance(z, float):
            if np.isnan(x) or np.isnan(z):
                print(f"Предупреждение: NaN обнаружен в координатах ({x}, {z})")
                return None
            out = [int((x - self.x_min) / self.x_step), int((z - self.z_min) / self.z_step)]
        elif isinstance(x, np.ndarray) and isinstance(z, sl.Pose):
            # Go to camera current pose
            rotation = z.get_rotation_matrix()
            rotation.inverse()
            tmp = x - (z.get_translation() * rotation.get_orientation()).get()
            tmpx = (tmp[0] - self.x_min)/self.x_step+0.5
            tmpz = (tmp[2] - self.z_min)/self.z_step+0.5
            if np.isnan(tmpx) or np.isnan(tmpz):
                print(f"Предупреждение: NaN обнаружен в преобразованных координатах ({tmpx}, {tmpz})")
                return None  # Возвращаем None, чтобы указать на ошибку
            out = np.array([int(tmpx),int(tmpz)])
            
        elif isinstance(x, TrackPoint) and isinstance(z, sl.Pose):
            pos = x.get_xyz()
            out = self.to_cv_point(pos, z)
        else:
            return None
        return out



    def addToTracklets(self, objects):
        current_timestamp = objects.timestamp.get_nanoseconds()
        for obj in objects.object_list:
            if obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK or not math.isfinite(obj.position[0]):
                continue

            new_object = True
            for track in self.tracklets:

                if track.id == obj.id and track.is_alive:
                    new_object = False
                    track.addDetectedPoint(obj, current_timestamp, self.smoothing_window_size)

            if new_object:
                new_track = Tracklet(obj, obj.label, current_timestamp)
                self.tracklets.append(new_track)

    def detectUnchangedTrack(self, current_timestamp):
        for track in self.tracklets:
            if track.last_detected_timestamp < current_timestamp and track.last_detected_timestamp > 0:
                if current_timestamp - track.last_detected_timestamp >= self.history_duration:
                    track.is_alive = False

    def pruneOldPoints(self, ts):
        track_to_delete = []
        for i, track in enumerate(self.tracklets):
            if track.is_alive:
                while len(track.positions) > 0 and track.positions[0].timestamp < ts - self.history_duration:
                    track.positions.pop(0)
                while len(track.positions_to_draw) > 0 and track.positions_to_draw[0].timestamp < ts - self.history_duration:
                    track.positions_to_draw.pop(0)
            else:
                for _ in range(4):
                    if len(track.positions) > 0:
                        track.positions.pop(0)
                    if len(track.positions_to_draw) > 0:
                        track.positions_to_draw.pop(0)
                    else:
                        track_to_delete.append(i)
                        break

        for i in reversed(track_to_delete):
            self.tracklets.pop(i)
                    
    def generateBackground(self):
        self.drawCamera()
        # self.drawHotkeys()
        self.has_background_ready = True




    def drawCamera(self):
        # Configuration
        camera_color = (255, 217, 44, 255)

        camera_size = 10
        camera_height = self.window_height - self.camera_offset
        camera_left_pt = (self.window_width // 2 - camera_size // 2, camera_height)
        camera_right_pt = (self.window_width // 2 + camera_size // 2, camera_height)

        # Drawing camera
        camera_pts = [
            (self.window_width // 2 - camera_size, camera_height),
            (self.window_width // 2 + camera_size, camera_height),
            (self.window_width // 2 + camera_size, camera_height + camera_size // 2),
            (self.window_width // 2 - camera_size, camera_height + camera_size // 2)
        ]
        cv2.fillConvexPoly(self.background, np.array(camera_pts), camera_color)

        # Compute the FOV
        if self.fov < 0.0:
            self.computeFOV()

        # Get FOV intersection with window borders
        z_at_x_max = self.x_max / math.tan(self.fov / 2.0)
        left_intersection_pt = self.to_cv_point(self.x_min, -z_at_x_max)
        right_intersection_pt = self.to_cv_point(self.x_max, -z_at_x_max)

        clr = np.array(camera_color, dtype=np.uint8)
        
        # Draw FOV
        # Second try: dotted line
        left_line_it = createLineIterator(self.background, tuple(camera_left_pt), tuple(left_intersection_pt))
        for i, point in enumerate(left_line_it):
            current_pos = point
            if i % 5 == 0 or i % 5 == 1:
                self.background[current_pos[1], current_pos[0]] = clr

            for r in range(current_pos[1]):
                ratio = float(r) / camera_height
                self.background[r, current_pos[0]] = applyFading(self.background_color, ratio, self.fov_color)

        right_line_points = createLineIterator(self.background, tuple(camera_right_pt), tuple(right_intersection_pt))
        for i, point in enumerate(right_line_points):
            
            current_pos = point
            if i % 5 == 0 or i % 5 == 1:
                self.background[current_pos[1], current_pos[0]] = clr

            for r in range(current_pos[1]):
                ratio = float(r) / camera_height
                self.background[r, current_pos[0]] = applyFading(self.background_color, ratio, self.fov_color)

        for c in range(self.window_width // 2 - camera_size // 2, self.window_width // 2 + camera_size // 2 + 1):
            for r in range(camera_height):
                ratio = float(r) / camera_height
                self.background[r, c] = applyFading(self.background_color, ratio, self.fov_color)
    
    def computeFOV(self):
        image_size = self.camera_calibration.left_cam.image_size
        fx = self.camera_calibration.left_cam.fx
        self.fov = 2.0 * math.atan(image_size.width / (2.0 * fx))

    def set_camera_calibration(self,calib):
        self.camera_calibration = calib 
        self.has_background_ready = False 

    def zoom(self, factor):
        self.x_min *= factor
        self.x_max *= factor
        self.z_min *= factor
        self.x_step = (self.x_max - self.x_min) / self.window_width
        self.z_step = abs(self.z_min) / (self.window_height - self.camera_offset)

    def zoomIn(self):
        self.zoom(0.9)

    def zoomOut(self):
        self.zoom(1.0 / 0.9)
                
    
    # def drawHotkeys(self):
    #     hotkeys_clr = (0, 0, 0, 255)
    #     cv2.putText(self.background, "Press 'i' to zoom in", (25, self.window_height - 25), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1.0, hotkeys_clr, 1)
    #     cv2.putText(self.background, "Press 'o' to zoom out", (25, self.window_height - 15), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1.0, hotkeys_clr, 1)
    

    # def drawScale(self, tracking_view):
    #     x_step = 0.1  # Replace with your actual value
    #     one_meter_horizontal = int(1000 / x_step + 0.5)
    #     st_pt = (25, self.window_height - 50)
    #     end_pt = (25 + one_meter_horizontal, self.window_height - 50)
    #     thickness = 1
    #     # Scale line
    #     cv2.line(tracking_view, st_pt, end_pt, (220, 220, 0, 255), thickness)
    #     # Add ticks
    #     cv2.line(tracking_view, tuple(np.add(st_pt, [0, -3])), tuple(np.add(st_pt, [0, 3])), (220, 220, 0, 255), thickness)
    #     # Scale text
    #     cv2.putText(tracking_view, "1m", tuple(np.add(end_pt, [5, 5])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0, 255), 1)
    
    
def renderObject(i,isTrackingON):
	if isTrackingON:
		return i.tracking_state == sl.OBJECT_TRACKING_STATE.OK
	else:
		return i.tracking_state == sl.OBJECT_TRACKING_STATE.OK or i.tracking_state == sl.OBJECT_TRACKING_STATE.OFF


def cvt(pt,scale):
    return np.array([int(pt[0]*scale[0]),int(pt[1]*scale[1])])

def drawVerticalLine(display,start_pt,end_pt,clr,thickness):
    n_steps = 7
    pt1x = ((n_steps -1) * start_pt[0] + end_pt[0]) / n_steps 
    pt1y = ((n_steps -1) * start_pt[1] + end_pt[1]) / n_steps 
    pt4x = (start_pt[0] + (n_steps-1)*end_pt[0])/n_steps
    pt4y = (start_pt[1] + (n_steps-1)*end_pt[1])/n_steps
    pt1 = np.array([int(pt1x),int(pt1y)])
    pt4 = np.array([int(pt4x),int(pt4y)])
    cv2.line(display,start_pt,pt1,clr,thickness)
    cv2.line(display,pt4,end_pt,clr,thickness)


def getImagePosition(bounding_box_image, img_scale):
    out_position = np.zeros(2)
    out_position[0] = (bounding_box_image[0][0] + (bounding_box_image[2][0] - bounding_box_image[0][0])*0.5) * img_scale[0]
    out_position[1] = (bounding_box_image[0][1] + (bounding_box_image[2][1] - bounding_box_image[0][1])*0.5) * img_scale[1]
    return out_position


def LineIterator(img,P1, P2):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer

def _applyFading(val, current_alpha, current_clr):
    return int(current_alpha * current_clr + (1.0 - current_alpha) * val)

def applyFading(val, current_alpha, current_clr):
    out = np.empty(4, dtype=np.uint8)
    out[0] = _applyFading(val[0], current_alpha, current_clr[0])
    out[1] = _applyFading(val[1], current_alpha, current_clr[1])
    out[2] = _applyFading(val[2], current_alpha, current_clr[2])
    out[3] = 255
    return out


def createLineIterator(img, P1, P2):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),2),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX/dY
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)) + P1X
        else:
            slope = dY/dX
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray

    return itbuffer.astype(int)