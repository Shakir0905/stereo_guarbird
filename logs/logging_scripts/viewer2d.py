#viewer2d.py

# ----------------------------------------------------------------------
#       2D LEFT VIEW
# ----------------------------------------------------------------------

import cv2
import pyzed.sl as sl
import numpy as np
from configs import config

class CVViewer:
    def __init__(self):
        self.dynamic_polygon = []  # Динамически обновляемый полигон
        self.static_polygon = []  # Статический полигон, используется после фиксации
        self.use_static_polygon = False  # Флаг, указывающий, какой полигон использовать
        self.static_polygon_visible = False

    def toggle_polygon_mode(self):
        self.use_static_polygon = not self.use_static_polygon
        self.static_polygon_visible = self.use_static_polygon
        if self.use_static_polygon:
            if config.polygon:
            # Сохраняем текущий динамический полигон в статический
                self.static_polygon = config.polygon_gl
            else:
                self.static_polygon = self.dynamic_polygon.copy()

        else:
            # Восстанавливаем сохраненный статический полигон в динамический
            self.dynamic_polygon = self.static_polygon.copy()
            self.static_polygon = []
            self.update_dynamic_polygon(self.static_polygon)
    
    def update_dynamic_polygon(self, new_polygon):
        # Обновление динамического полигона, если не используется статический
        if not self.use_static_polygon:
            self.dynamic_polygon = new_polygon       
        
    @staticmethod
    def render_object(object_data, is_tracking_on):
        return object_data.tracking_state in [sl.OBJECT_TRACKING_STATE.OK, sl.OBJECT_TRACKING_STATE.OFF] if not is_tracking_on else object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK

    @staticmethod
    def cvt(pt, scale):
        return [pt[0] * scale[0], pt[1] * scale[1]]

    @staticmethod
    def draw_vertical_line(left_display, start_pt, end_pt, clr, thickness):
        n_steps = 7
        pt1 = [((n_steps - 1) * start_pt[0] + end_pt[0]) / n_steps, ((n_steps - 1) * start_pt[1] + end_pt[1]) / n_steps]
        pt4 = [(start_pt[0] + (n_steps - 1) * end_pt[0]) / n_steps, (start_pt[1] + (n_steps - 1) * end_pt[1]) / n_steps]
        cv2.line(left_display, (int(start_pt[0]), int(start_pt[1])), (int(pt1[0]), int(pt1[1])), clr, thickness)
        cv2.line(left_display, (int(pt4[0]), int(pt4[1])), (int(end_pt[0]), int(end_pt[1])), clr, thickness)

    @staticmethod
    def get_image_position(bounding_box_image, img_scale):
        out_position = np.zeros(2)
        out_position[0] = (bounding_box_image[0][0] + (bounding_box_image[2][0] - bounding_box_image[0][0]) * 0.5) * img_scale[0]
        out_position[1] = (bounding_box_image[0][1] + (bounding_box_image[2][1] - bounding_box_image[0][1]) * 0.5) * img_scale[1]
        return out_position

    def render_objects(self, left_display, img_scale, objects, is_tracking_on, viewer3d):
        line_thickness = 2
        overlay = left_display.copy()
        
        for obj in objects.object_list:
            if obj.raw_label == 1:
                base_color = [0, 255, 255, 255] if viewer3d.is_polygon_dynamic or not viewer3d.current_polygon else [255, 0, 0, 255]  # Добавляем альфа-канал к цвету
            else:
                base_color = [0, 255, 0, 255] if viewer3d.person_status.get(obj.id, False) else [0, 0, 255, 255]  # Добавляем альфа-канал

            if self.render_object(obj, is_tracking_on):
                self.draw_object(left_display, img_scale, obj, base_color, line_thickness, overlay)

        return overlay

    def draw_object(self, left_display, img_scale, obj, base_color, line_thickness, overlay):
        top_left_corner = self.cvt(obj.bounding_box_2d[0], img_scale)
        top_right_corner = self.cvt(obj.bounding_box_2d[1], img_scale)
        bottom_right_corner = self.cvt(obj.bounding_box_2d[2], img_scale)
        bottom_left_corner = self.cvt(obj.bounding_box_2d[3], img_scale)

        cv2.line(left_display, (int(top_left_corner[0]), int(top_left_corner[1])), (int(top_right_corner[0]), int(top_right_corner[1])), base_color, line_thickness)
        cv2.line(left_display, (int(bottom_left_corner[0]), int(bottom_left_corner[1])), (int(bottom_right_corner[0]), int(bottom_right_corner[1])), base_color, line_thickness)
        self.draw_vertical_line(left_display, bottom_left_corner, top_left_corner, base_color, line_thickness)
        self.draw_vertical_line(left_display, bottom_right_corner, top_right_corner, base_color, line_thickness)

        roi_height = int(top_right_corner[0] - top_left_corner[0])
        roi_width = int(bottom_left_corner[1] - top_left_corner[1])
        overlay_roi = overlay[int(top_left_corner[1]):int(top_left_corner[1] + roi_width),
                        int(top_left_corner[0]):int(top_left_corner[0] + roi_height)]

        for i in range(3):
            overlay_roi[..., i] = overlay_roi[..., i] * (1 - base_color[3]/255.0) + base_color[i] * (base_color[3]/255.0)
        
        position_image = self.get_image_position(obj.bounding_box_2d, img_scale)
        # Отображение класса объекта
        cv2.putText(left_display, "Id " + str(obj.id), 
                    (int(position_image[0] - 20), int(position_image[1] - 12)), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)

        # Отображение расстояния, если оно определено
        if np.isfinite(obj.position[2]):
            cv2.putText(left_display, f"{round(abs(obj.position[2]), 2)} m", 
                        (int(position_image[0] - 20), int(position_image[1])), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)

        # Отображение уверенности (confidence)
        confidence = round(obj.confidence / 100, 1)  # Преобразование в диапазон от 0 до 1 и округление до одного десятичного знака
        cv2.putText(left_display, f"conf: {confidence}", 
                    (int(position_image[0] - 20), int(position_image[1] + 12)),  # Смещаем ниже на 12 пикселей от позиции расстояния
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)

    def draw_polygon(self, left_display, img_scale, objects, viewer3d):
        line_thickness = 2
        overlay = left_display.copy()

        if not self.use_static_polygon:
            self.update_dynamic_polygon(self.filter_objects(objects))  # Обновляем динамический полигон

        polygon_to_use = self.static_polygon if self.use_static_polygon else self.filter_objects(objects)
        
        if len(polygon_to_use) > 2:
            sorted_objects = self.sort_objects(polygon_to_use)
            self.draw_polygon_lines(left_display, sorted_objects, img_scale, line_thickness, viewer3d)

        return overlay
    
    @staticmethod
    def filter_objects(objects):
        return [obj for obj in objects.object_list if obj.raw_label == 1]

    @staticmethod
    def sort_objects(filtered_objects):
        try:
            center = np.mean([[obj.position[0], obj.position[1]] for obj in filtered_objects], axis=0)
        except ValueError:
            print("ERRor")
        return sorted(filtered_objects, key=lambda obj: np.arctan2(obj.position[1] - center[1], obj.position[0] - center[0]))
    
    @staticmethod
    def draw_polygon_lines(left_display, sorted_objects, img_scale, line_thickness, viewer3d):
        base_color = [0, 255, 0, 1] if viewer3d.all_persons_inside else [0, 0, 255, 1]
        if config.polygon:
            for i in range(len(config.STATIC_POLYGON_COORDS) - 1):
                start_point, end_point = config.STATIC_POLYGON_COORDS[i], config.STATIC_POLYGON_COORDS[i + 1]
                cv2.line(left_display, start_point, end_point, base_color, line_thickness)
        else:    
            for i in range(len(sorted_objects)):
                current_obj = sorted_objects[i]
                next_obj = sorted_objects[(i + 1) % len(sorted_objects)]  # Это гарантирует замыкание полигона

                # Получаем позиции для текущего и следующего объекта
                current_position = CVViewer.get_image_position(current_obj.bounding_box_2d, img_scale)
                next_position = CVViewer.get_image_position(next_obj.bounding_box_2d, img_scale)

                # Изменяем координаты Y, чтобы линия была ниже (ближе к нижней части bounding box'а)
                # Например, увеличиваем Y на некоторую долю высоты bounding box'а
                current_box_height = abs(current_obj.bounding_box_2d[0][1] - current_obj.bounding_box_2d[2][1]) * img_scale[1]
                next_box_height = abs(next_obj.bounding_box_2d[0][1] - next_obj.bounding_box_2d[2][1]) * img_scale[1]
                
                # Увеличиваем Y, чтобы сдвинуть линию ниже. Коэффициент (например, 0.8) можно настроить по необходимости.
                adjusted_current_position_y = current_position[1] + current_box_height * 0.1
                adjusted_next_position_y = next_position[1] + next_box_height * 0.1
                

                start_point = (int(current_position[0]), int(adjusted_current_position_y))
                end_point = (int(next_position[0]), int(adjusted_next_position_y))
                cv2.line(left_display, start_point, end_point, base_color, line_thickness)

    def render_2D(self, left_display, img_scale, objects, is_tracking_on, viewer3d):
        overlay_objects = self.render_objects(left_display, img_scale, objects, is_tracking_on, viewer3d)
        overlay_polygon = self.draw_polygon(left_display, img_scale, objects, viewer3d)
        cv2.addWeighted(left_display, 0.7, overlay_objects, 0.3, 0.0, left_display)
        cv2.addWeighted(left_display, 0.7, overlay_polygon, 0.3, 0.0, left_display)