import cv2
import numpy as np
from configs import config

class ImageOverlay:
    def __init__(self):
        self.green_logo_path = config.green_logo_path
        self.red_logo_path = config.red_logo_path
        self.yellow_logo_path = config.yellow_logo_path
        self.media_logo_path = config.media_logo_path
        
        self.logo_normal = cv2.imread(self.green_logo_path, cv2.IMREAD_UNCHANGED)
        self.logo_trevoga = cv2.imread(self.red_logo_path, cv2.IMREAD_UNCHANGED)
        self.logo_settings = cv2.imread(self.yellow_logo_path, cv2.IMREAD_UNCHANGED)
        self.media_techtrans = cv2.imread(self.media_logo_path, cv2.IMREAD_UNCHANGED)

        # Ensure logos have an alpha channel
        self.logo_normal = self._add_alpha_channel(self.logo_normal)
        self.logo_trevoga = self._add_alpha_channel(self.logo_trevoga)
        self.logo_settings = self._add_alpha_channel(self.logo_settings)
        self.media_techtrans = self._make_black_transparent(self.media_techtrans)

    def _make_black_transparent(self, image):
        """
        Convert black pixels to transparent in the image.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image_rgba[:, :, 3] = mask

        return image_rgba

    def _add_alpha_channel(self, image):
        """
        Add an alpha channel to the image if it doesn't have one.
        """
        if image.shape[2] == 3:
            alpha_channel = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
            return np.concatenate((image, alpha_channel), axis=-1)

        return image

    def overlay_logo_and_text(self, frame, viewer3d):
        """
        Overlay logo and text on the frame based on viewer conditions.
        """
        if viewer3d.all_persons_inside and not viewer3d.is_polygon_dynamic:
            logo = self.logo_normal
            logo_scale = 0.2
            text = "НОРМА"
            font_color = (0, 255, 0)
        elif viewer3d.is_polygon_dynamic or not viewer3d.current_polygon:
            logo = self.logo_settings
            logo_scale = 0.2
            text = "НАСТРОЙКА..."
            font_color =(0, 255, 255)
        else:
            logo = self.logo_trevoga
            logo_scale = 0.2
            text = "ТРЕВОГА"
            font_color = (0, 0, 255)

        logo_resized = cv2.resize(logo, (0, 0), fx=logo_scale, fy=logo_scale)
        logo_x_offset = 280
        logo_y_offset = 5
        
        if text == 'НАСТРОЙКА...':
            self._apply_transparency(frame, logo_resized, logo_x_offset+80, logo_y_offset)
        else:
            self._apply_transparency(frame, logo_resized, logo_x_offset, logo_y_offset)
            
        text_x_offset = logo_resized.shape[1] - 175 
        text_y_offset = 82
        
        if text == 'НАСТРОЙКА...': 
            cv2.putText(frame, text, (text_x_offset-60, text_y_offset), cv2.FONT_HERSHEY_COMPLEX, 2, font_color, 2)
        else:
            cv2.putText(frame, text, (text_x_offset, text_y_offset), cv2.FONT_HERSHEY_COMPLEX, 2, font_color, 2)
        
        media_logo_resized = cv2.resize(self.media_techtrans, (0, 0), fx=0.2, fy=0.2)
        media_logo_x_offset = frame.shape[1] - media_logo_resized.shape[1] -20 
        media_logo_y_offset = 10
        self._apply_transparency(frame, media_logo_resized, media_logo_x_offset, media_logo_y_offset)

        return frame

    def _apply_transparency(self, frame, overlay, x_offset, y_offset):
        """
        Apply transparency to overlay on the frame.
        """
        y1, y2 = y_offset, y_offset + overlay.shape[0]
        x1, x2 = x_offset, x_offset + overlay.shape[1]
        alpha_s = overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * overlay[:, :, c] +
                                      alpha_l * frame[y1:y2, x1:x2, c])
