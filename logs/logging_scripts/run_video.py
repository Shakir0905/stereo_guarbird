import os
import cv2
import sys

sys.path.append('/home/ramazanov/stereo-guardbird')

from pathlib import Path
from datetime import datetime
base_path = Path(__file__).parent.parent

current_date = datetime.now().strftime('%Y-%m-%d')

def play_latest_video():
    log_dir =  str(base_path / 'videos_logs' / f'video_{current_date}')

    video_path = max((os.path.join(log_dir, f) for f in os.listdir(log_dir)), key=os.path.getmtime)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл.")
        return

    print("Видео успешно открыто.")

    # Получите размеры экрана
    screen_width, screen_height = 1920, 1080  # Замените значения на разрешение вашего экрана

    # Установите положение окна вывода
    cv2.namedWindow('Video Playback', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Playback', screen_width, screen_height)
    cv2.moveWindow('Video Playback', 0, 0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Ошибка: Не удалось прочитать кадр. Возможно, видео закончилось.")
            break

        cv2.imshow('Video Playback', frame)

        # Задержка в 30 миллисекунд между кадрами
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Видео закрыто и ресурсы освобождены.")
    
# Пример использования функции с аргументом по умолчанию "video"
if __name__ == "__main__":
    play_latest_video()
