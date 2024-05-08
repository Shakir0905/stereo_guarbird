from logs.logging_scripts.logger_config import setup_video_logger
from datetime import datetime, timedelta
import shutil
import atexit
import cv2
import os
from pathlib import Path

class VideoLogger:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent  # Автоматическое определение базового пути проекта
        self.out = None
        self.logger = setup_video_logger()
        self.start_time = datetime.now()
        self.initializing = False
        atexit.register(self.at_exit)
        
    def get_current_paths(self):
        """Динамическое формирование и обновление путей на основе текущей даты."""
        current_date = datetime.now().strftime('%Y-%m-%d')
        video_logs_path = self.base_path / 'videos_logs' / f'video_{current_date}'
        videos_path = self.base_path / 'videos_logs'
        os.makedirs(video_logs_path, exist_ok=True)
        return video_logs_path, videos_path

    def initialize_video_writer(self):
        if self.initializing:
            return
        
        self.initializing = True
        if self.out is not None:
            self.out.release()
            cv2.destroyAllWindows()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        video_logs_path, _ = self.get_current_paths()
        filename = f"video_{timestamp}.mp4"
        output_path = os.path.join(video_logs_path, filename)
        self.out = cv2.VideoWriter(output_path, fourcc, 10.0, (1680, 770))
        
        self.start_time = datetime.now()
        self.initializing = False

    def record_video(self, image):
        if datetime.now() - self.start_time >= timedelta(minutes=10) and not self.initializing:
            self.initialize_video_writer()
        self.out.write(image)

    def stop_recording(self):
        if self.out is not None:
            self.out.release()
            cv2.destroyAllWindows()
            _, videos_path = self.get_current_paths()
            self.delete_old_dirs(videos_path)
            self.logger.info("Video recording stopped and archived.")

    def at_exit(self):
        self.stop_recording()

    def delete_old_dirs(self, directory_path, days=3):
        """Delete directories created more than 'days' days ago."""
        self.logger.info(f"Starting to delete old archives for {directory_path} with a threshold of {days} days.")
        now = datetime.now()
        cutoff_date = now - timedelta(days=days)
        for item_name in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item_name)
            if os.path.isdir(item_path):
                item_mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                if item_mtime < cutoff_date:
                    try:
                        shutil.rmtree(item_path)
                        self.logger.info(f"Directory {item_name} deleted.")
                    except Exception as e:
                        self.logger.error(f"Error deleting directory {item_name}: {e}")
                else:
                    self.logger.info(f"Directory {item_name} retained, as it is newer than the cutoff date.")
 