# logger_config.py
import logging
from pathlib import Path
from datetime import datetime
# Автоматическое определение базового пути проекта
base_path = Path(__file__).parent.parent

def setup_video_logger(name="VideoLogger"):
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_filename = Path(str(base_path / 'logs' / 'text_logs' / f"video_logger_{current_date}.log"))
     
    if not log_filename.parent.exists():
        log_filename.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Установка уровня логирования
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Создание и установка формата логирования
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # Добавление обработчика к логгеру
    logger.addHandler(file_handler)

    # Для предотвращения дублирования сообщений в случае повторной настройки
    logger.propagate = False

    return logger
