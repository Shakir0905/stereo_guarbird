from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

def concatenate_videos(directory):
    clips = []
    filenames = sorted([f for f in os.listdir(directory) if f.endswith(".mp4")])
    if not filenames:
        print("Видео для обработки не найдено.")
        return

    output_filename = filenames[0]  # Имя итогового файла как имя первого в списке
    output_path = os.path.join(directory, output_filename)  # Путь к итоговому файлу

    for filename in filenames:
        filepath = os.path.join(directory, filename)
        try:
            video_clip = VideoFileClip(filepath)
            if video_clip.duration > 0:  # Проверяем, что длительность видео больше 0
                clips.append(video_clip)
                print(f"Файл {filename} успешно добавлен. Длительность: {video_clip.duration} секунд.")
            else:
                print(f"Файл {filename} пропущен, так как его длительность нулевая или не может быть определена.")
        except Exception as e:
            print(f"Ошибка при добавлении файла {filename}: {e}")
    
    if clips:
        final_clip = concatenate_videoclips(clips, method="compose")
        # Генерация имени для итогового файла, чтобы избежать конфликта с исходным
        temp_output_path = os.path.join(directory, "temp_" + output_filename)
        final_clip.write_videofile(temp_output_path, codec="libx264", audio_codec="aac")
        print(f"Видео успешно склеены в файл: {temp_output_path}")

        # Удаление исходных файлов
        for filename in filenames:
            filepath = os.path.join(directory, filename)
            os.remove(filepath)
            print(f"Файл {filepath} удалён.")

        # Переименование итогового файла
        os.rename(temp_output_path, output_path)
        print(f"Итоговый файл переименован в: {output_path}")
    else:
        print("Видео для обработки не найдено.")

if __name__ == "__main__":
    current_directory = 'logs/videos_logs/video_2024-04-08'  # Путь к директории с видеофайлами
    concatenate_videos(current_directory)
