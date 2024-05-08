import pygame
from time import sleep

class AudioPlayer:
    def __init__(self):
        self.count_conuses = 0
        self.all_persons_inside = False
        self.is_polygon_dynamic = True
        self.first_sound = True  # Установка начального состояния для флага первого звука    
        pygame.mixer.init()

    def play_turn_on_voice(self):
        """Загрузка и воспроизведение приветственного голосового сообщения."""
        sound_to_play = "resourses/audio/turn_on.mp3"
        pygame.mixer.music.load(sound_to_play)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Ожидание окончания воспроизведения
            sleep(0.01)

    def play(self):
        """Основной метод воспроизведения звуков в зависимости от состояния."""
        self.play_turn_on_voice()

        while True:
            # Определение, какой звук следует загрузить и воспроизвести
            if self.is_polygon_dynamic:
                sound_to_play = "resourses/audio/perimeter_init.mp3"
                self.first_sound = True

                pygame.mixer.music.load(sound_to_play)
                pygame.mixer.music.play()
                # Ожидание, пока звук воспроизводится, прежде чем перезагружать или останавливать
                while pygame.mixer.music.get_busy():
                        sleep(5)
            elif not self.is_polygon_dynamic and self.first_sound:
                if self.count_conuses:
                    sound_to_play = f"resourses/audio/count_conuses/{self.count_conuses}.mp3"

                    pygame.mixer.music.load(sound_to_play)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                            sleep(0.01)    
                self.first_sound = False  
            else:
                if not self.all_persons_inside:
                    sound_to_play = f"resourses/audio/alarm.mp3"

                    pygame.mixer.music.load(sound_to_play)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                            sleep(4)
                else:
                    sound_to_play = f"resourses/audio/pass.mp3"
                    pygame.mixer.music.load(sound_to_play)
                    pygame.mixer.music.play()
                    