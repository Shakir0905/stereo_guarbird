"""! Проверка состояния камеры на основе BRIEF.

С помощью данного модуля можно проверить, находится ли камера в "хорошем состоянии", т.е.
не загрязнена или не заслонена ли камера.
Алгоритм сравнивает картинку с предыдущим состоянием. Если состояние резко изменилось, значит что-то пошло не так.
"""

import cv2
import numpy as np


class BRIEFChecker:
    """! Проверка состояния камеры.

    Вначале происходит инициализация базового изображения в методе init_base_img(). 
    Затем в методе __check_init происходит проверка, правильно ли инициализировалась камера.
    Основным методом является step(). В нём вызываются методы check и update при 
        "основной работе" и метод init_base_img при инициализации.
    """

    def __init__(self, threshold=0.5, size=(200, 200), verbose=1,
                 init_number=10, init_every=3, check_every=3, update_every=15,
                 init_matching_ratio=0.6, update_matching_count_ratio=0.4, update_base_img_ratio=0.8):
        """! Конструктор.
        @param threshold Порог - определяет, при каком проценте матчинга текущего изображения
                с base_img будет считаться, что камера замутнена/загорожена. Число от 0 до 1.
        @param size (ширина, высота) - размер, в который будет изменена входящая картинка.
                Если не нужно изменять входящую картинку, то можно указать -1 или None.
        @param verbose вывод программы 
                0 - ничего не выводить
                1 - выводить только текст
                2 - выводить изображения
                3 - выводить и текст, и изображения
        @param init_number Количество картинок нужных для инициализации.
        @param init_every Добавлять каждые n кадров картинку в стек для инициализации.
        @param check_every Проверять каждые n кадров, что картинка замутнена/загорожена.
        @param update_every Обновлять каждые n кадров base_img.
        @param init_matching_ratio Минимальный порог схожести инициализированной картинки 
                base_img с текущей картинкой. Число от 0 до 1.
        @param update_matching_count_ratio Соотношение текущего количества смэтченных точек с предыдущим
                количеством смэтченных точек при обновлении matching_count. Формула обновления matching_count 
                есть в методе  update. Является числом от 0 до 1.
        @param update_base_img_ratio Показывает какой процент новой "идеальной" картинки будет находиться в 
                base_img при её обновлении. Является числом от 0 до 1.
        """
        self.threshold = threshold
        self.size = size
        self.verbose = verbose

        self.init_number = init_number
        self.init_every = init_every
        self.check_every = check_every
        self.update_every = update_every

        self.fast = cv2.FastFeatureDetector_create()
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.base_img = []

        self.counter = 0
        self.initialized = False
        self.obscured = True

        self.frame_keypoints = []
        self.frame_descriptor = None
        self.base_keypoints = []
        self.base_descriptor = None

        # "Идеальное" количестов смэтченных точек
        self.matching_count = -1
        # Количество точек, сметченных последний раз.
        self.num_matches = 0

        self.init_matching_ratio = init_matching_ratio
        self.update_matching_count_ratio = update_matching_count_ratio
        self.update_base_img_ratio = update_base_img_ratio


    def __get_keypoints(self, image):
        """! Возвращает ключевые точки от BRIEF
        @param image изображение, с которого получить точки
        @return tuple(keypoints, descriptor)
        """
        keypoints = self.fast.detect(image, None)
        return self.brief.compute(image, keypoints)

    def __draw_output(self, frame):
        """! Рисует как выглядит base_img, и где расположены 
                BRIEF точки на текущем изображении и на base_img.
        @param frame Входящее изображение
        """
        base_img_copy = np.copy(self.base_img)
        base_keypoints_img = np.copy(self.base_img)
        text = "Is obscured" if self.obscured else "Good condition"
        cv2.putText(base_img_copy, text, (10, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.drawKeypoints(frame, self.frame_keypoints,
                          frame, flags=1, color=(255, 0, 0))
        cv2.drawKeypoints(base_keypoints_img, self.base_keypoints,
                          base_keypoints_img, flags=1, color=(255, 0, 0))
        cv2.imshow('Frame keypoints ' + self.camera_name, frame)
        cv2.imshow('Base keypoints ' + self.camera_name, base_keypoints_img)
        cv2.imshow('Base image ' + self.camera_name, base_img_copy)

    def __check_init(self, frame):
        """! Проверка, что инициализация base_img прошла верно.
        @param frame Входящее изображение
        """
        self.base_keypoints, self.base_descriptor = self.__get_keypoints(
            self.base_img)
        self.frame_keypoints, self.frame_descriptor = self.__get_keypoints(
            frame)

        # Проверка, что на изображениях есть хотя бы по точке.
        # Обычно такое происходит, если изображение полностью чёрное или однородное.
        if len(self.frame_keypoints) == 0 or len(self.base_keypoints) == 0:
            self.base_img = []
            return

        matches = self.matcher.match(
            self.base_descriptor, self.frame_descriptor)
        self.matching_count = len(matches)

        # Проверка, достаточно ли схоже base_img с последней картинкой.
        if self.matching_count < self.init_matching_ratio * \
                min(len(self.frame_keypoints), len(self.base_keypoints)):
            self.base_img = []
        else:
            # Инициализация прошла успешно
            self.obscured = False
            self.initialized = True

    def init_base_img(self, frame):
        """! Инициализация базового изображения, с которым будет происходить сравнение
        @param frame полученный кадр в grayscale
        """
        if self.counter % self.init_every == 0:
            self.base_img.append(frame)

            # Если набралось достаточное количество картинок в стеке.
            if len(self.base_img) == self.init_number:
                self.base_img = np.mean(
                    np.array(self.base_img), axis=0).astype("uint8")
                self.__check_init(frame)

    def check(self, frame):
        """! Проверка текущего изображения на замутненте/загорождение.
        @param frame текущий кадр в grayscale
        """
        self.frame_keypoints, self.frame_descriptor = \
            self.__get_keypoints(frame)

        # Проверка, что изображение не чёрное
        if len(self.frame_keypoints) > 0:
            matches = self.matcher.match(
                self.base_descriptor, self.frame_descriptor)
            self.num_matches = len(matches)
            self.obscured = self.num_matches/self.matching_count < self.threshold
        else:
            self.num_matches = 0
            self.obscured = True


    def update(self, frame):
        """! Обновление base_img.
        @param frame текущий кадр в grayscale
        """
        self.matching_count = self.update_matching_count_ratio * self.num_matches + \
            (1 - self.update_matching_count_ratio) * self.matching_count
        self.base_img = (self.update_base_img_ratio * frame + (1 -
                         self.update_base_img_ratio) * self.base_img).astype("uint8")
        self.base_keypoints, self.base_descriptor = \
            self.__get_keypoints(self.base_img)


    def step(self, frame:np.ndarray) -> bool:
        """! Основной метод класса
        @param frame полученный кадр
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.size != -1 and self.size is not None:
            frame = cv2.resize(frame, dsize=self.size)

        if not self.initialized:
            self.init_base_img(frame)
        else:
            # Check on obscureness
            if self.counter % self.check_every == 0:
                self.check(frame)

            # Update base image
            if not self.obscured and self.counter % self.update_every == 0:
                self.update(frame)

            if self.verbose & 2:
                self.__draw_output(np.copy(frame))

        self.counter += 1

        return self.obscured

# Пример взаимодействия 


"""

if __name__ == "__main__":
    x = BRIEFChecker()
    cap = cv2.VideoCapture(0) # вебку включи

    while True:
        ret, img = cap.read()

        is_obscured = x.step(img)

        print(is_obscured) 

        if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()


"""