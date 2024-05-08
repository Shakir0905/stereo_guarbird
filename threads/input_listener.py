import readchar

class InputListener:
    """
    Класс для прослушивания ввода с клавиатуры.
    """
    def __init__(self, detector, mode_controller):
        self.detector = detector
        self.mode_controller = mode_controller

    def input_listening(self):
        """
        Начинает прослушивание ввода с клавиатуры.
        """
        print("Нажмите 'q' для выхода, '1' для переключения режимов.")
        while True:
            key = readchar.readkey()
            if key == '1':
                self.mode_controller.toggle_all_modes()
            elif key.lower() == 'q':
                self.detector.exit_signal = True
                break
