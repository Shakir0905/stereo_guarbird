from threading import Thread


def start_threads(detector, audio_player, input_listener):
    """
    Инициализация и запуск всех необходимых потоков.
    """

    threads = [
        Thread(target=detector.torch_thread, name="CaptureThread"),
        Thread(target=audio_player.play, daemon=True, name="AudioThread"),
        Thread(target=input_listener.input_listening, daemon=True, name="InputThread")
    ]
    
    for thread in threads:
        thread.start()
