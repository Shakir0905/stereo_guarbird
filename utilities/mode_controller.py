class ModeController:
    def __init__(self, track_view_gen, viewer_2d, viewer_3d):
        self.track_view_gen = track_view_gen
        self.viewer_2d = viewer_2d
        self.viewer_3d = viewer_3d

    def toggle_all_modes(self):
        self.track_view_gen.toggle_polygon_mode()
        self.viewer_2d.toggle_polygon_mode()
        self.viewer_3d.toggle_polygon_mode()
        print("Режимы переключены")
