from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

from shapely.geometry import Point, Polygon
from shapely.errors import GEOSException

from ogl_viewer.zed_model import *
from threading import Lock
from configs import config
import pyzed.sl as sl

import numpy as np
import ctypes
import array
import math
import cv2

class Shader:
    def __init__(self, vs, fs):
        # Создание программы шейдеров и компиляция вершинного и фрагментного шейдеров
        self.program_id = glCreateProgram()
        vertex_id = self.compile(GL_VERTEX_SHADER, vs)  # Компиляция вершинного шейдера
        fragment_id = self.compile(GL_FRAGMENT_SHADER, fs)  # Компиляция фрагментного шейдера

        glAttachShader(self.program_id, vertex_id)  # Прикрепление вершинного шейдера к программе
        glAttachShader(self.program_id, fragment_id)  # Прикрепление фрагментного шейдера к программе
        glBindAttribLocation(self.program_id, 0, "in_vertex")  # Привязка атрибута in_vertex к индексу 0
        glBindAttribLocation(self.program_id, 1, "in_texCoord")  # Привязка атрибута in_texCoord к индексу 1
        glLinkProgram(self.program_id)  # Связывание программы шейдеров

        if glGetProgramiv(self.program_id, GL_LINK_STATUS) != GL_TRUE:
            # Проверка статуса связывания программы и вывод сообщения об ошибке при неудаче
            info = glGetProgramInfoLog(self.program_id)
            glDeleteProgram(self.program_id)
            glDeleteShader(vertex_id)
            glDeleteShader(fragment_id)
            raise RuntimeError('Ошибка при связывании программы: %s' % (info))
        glDeleteShader(vertex_id)
        glDeleteShader(fragment_id)

    def compile(self, shader_type, source):
        try:
            shader_id = glCreateShader(shader_type)  # Создание объекта шейдера указанного типа
            if shader_id == 0:
                print("ОШИБКА: Тип шейдера {0} не существует".format(shader_type))
                exit()

            glShaderSource(shader_id, source)  # Установка исходного кода шейдера
            glCompileShader(shader_id)  # Компиляция шейдера
            if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
                # Проверка статуса компиляции шейдера и вывод сообщения об ошибке при неудаче
                info = glGetShaderInfoLog(shader_id)
                glDeleteShader(shader_id)
                raise RuntimeError('Ошибка компиляции шейдера: %s' % (info))
            return shader_id
        except Exception:
            glDeleteShader(shader_id)
            raise

    def get_program_id(self):
        # Получение идентификатора программы шейдеров
        return self.program_id


class Simple3DObject:
    """
    Класс для управления простыми 3D объектами для рендеринга в OpenGL.
    """

    def __init__(self, _is_static):
        self.vaoID = 0                    # Идентификатор Vertex Array Object
        self.drawing_type = GL_TRIANGLES  # Тип отрисовки (треугольники)
        self.is_static = _is_static       # Статический ли объект
        self.elementbufferSize = 0        # Размер буфера элементов
        
        self.vertices = array.array('f')  # Массив вершин (float)
        self.colors = array.array('f')    # Массив цветов (float)
        self.normals = array.array('f')   # Массив нормалей (float)
        self.indices = array.array('I')   # Массив индексов (unsigned int)

    def add_line(self, p1, p2, clr):
        self.add_point_clr(p1, clr)
        self.add_point_clr(p2, clr)

    def add_point_clr(self, pt, clr):
        self.add_pt(pt)  
        self.add_clr(clr)  
        self.indices.append(len(self.indices))

    def add_pt(self, _pts):
        for pt in _pts:
            self.vertices.append(pt)  # Добавление каждой точки в массив вершин

    def add_clr(self, _clrs):
        for clr in _clrs:
            self.colors.append(clr)  # Добавление каждого цвета в массив цветов

    def add_full_edges(self, pts, clr):
        """
        Add full edges to the object.

        :param pts: List of points to create the edges.
        :param clr: Color of the edges.
        """
        start_id = int(len(self.vertices) / 3)
        clr[3] = 0.2

        for pt in pts:
            self.add_pt(pt)
            self.add_clr(clr)

        box_links_top = np.array([0, 1, 1, 2, 2, 3, 3, 0])
        i = 0
        while i < box_links_top.size:
            self.indices.append(start_id + box_links_top[i])
            self.indices.append(start_id + box_links_top[i+1])
            i += 2

        box_links_bottom = np.array([4, 5, 5, 6, 6, 7, 7, 4])
        i = 0
        while i < box_links_bottom.size:
            self.indices.append(start_id + box_links_bottom[i])
            self.indices.append(start_id + box_links_bottom[i+1])
            i += 2
    
    def add_vertical_edges(self, pts, clr):
        """
        Добавление вертикальных граней объекта.

        :param pts: Список точек для создания граней.
        :param clr: Цвет граней.
        """
        self.__add_single_vertical_line(pts[0], pts[4], clr)
        self.__add_single_vertical_line(pts[1], pts[5], clr)
        self.__add_single_vertical_line(pts[2], pts[6], clr)
        self.__add_single_vertical_line(pts[3], pts[7], clr)
    
    def __add_single_vertical_line(self, top_pt, bottom_pt, clr):
        """
        Добавление одной вертикальной линии в объект.

        :param top_pt: Верхняя точка линии.
        :param bottom_pt: Нижняя точка линии.
        :param clr: Цвет линии.
        """
        current_pts = np.array([
            top_pt,
            ((GRID_SIZE - 1) * np.array(top_pt) + np.array(bottom_pt)) / GRID_SIZE,
            ((GRID_SIZE - 2) * np.array(top_pt) + np.array(bottom_pt) * 2) / GRID_SIZE,
            (2 * np.array(top_pt) + np.array(bottom_pt) * (GRID_SIZE - 2)) / GRID_SIZE,
            (np.array(top_pt) + np.array(bottom_pt) * (GRID_SIZE - 1)) / GRID_SIZE,
            bottom_pt
        ], np.float32)
        start_id = int(len(self.vertices) / 3)
        for i in range(len(current_pts)):
            self.add_pt(current_pts[i])
            if i == 2 or i == 3:
                clr[3] = 0
            else:
                clr[3] = 0.2
            self.add_clr(clr)

        box_links = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        i = 0
        while i < box_links.size:
            self.indices.append(start_id + box_links[i])
            self.indices.append(start_id + box_links[i + 1])
            i += 2

    def add_vertical_faces(self, _pts, _clr):
        # Определение четырех граней (квадов) объекта. Каждый квад определяется четырьмя индексами точек.
        # Первые два индекса в каждом кваде - это верхние точки.
        quads = [
            [0, 3, 7, 4],  # Передняя грань
            [3, 2, 6, 7],  # Правая грань
            [2, 1, 5, 6],  # Задняя грань
            [1, 0, 4, 5]   # Левая грань
        ]

        alpha = 0.25  # Начальное значение прозрачности для граней

        # Создание граней с постепенно уменьшающейся прозрачностью
        for quad in quads:
            # Создание первого слоя грани
            quad_pts_1 = [
                _pts[quad[0]],
                _pts[quad[1]],
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[1]]) + 0.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[0]]) + 0.5 * np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_1, alpha, alpha, _clr)

            # Создание второго слоя грани с уменьшенной прозрачностью
            quad_pts_2 = [
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[0]]) + 0.5 * np.array(_pts[quad[3]])) / GRID_SIZE,
                ((GRID_SIZE - 0.5) * np.array(_pts[quad[1]]) + 0.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[1]]) + np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[0]]) + np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_2, alpha, 2 * alpha / 3, _clr)

            # Создание третьего слоя грани
            quad_pts_3 = [
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[0]]) + np.array(_pts[quad[3]])) / GRID_SIZE,
                ((GRID_SIZE - 1.0) * np.array(_pts[quad[1]]) + np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.5) * np.array(_pts[quad[1]]) + 1.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 1.5) * np.array(_pts[quad[0]]) + 1.5 * np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_3, 2 * alpha / 3, alpha / 3, _clr)

            # Создание четвертого слоя грани
            quad_pts_4 = [
                ((GRID_SIZE - 1.5) * np.array(_pts[quad[0]]) + 1.5 * np.array(_pts[quad[3]])) / GRID_SIZE,
                ((GRID_SIZE - 1.5) * np.array(_pts[quad[1]]) + 1.5 * np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 2.0) * np.array(_pts[quad[1]]) + 2.0 * np.array(_pts[quad[2]])) / GRID_SIZE,
                ((GRID_SIZE - 2.0) * np.array(_pts[quad[0]]) + 2.0 * np.array(_pts[quad[3]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_4, alpha / 3, 0.0, _clr)

            # Создание пятого слоя грани
            quad_pts_5 = [
                (np.array(_pts[quad[1]]) * 2.0 + (GRID_SIZE - 2.0) * np.array(_pts[quad[2]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 2.0 + (GRID_SIZE - 2.0) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[2]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_5, 0.0, alpha / 3, _clr)

            # Создание шестого слоя грани
            quad_pts_6 = [
                (np.array(_pts[quad[1]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[2]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 1.5 + (GRID_SIZE - 1.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[2]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_6, alpha / 3, 2 * alpha / 3, _clr)

            # Создание седьмого слоя грани
            quad_pts_7 = [
                (np.array(_pts[quad[1]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[2]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) + (GRID_SIZE - 1.0) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[0]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[2]])) / GRID_SIZE
            ]
            self.__add_quad(quad_pts_7, 2 * alpha / 3, alpha, _clr)

            # Создание восьмого и последнего слоя грани
            quad_pts_8 = [
                (np.array(_pts[quad[0]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[3]])) / GRID_SIZE,
                (np.array(_pts[quad[1]]) * 0.5 + (GRID_SIZE - 0.5) * np.array(_pts[quad[2]])) / GRID_SIZE,
                np.array(_pts[quad[2]]),
                np.array(_pts[quad[3]])
            ]
            self.__add_quad(quad_pts_8, alpha, alpha, _clr)
        
    def add_top_face(self, pts, clr):
        """
        Добавление верхней грани объекта.

        :param pts: Список точек для создания верхней грани.
        :param clr: Цвет верхней грани.
        """
        clr[3] = 0.25
        for pt in pts:
            self.add_point_clr(pt, clr)
   
    def __add_quad(self, quad_pts, alpha1, alpha2, clr):

        """
        Добавление четырехугольной грани к объекту.

        :param quad_pts: Список из четырех точек.
        :param alpha1: Значение альфа для первых двух точек.
        :param alpha2: Значение альфа для последних двух точек.
        :param clr: Цвет четырехугольной грани.
        """
        for i in range(len(quad_pts)):
            self.add_pt(quad_pts[i])
            if i < 2:
                clr[3] = alpha1
            else:
                clr[3] = alpha2
            self.add_clr(clr)

        self.indices.append(len(self.indices))
        self.indices.append(len(self.indices))
        self.indices.append(len(self.indices))
        self.indices.append(len(self.indices))

    def __del__(self):
        """
        Деструктор для освобождения ресурсов.
        """
        if self.vaoID:
            self.vaoID = 0  # Обнуление идентификатора VAO при удалении объекта

    def add_normal(self, _normals):
        """
        Добавление уникальной нормали в список нормалей.

        :param _normals: Список нормалей для добавления.
        """
        for normal in _normals:
            self.normals.append(normal)  # Добавление каждой нормали в массив нормалей

    def add_points(self, _pts, _base_clr):
        """
        Добавление набора точек в список точек и соответствующих цветов.

        :param _pts: Список точек.
        :param _base_clr: Базовый цвет для точек.
        """
        for i in range(len(_pts)):
            pt = _pts[i]
            self.add_pt([pt]) 
            self.add_clr([_base_clr])  
            current_size_index = (len(self.vertices) // 3) - 1
            self.indices.extend([current_size_index, current_size_index + 1])  # Добавление индексов

    def push_to_GPU(self):
        # Создание четырех буферов в GPU для хранения различных типов данных
        self.vboID = glGenBuffers(4)

        # Загрузка вершин в GPU
        if len(self.vertices):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glBufferData(GL_ARRAY_BUFFER, len(self.vertices) * self.vertices.itemsize, 
                         (GLfloat * len(self.vertices))(*self.vertices), GL_STATIC_DRAW)

        # Загрузка цветов в GPU
        if len(self.colors):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ARRAY_BUFFER, len(self.colors) * self.colors.itemsize, 
                         (GLfloat * len(self.colors))(*self.colors), GL_STATIC_DRAW)

        # Загрузка индексов в GPU
        if len(self.indices):
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.indices) * self.indices.itemsize, 
                         (GLuint * len(self.indices))(*self.indices), GL_STATIC_DRAW)

        # Загрузка нормалей в GPU
        if len(self.normals):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[3])
            glBufferData(GL_ARRAY_BUFFER, len(self.normals) * self.normals.itemsize, 
                         (GLfloat * len(self.normals))(*self.normals), GL_STATIC_DRAW)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0)
            glEnableVertexAttribArray(2)

        # Установка размера буфера элементов
        self.elementbufferSize = len(self.indices)

    def clear(self):
        # Очистка массивов данных
        self.vertices = array.array('f')
        self.colors = array.array('f')
        self.normals = array.array('f')
        self.indices = array.array('I')

    def set_drawing_type(self, _type):
        # Установка типа отрисовки (например, GL_TRIANGLES, GL_LINES)
        self.drawing_type = _type

    def draw(self):
        # Отрисовка объекта с использованием загруженных данных
        if self.elementbufferSize:
            # Активация и настройка атрибута вершин
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

            # Активация и настройка атрибута цвета
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)

            # Связывание буфера индексов и отрисовка элементов
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glDrawElements(self.drawing_type, self.elementbufferSize, GL_UNSIGNED_INT, None)

            # Отключение атрибутов массива после отрисовки
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)


class ImageHandler:
    """
    Класс для управления потоком изображений и их рендеринга с помощью OpenGL.
    """
    def __init__(self):
        self.tex_id = 0
        self.image_tex = 0
        self.quad_vb = 0
        self.is_called = 0

    def close(self):
        # Очистка текстуры изображения
        if self.image_tex:
            self.image_tex = 0

    def initialize(self, _res):
        # Инициализация шейдера и настройка текстуры
        self.shader_image = Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER)
        self.tex_id = glGetUniformLocation(self.shader_image.get_program_id(), "texImage")

        # Создание и заполнение вершинного буфера для квада
        g_quad_vertex_buffer_data = np.array([-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0], np.float32)
        self.quad_vb = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vb)
        glBufferData(GL_ARRAY_BUFFER, g_quad_vertex_buffer_data.nbytes, g_quad_vertex_buffer_data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Настройка параметров текстуры
        glEnable(GL_TEXTURE_2D)
        self.image_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.image_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _res.width, _res.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

    def push_new_image(self, _zed_mat):
        # Загрузка нового изображения в текстуру
        glBindTexture(GL_TEXTURE_2D, self.image_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _zed_mat.get_width(), _zed_mat.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(_zed_mat.get_pointer()))
        glBindTexture(GL_TEXTURE_2D, 0)
        
    def draw(self):
        # Отрисовка изображения
        glUseProgram(self.shader_image.get_program_id())
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.image_tex)
        glUniform1i(self.tex_id, 0)

        # Установка флагов инвертирования и изменения цвета
        glUniform1i(glGetUniformLocation(self.shader_image.get_program_id(), "revert"), 1)
        glUniform1i(glGetUniformLocation(self.shader_image.get_program_id(), "rgbflip"), 1)

        # Активация и настройка вершинного буфера
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vb)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)


class GLViewer:
    """
    Класс, управляющий рендерингом в OpenGL.
    """
    def __init__(self):
        # Инициализация основных атрибутов
        self.available = False
        self.objects_name = []
        self.mutex = Lock()
        self.is_tracking_on = False
        
        # Инициализация атрибутов управления состоянием
        self.all_persons_inside = False
        self.saved_polygon = []
        self.saved_polygon_state = None
        self.is_polygon_dynamic = True  
        self.current_polygon = []
        self.person_status = {}

        # Инициализация зависимостей
        self.viewer2d = None
        self.track_view_generator = None
        self.mode_controller = None

    def set_dependencies(self, viewer2d, track_view_generator, mode_controller):
        self.track_view_generator = track_view_generator
        self.mode_controller = mode_controller

    def init(self, _params, _is_tracking_on):
        glutInit()
        wnd_w = glutGet(GLUT_SCREEN_WIDTH)
        wnd_h = glutGet(GLUT_SCREEN_HEIGHT)
        width = (int)(wnd_w*0.9)
        height = (int)(wnd_h*0.9)

        glutInitWindowSize(width, height)
        glutInitWindowPosition((int)(wnd_w*0.05),(int)(wnd_h*0.05))
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_SRGB)


        # Initialize image renderer
        self.image_handler = ImageHandler()
        
        if config.enable_gui:
            glutCreateWindow("ZED Object detection")
            self.image_handler.initialize(_params.image_size)
            self.shader_image = Shader(VERTEX_SHADER, FRAGMENT_SHADER)
            self.shader_MVP = glGetUniformLocation(self.shader_image.get_program_id(), "u_mvpMatrix")
            glutDisplayFunc(self.draw_callback)
            glutIdleFunc(self.idle)
            # Register the function called on key pressed
            glutKeyboardFunc(self.keyPressedCallback)
            # Register the closing function
            glutCloseFunc(self.close_func)
        
        glViewport(0, 0, width, height)

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_FRAMEBUFFER_SRGB)

        # Create the rendering camera
        self.projection = array.array('f')
        self.set_render_camera_projection(_params, 0.5, 20)

        # Create the bounding box object
        self.BBox_edges = Simple3DObject(False)
        self.BBox_edges.set_drawing_type(GL_LINES)

        self.BBox_faces = Simple3DObject(False)
        self.BBox_faces.set_drawing_type(GL_QUADS)

        self.BBox_lines = Simple3DObject(False)
        self.BBox_lines.set_drawing_type(GL_QUADS)
    
        self.is_tracking_on = _is_tracking_on

        # Set OpenGL settings
        glDisable(GL_DEPTH_TEST)    # avoid occlusion with bbox
        glLineWidth(1.5)

        self.available = True

    def set_render_camera_projection(self, _params, _znear, _zfar):
        # Just slightly move up the ZED camera FOV to make a small black border
        fov_y = (_params.v_fov + 0.5) * M_PI / 180
        fov_x = (_params.h_fov + 0.5) * M_PI / 180

        self.projection.append( 1 / math.tan(fov_x * 0.5) )  # Horizontal FoV.
        self.projection.append( 0)
        # Horizontal offset.
        self.projection.append( 2 * ((_params.image_size.width - _params.cx) / _params.image_size.width) - 1)
        self.projection.append( 0)

        self.projection.append( 0)
        self.projection.append( 1 / math.tan(fov_y * 0.5))  # Vertical FoV.
        # Vertical offset.
        self.projection.append(-(2 * ((_params.image_size.height - _params.cy) / _params.image_size.height) - 1))
        self.projection.append( 0)

        self.projection.append( 0)
        self.projection.append( 0)
        # Near and far planes.
        self.projection.append( -(_zfar + _znear) / (_zfar - _znear))
        # Near and far planes.
        self.projection.append( -(2 * _zfar * _znear) / (_zfar - _znear))

        self.projection.append( 0)
        self.projection.append( 0)
        self.projection.append( -1)
        self.projection.append( 0)

    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available

    def render_object(self, _object_data):      # _object_data of type sl.ObjectData
        if self.is_tracking_on:
            return _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK
        else:
            return _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK or _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF

    def update_view(self, _image, _objs):  # _objs of type sl.Objects
        self.mutex.acquire()

        # Обновление изображения
        if config.enable_gui:
            self.image_handler.push_new_image(_image)

        self.clear_previous_objects()
        self.process_objects(_objs) 

        self.check_person_inside_static_polygon(_objs)
        self.draw_walls_between_objects(_objs)
        self.mutex.release()

    def clear_previous_objects(self):
        self.BBox_edges.clear()
        self.BBox_faces.clear()
        self.BBox_lines.clear()
        self.objects_name = []
        
    def process_objects(self, _objs):
        for obj in _objs.object_list:
            if not self.render_object(obj):
                continue

            bounding_box = np.array(obj.bounding_box)
            if not bounding_box.any():
                continue

            base_color = self.get_base_color(obj)
            pos = [obj.position[0], obj.bounding_box[0][1], obj.position[2]]

            self.add_rendering_info(pos, base_color, obj, offset_y=0.2, label="ID", value=f"Id {obj.id}")
            self.add_rendering_info(pos, base_color, obj, offset_y=0.1, label="Distance", value=f"{abs(round(pos[2], 2))} m")
            self.add_rendering_info(pos, base_color, obj, offset_y=0, label="Confidence", value=f"conf {round(obj.confidence / 100, 1)}")
            self.create_bbox_rendering(bounding_box, base_color)
    
    def add_rendering_info(self, pos, base_color, obj, offset_y=0, label="", value=""):
        """
        Добавляет информацию для рендеринга на сцене.
        """
        tmp = ObjectClassName()
        tmp.position = np.array([pos[0], pos[1] + offset_y, pos[2]], np.float32)
        tmp.color = base_color

        if label == "ID":
            tmp.name = value
        elif label == "Distance":
            tmp.name = value
        elif label == "Confidence":
            tmp.name = value
        else:
            print("Неизвестный тип лейбла")

        self.objects_name = np.append(self.objects_name, tmp)
    
    def get_base_color(self, obj):
        if obj.raw_label == 1:  # Конус: желтый для динамического, синий для статического
            return [255, 255, 0, 0.5] if self.is_polygon_dynamic or not self.current_polygon else [0, 0, 255, 0.5]
        elif obj.raw_label == 0:  # Человек: зеленый внутри, красный снаружи полигона
            person_point = Point(obj.position[0], obj.position[2])
            polygon_to_check = None
            try:
                if self.is_polygon_dynamic and self.current_polygon:
                    polygon_to_check = Polygon([(p[0], p[2]) for quad in self.current_polygon for p in quad])
                elif self.saved_polygon_state:
                    polygon_to_check = Polygon([(p[0], p[2]) for quad in self.saved_polygon_state for p in quad])
                
                if polygon_to_check and polygon_to_check.contains(person_point):
                    return [102/255, 205/255, 170/255, 0.5]  # Зеленый
                else:
                    return [255, 0, 0, 0.5]  # Красный
            except (TypeError, GEOSException) as e:
                print(f"Ошибка при проверке полигона: {e}")
                return [255, 255, 255, 0.5]  # Красный по умолчанию при ошибке


    def draw_walls_between_objects(self, _objs):
        if not self.is_polygon_dynamic and self.saved_polygon_state:
            for quad_pts in self.saved_polygon_state:
                self.add_wall(quad_pts, 0.9, 0.4)
            return

        self.saved_polygon.clear()

        filtered_objects = [obj for obj in _objs.object_list if obj.raw_label == 1]
        if len(filtered_objects) <= 2:
            return

        center = np.mean([[obj.position[0], obj.position[1]] for obj in filtered_objects], axis=0)
        sorted_objects = sorted(filtered_objects, key=lambda obj: np.arctan2(obj.position[1] - center[1], obj.position[0] - center[0]))

        wall_offset = 0.15

        for i, obj in enumerate(sorted_objects):
            next_obj = sorted_objects[(i + 1) % len(sorted_objects)]
            self.create_wall_between_objects(obj, next_obj, center, wall_offset)

    def create_wall_between_objects(self, obj, next_obj, center, wall_offset):
        direction_obj = np.array([obj.position[0] - center[0], obj.position[1] - center[1]])
        direction_next_obj = np.array([next_obj.position[0] - center[0], next_obj.position[1] - center[1]])

        norm_direction_obj = direction_obj / np.linalg.norm(direction_obj)
        norm_direction_next_obj = direction_next_obj / np.linalg.norm(direction_next_obj)

        wall_height = -obj.dimensions[1] / 2

        offset_point_obj = [obj.position[0] + norm_direction_obj[0] * wall_offset, obj.position[1] + norm_direction_obj[1] * wall_offset, obj.position[2]]
        offset_point_next_obj = [next_obj.position[0] + norm_direction_next_obj[0] * wall_offset, next_obj.position[1] + norm_direction_next_obj[1] * wall_offset, next_obj.position[2]]

        quad_pts = [
            offset_point_obj,
            offset_point_next_obj,
            [offset_point_next_obj[0], offset_point_next_obj[1] + wall_height, offset_point_next_obj[2]],
            [offset_point_obj[0], offset_point_obj[1] + wall_height, offset_point_obj[2]]
        ]
        self.add_wall(quad_pts, 0.9, 0.4)

    
    def toggle_polygon_mode(self):  
        if self.is_polygon_dynamic and self.current_polygon:
            if config.polygon:
                self.saved_polygon_state = config.polygon_gl
            else:
                self.saved_polygon_state = list(self.saved_polygon)
            
            self.saved_polygon.clear()
            self.is_polygon_dynamic = False
        else:
            self.saved_polygon_state = None
            self.is_polygon_dynamic = True
    
    def add_wall(self, quad_pts, alpha_start, alpha_end):
        if self.is_polygon_dynamic:
            self.current_polygon.append(quad_pts)  # Обновите текущее состояние динамического полигона
        # Определение цвета стены на основе текущего состояния
        clr_start, clr_end = ([0, 205, 0, alpha] if self.all_persons_inside else
                              [205, 0, 0, alpha] for alpha in [alpha_start, alpha_end])
                
        # Добавление нижних и верхних точек стены
        for pt, clr in zip(quad_pts, [clr_start] * 2 + [clr_end] * 2):
            self.BBox_lines.add_pt(pt)
            self.BBox_lines.add_clr(clr)

        # Добавление индексов для формирования четырехугольника
        self.BBox_lines.indices.extend([len(self.BBox_lines.indices) + i for i in range(4)])
        self.saved_polygon.append(quad_pts)
    
    def check_person_inside_static_polygon(self, _objs):
        self.person_status.clear()
        polygon_to_check = None

        # Выбор полигона для проверки
        try:
            if self.is_polygon_dynamic and self.current_polygon:
                polygon_to_check = Polygon([(p[0], p[2]) for quad in self.current_polygon for p in quad])
                self.current_polygon.clear()
            elif self.saved_polygon_state:
                polygon_to_check = Polygon([(p[0], p[2]) for quad in self.saved_polygon_state for p in quad])
        except:
            print(1)
        # Если полигон не определен, статус зависит от наличия людей
        if not polygon_to_check:
            self.all_persons_inside = not any(obj.raw_label == 0 for obj in _objs.object_list)
            return
        
        # Проверка и логирование положения людей относительно полигона
        for obj in filter(lambda x: x.raw_label == 0, _objs.object_list):
            is_inside = polygon_to_check.contains(Point(obj.position[0], obj.position[2]))
            self.person_status[obj.id] = is_inside
            
        self.all_persons_inside = all(self.person_status.values())
    
    def create_bbox_rendering(self, _bbox, _bbox_clr):
        self.BBox_edges.add_full_edges(_bbox, _bbox_clr)
        self.BBox_edges.add_vertical_edges(_bbox, _bbox_clr)
        self.BBox_faces.add_vertical_faces(_bbox, _bbox_clr)
        self.BBox_faces.add_top_face(_bbox, _bbox_clr)
     
    def idle(self):
        if self.available:
            glutPostRedisplay()

    def exit(self):
        if self.available:
            self.available = False
            self.image_handler.close()

    def close_func(self):
        if self.available:
            self.available = False
            self.image_handler.close()

    def keyPressedCallback(self, key, x, y):
        if key == b'q' or key == b'Q':
            self.close_func()
        elif key == b'1':  # Обратите внимание на использование байтового литерала
            # self.toggle_polygon_mode()
            if self.mode_controller:
                self.mode_controller.toggle_all_modes()
            else:
                print("ModeController не установлен")

    def draw_callback(self):
        if self.available:
            # Очистка буферов цвета и глубины
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Включение теста глубины
            # glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LEQUAL)

            self.mutex.acquire()
            self.update()
            self.draw()
            self.print_text()
            self.mutex.release()

            glutSwapBuffers()
            glutPostRedisplay()

    def update(self):
        self.BBox_edges.push_to_GPU()
        self.BBox_faces.push_to_GPU()
        self.BBox_lines.push_to_GPU()

    def draw(self):
        self.image_handler.draw()

        glUseProgram(self.shader_image.get_program_id())
        glUniformMatrix4fv(self.shader_MVP, 1, GL_TRUE,  (GLfloat * len(self.projection))(*self.projection))
        self.BBox_edges.draw()
        self.BBox_faces.draw()
        self.BBox_lines.draw()
        glUseProgram(0)

    def print_text(self):
        glDisable(GL_BLEND)

        wnd_size = sl.Resolution()
        wnd_size.width = glutGet(GLUT_WINDOW_WIDTH)
        wnd_size.height = glutGet(GLUT_WINDOW_HEIGHT)

        if len(self.objects_name) > 0:
            for obj in self.objects_name:
                pt2d = self.compute_3D_projection(obj.position, self.projection, wnd_size)
                glColor4f(obj.color[0], obj.color[1], obj.color[2], obj.color[3])
                glWindowPos2f(pt2d[0], pt2d[1])
                for i in range(len(obj.name)):
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ctypes.c_int(ord(obj.name[i])))
            glEnable(GL_BLEND)

    def compute_3D_projection(self, _pt, _cam, _wnd_size):
        pt4d = np.array([_pt[0],_pt[1],_pt[2], 1], np.float32)
        _cam_mat = np.array(_cam, np.float32).reshape(4,4)

        proj3D_cam = np.matmul(pt4d, _cam_mat)     # Should result in a 4 element row vector
        proj3D_cam[1] = proj3D_cam[1] + 0.25
        proj2D = [((proj3D_cam[0] / pt4d[3]) * _wnd_size.width) / (2. * proj3D_cam[3]) + (_wnd_size.width * 0.5)
                , ((proj3D_cam[1] / pt4d[3]) * _wnd_size.height) / (2. * proj3D_cam[3]) + (_wnd_size.height * 0.5)]
        return proj2D
     
    def capture_opengl_window(window_name):
        # Убедитесь, что контекст OpenGL активен
        glutSetWindow(glutGetWindow())

        # Захват изображения с экрана
        glReadBuffer(GL_FRONT)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)  # Установите соответствующий выравнивание пикселей
        data = glReadPixels(0, 0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT), GL_BGR, GL_UNSIGNED_BYTE)

        # Попытка преобразовать данные в изображение OpenCV
        try:
            image_opengl = cv2.flip(cv2.cvtColor(np.frombuffer(data, dtype=np.uint8).reshape((glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_WINDOW_WIDTH), 3)), cv2.COLOR_BGR2RGB), 0)
        except cv2.error as e:
            print("Ошибка при создании изображения OpenCV:", e)

        return image_opengl


class ObjectClassName:
    def __init__(self):
        self.position = [0,0,0] # [x,y,z]
        self.name = ""
        self.color = [0,0,0,0]  # [r,g,b,a]
