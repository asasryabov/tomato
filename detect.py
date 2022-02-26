#!python3
# -*- coding: utf-8 -*-

import os # для установки переменных окружения
import numpy as np # для работы с матрицами
import json # для чтения set.json
import toga # графический интерфейс

from keras.models import load_model # для загрузки обученной модели
from keras.preprocessing import image # для загрузки выбранного изображения

from toga.style import Pack # Управление стилями рафического интерфейса
from toga.constants import COLUMN # константа для графического интерфейса

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # убираем предупреждения от tensorflow
np.set_printoptions(precision=2) # вывод numpy чисел с 2 знаками после запятой

def main(): 
    """Запуск приложения"""
    return App('Tomato проверка на болезни', 'org.beeware.widgets.tomato')

class App(toga.App):
    """Класс описывающий графический интерфейс"""
    def startup(self):
        """Начальные настройки графического интерфейса"""
        self.image_size = 224 # размер нормализованного изображения
        self.model_path = "./checkpoint-.87" # путь до обученной модели
        self.model = load_model(self.model_path) # загружаем модель

        """Загружаем описание категорий"""
        with open('./set/set.json') as json_file:
            data = json.load(json_file)

        """Считываем категории в set"""
        categories = {}
        for v in data.get('categories'):
            categories[v["id"]] = v

        """сохраняем категории в параметре объекта"""
        self.categories = categories
        self.main_window = toga.MainWindow(title=self.name) # основное окно 
        self.on_exit = self.exit_handler # обработчик выхода из програмы
        btn_style = Pack(flex=1) # стиль для кнопки
        btn_open_filtered = toga.Button( # кнопка для выбора изображения
            'Выбор файла',
            on_press=self.action_open_file_filtered_dialog, # обработчик загрузки изображения
            style=btn_style
        )

        self.label = toga.Label('Выбери файл', style=Pack(padding_top=20)) # строка для вывода категории
        self.hint = toga.MultilineTextInput('',  style=Pack(padding_top=20),readonly=True) # для вывода рекомендации
        self.image_view = toga.ImageView(id='image to predict') # для показа изображения
        self.box = toga.Box( # контейнер для графического интерфейса
            children=[
                btn_open_filtered,
                self.label,
                self.hint,
                self.image_view
            ],
            style=Pack( # настройки стиля
                flex=1,
                direction=COLUMN,
                padding=10
            )
        )
        self.main_window.content = self.box
        self.main_window.show() # показываем окно

    """предсказание категории"""
    def predict(self):
        predicted_value = self.model.predict(self.img_tensor) # предсказанные категории со значениями вероятностей
        category_idx = predicted_value[0].argmax() # находим максимальную вероятность
        category_name = self.categories[category_idx]["name"] # получаем название категории
        self.label.text = category_name # выводим название категории в окно программы
        self.hint.value = self.categories[category_idx]["hint"] # выводим рекомендации в окно программы

    """открываем выбранное изображение"""
    def open_image(self, fname):
        # загрузка изображения из файла и его нормализация к нужному размеру
        img = image.load_img(fname, target_size=(self.image_size, self.image_size))
        # для изображенияполучаем матрицу, куторую можно передавать на вход нейросети
        img_tensor = image.img_to_array(img) 
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        # запоминаем в параметрах объекта
        self.opened_image = img
        self.img_tensor = img_tensor
        
        # отображаем загруженное изображение
        self.image_view.image = toga.Image(fname)
        # вызываем определение категории по загруженным данным
        self.predict()

    """обработчик нажания на кнопку загрухки изображения"""
    def action_open_file_filtered_dialog(self, widget):
        try:
            # показываем диалог выбора файла
            fname = self.main_window.open_file_dialog(
                title="Открыть файл",
                multiselect=False,
                file_types=['jpg', 'jpeg', 'png'], # фильтр по расширениям
            )
            if fname is not None:
                self.label.text = "Открываем файл:" + fname
                """вызов метода для открытия файла"""
                self.open_image(fname)
            else:
                self.label.text = "Файл не выбран!"
        except ValueError: # Обработка ошибки если файл не был выбран
            self.label.text = "Отмена. Файл не выбран!"

    def exit_handler(self, app):
        # обработчик выхода из программы, если возвращаем True программа закрывается
        return True

if __name__ == '__main__':
    # создаем объект для отображения графического интерфейса
    app = main()
    # запускаем графический интерфейс
    app.main_loop()
