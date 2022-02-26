# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from keras import backend as K
from tensorflow import keras
from tensorflow.keras import utils # для рисования схемы модели
from keras.models import Sequential # для создания топологии нейросети
from keras.layers import Dropout, Flatten, Dense # для персептрона
from keras.layers import Conv2D, MaxPooling2D # для сверточной модели
from keras.layers import Input # для входных данных модели
from keras.preprocessing.image import ImageDataGenerator # для обработки изображений в нужный формат

class Trainer(object): 
    """Класс для обучения нейронной сети"""
    
    
    def __init__(self, data_set="./set", validation_set="./validation_set", image_size=50
        ):
        """
        Конструктор: 
            Входные параметры:
            data_set - директория с изображениями для обучения. В директории
                должны быть директории с картинками для
                каждой категории.
            validation_set - директория с изображениями для валидации обучения.
                структура директории такая же как в data_set.
            image_size - размер одной стороны изображения. Размер изображения
                нормализуется до квадратного.
        """
        K.clear_session() # очищаем сессию Keras
        self.data_set       = data_set 
        self.validation_set = validation_set
        self.image_size     = image_size
        self.batch_size     = 25 # размер пакетов для обучения

    def train(self, epochs = 10, out = "./tomato.h5", verbose = 0):
        """
        Функция обучения нейронной сети
            epochs  - количество эпох обучения
            out     - файл для сохранения весов
            verbose - уровень логирования
        """
        # model = self.conv_model() # создаем модель
        model = self.conv_model()
        if verbose: # выводим информацию о размере изображения и о структуре модели
            print(f"Image size: {self.image_size}x{self.image_size}")
            print(model.summary())

        # callback для сохранения модели с лучшим результатом
        # сохраняем по максимальному значению точности
        # на данных для валидации
        checkpoint_filepath = './checkpoint'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            verbose = 1,
            save_best_only=True
        )

        # обучаем модель
        history = model.fit(
            x                     = self.train_processing(),      # генератор изображений для обучения
            batch_size            = self.batch_size,              # размер пачек для обучения
            epochs                = epochs,                       # количество эпох обучения
            validation_data       = self.validation_processing(), # генератор изображений для валидации
            validation_batch_size = self.batch_size,              # размер пачки для валидации
            callbacks             = [model_checkpoint_callback],
            verbose               = verbose                       # уровень логирования
        )
        model.save(out) # сохраняем информацию о модели и найденых весах

        # создаем график точности обучения
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f"Tomato model (Accuracy)")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train accuracy', 'test accuracy'], loc='upper left')
        plt.savefig("graph_accuracy.png")

        # создаем график функции loss
        plt.cla()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f"Tomato model (Loss)")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'test loss'], loc='upper left')
        plt.savefig("graph_loss.png")

    def train_processing(self):
        """Генератор изображений для обучения"""
        train_data_generator = ImageDataGenerator(
            rescale         = 1. / 255, # разрешить менять масштаб
            shear_range     = 0.2,      # разрешить сдвиг
            zoom_range      = 0.2,      # разрешить увеличение
            # horizontal_flip = True,   # разрешит переворот по горизонтали
            # vertical_flip   = True,   # разрешит переворот по вертикали
            rotation_range  = 25        # разрешить вращение>
        )

        # возвращаем генератор изображений из директории
        return train_data_generator.flow_from_directory(
            self.data_set,
            target_size = (self.image_size, self.image_size), # размер нормализованных изображений
            batch_size  = self.batch_size, # пачка для генерации
            class_mode  = "categorical"    # включать категории
        )

    def validation_processing(self):
        test_data_generator = ImageDataGenerator(rescale = 1./255) # генератор с возможностью менять масштаб
        return test_data_generator.flow_from_directory(
            self.validation_set,
            target_size = (self.image_size, self.image_size), # размер нормализованных изображений
            batch_size = self.batch_size, # пачка для генерации
            class_mode = "categorical"    # включать категории
        )

    def conv_model(self):
        """Сверточная модель"""

        # проверяем формат входных данных
        # может быть два варианта
        # 'channels_first' - если первая размерность это каналы изображения
        #   Например в случае RGB - это три канала R - red, G - green, B - blue
        # 'channels_last' - каналы в последней размерности.
        if K.image_data_format == 'channels_firs':
            input_shape = (3, self.image_size, self.image_size)
        else:
            input_shape = (self.image_size, self.image_size, 3)

        # Создаем модель прямого распространения
        model = Sequential(name = "tomato")

        # описываем входные данные (сенсоры)
        model.add(Input(shape = input_shape))
        model.add(Conv2D(32, (2, 2), activation = "relu", bias_initializer='zeros')) # сверточный слой
        model.add(MaxPooling2D(pool_size = (2, 2))) # поллинг фрагментами 2x2

        model.add(Flatten()) # разворачиваем результат в одномерный
        model.add(Dense(32, activation = "relu")) # персептрон с 32 нейронами

        # выходные данные, активационная функция softmax для нескольких категорий
        model.add(Dense(7, activation = "softmax"))

        # компиляция созданной модели
        # оптимизатор rmsprop. Можно ее попробовать adam 
        # функция минимизации - категориальная кросс-энтропия
        model.compile(
            optimizer = "rmsprop",
            loss      = "categorical_crossentropy",
            metrics   = ["accuracy"]
        ) 

        # сохраняем схему модели
        utils.plot_model(
            model,
            to_file = "model.png"
        )

        return model

    def conv_model_81(self):
        """Сверточная модель"""

        # проверяем формат входных данных
        # может быть два варианта
        # 'channels_first' - если первая размерность это каналы изображения
        #   Например в случае RGB - это три канала R - red, G - green, B - blue
        # 'channels_last' - каналы в последней размерности.
        if K.image_data_format == 'channels_firs':
            input_shape = (3, self.image_size, self.image_size)
        else:
            input_shape = (self.image_size, self.image_size, 3)

        # Создаем модель прямого распространения
        model = Sequential(name = "tomato")

        # описываем входные данные (сенсоры)
        model.add(Input(shape = input_shape))
        model.add(Conv2D(32, (2, 2), activation = "relu")) # сверточный слой
        model.add(MaxPooling2D(pool_size = (2, 2))) # поллинг фрагментами 2x2
        model.add(Dropout(0.1)) # вносим шум в результат что бы избежать переобучения

        model.add(Flatten()) # разворачиваем результат в одномерный
        model.add(Dense(32, activation = "relu")) # персептрон с 32 нейронами

        # выходные данные, активационная функция softmax для нескольких категорий
        model.add(Dense(7, activation = "softmax"))

        # компиляция созданной модели
        # оптимизатор rmsprop. Можно ее попробовать adam 
        # функция минимизации - категориальная кросс-энтропия
        model.compile(
            optimizer = "rmsprop",
            loss      = "categorical_crossentropy",
            metrics   = ["accuracy"]
        ) 

        # сохраняем схему модели
        utils.plot_model(
            model,
            to_file = "model.png"
        )

        return model

    def dense_model_50(self):
        """Модель многослойного персептрона"""
        # проверяем формат входных данных
        # может быть два варианта
        # 'channels_first' - если первая размерность это каналы изображения
        #   Например в случае RGB - это три канала R - red, G - green, B - blue
        # 'channels_last' - каналы в последней размерности.
        if K.image_data_format == 'channels_firs':
            input_shape = (3, self.image_size, self.image_size)
        else:
            input_shape = (self.image_size, self.image_size, 3)

        # Создаем модель прямого распространения
        model = Sequential(name = "tomato")

        # описываем входные данные (сенсоры)
        model.add(Input(shape=input_shape))
        model.add(Flatten()) # разварачиваем данные в список
        model.add(Dropout(0.1)) # вносим шумы
        model.add(Dense(1000, activation="relu")) # персептрон
        model.add(Dense(7, activation="softmax")) # результат 

        # компилируем модель
        model.compile(
            optimizer = "rmsprop",
            loss      = "categorical_crossentropy",
            metrics   = ["accuracy"]
        ) 

        # сохраняем схематичное ищображение модели
        utils.plot_model(
            model,
            to_file = "model.png"
        )

        return model
