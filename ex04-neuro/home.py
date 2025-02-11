import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os


# Предобработка изображения
def preprocess_image(image: np.ndarray) -> np.ndarray:
    resized_image = cv2.resize(image, (28, 28))
    _, binary_image = cv2.threshold(resized_image, 128, 255, cv2.THRESH_BINARY_INV)
    return binary_image


# Создание архитектуры CNN для распознавания цифр
def build_model() -> Sequential:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model


# Обучение модели на данных MNIST
def train_model(model: Sequential) -> None:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_test, y_test))


# Оценка модели на тестовых данных MNIST
def evaluate_model(model: Sequential) -> None:
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_test = to_categorical(y_test)
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')


# Распознавание цифры на изображении с использованием обученной модели
def recognize_digit(image: np.ndarray, model: Sequential) -> int:
    processed_image = preprocess_image(image)
    processed_image = processed_image.reshape(1, 28, 28, 1).astype('float32') / 255
    prediction = model.predict(processed_image)
    return np.argmax(prediction)


# Разделение цифр в числе с использованием алгоритма кластеризации KMeans
def split_digits(image: np.ndarray) -> list:
    # Преобразование в бинарное изображение
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Поиск контуров
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Фильтрация слишком маленьких областей
            digit = binary_image[y:y + h, x:x + w]
            digit_resized = cv2.resize(digit, (28, 28))
            digit_images.append(digit_resized)
            # Отладочная информация: вывод координат и размеров контура
            print(f"Найден контур: x={x}, y={y}, w={w}, h={h}")

    if not digit_images:
        print("На изображении не найдено цифр.")

    return digit_images


# Пример использования функций
if __name__ == "__main__":
    # Построение и обучение модели
    model = build_model()
    train_model(model)
    evaluate_model(model)

    # Загрузка изображения с несколькими цифрами
    image_path = '/home/daria/PycharmProjects/test1.png'  # Путь к вашему изображению
    print(os.path.exists(image_path))
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден.")
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Не удалось загрузить изображение.")
        else:
            # Разделение цифр
            digit_images = split_digits(image)

            # Распознавание каждой цифры
            recognized_digits = [recognize_digit(digit, model) for digit in digit_images]

            # Вывод распознанных цифр
            print("Recognized digits:", recognized_digits)
