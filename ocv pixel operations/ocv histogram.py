# Импорт библиотек для работы с изображениями и визуализации
import cv2  # OpenCV для работы с изображениями
import numpy as np  # NumPy для численных операций
from matplotlib import pyplot as plt  # Matplotlib для построения графиков
plt.rcParams['figure.figsize'] = [15, 5]  # Настройка размера фигур по умолчанию
from time import time  # Для измерения времени выполнения

# ============================================================================
# ГИСТОГРАММА ИЗОБРАЖЕНИЯ
# ============================================================================
# Гистограмма показывает распределение значений яркости пикселей в изображении.
# Это важный инструмент для анализа контраста и яркости изображения.

# Загружаем изображение и преобразуем в оттенки серого
img = cv2.imread('data/kodim05.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Отображаем исходное изображение
plt.figure(figsize=(10, 5))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Исходное изображение в оттенках серого', fontsize=12, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()  # Отображаем графики

# ============================================================================
# ВЫЧИСЛЕНИЕ ГИСТОГРАММЫ ВРУЧНУЮ (ПОЭЛЕМЕНТНО)
# ============================================================================

# Засекаем время начала вычисления
start = time()

# Получаем размеры изображения
rows, cols = img.shape

# Создаём массив для хранения гистограммы (256 уровней яркости: от 0 до 255)
hist = np.zeros(256)

# Проходим по каждому пикселю изображения и подсчитываем количество пикселей
# каждого уровня яркости
for r in range(rows):
    for c in range(cols):
        hist[img[r,c]] = hist[img[r,c]] + 1

print('Время выполнения (поэлементно):', time() - start)

# Строим график гистограммы
plt.figure(figsize=(12, 5))
plt.plot(np.arange(0, 256), hist, linewidth=2, color='steelblue')
plt.grid(True, alpha=0.3)
plt.xlabel('Уровень яркости пикселя', fontsize=11)
plt.ylabel('Количество пикселей', fontsize=11)
plt.title('Гистограмма изображения\n(распределение значений яркости)', fontsize=12, fontweight='bold')
plt.xlim([0, 255])
plt.tight_layout()
plt.show()  # Отображаем графики

# ============================================================================
# ЭКВАЛИЗАЦИЯ ГИСТОГРАММЫ
# ============================================================================
# Эквализация гистограммы - это метод улучшения контраста изображения
# путём перераспределения интенсивностей пикселей для получения равномерной гистограммы.

# Вычисляем кумулятивную функцию распределения (CDF - Cumulative Distribution Function)
# CDF показывает накопленную сумму вероятностей
cdf = np.zeros(256)
for idx, h in enumerate(hist):
    # Накопленная сумма элементов гистограммы до текущего индекса
    cdf[idx] = np.sum(hist[0:idx+1])

# Нормализуем CDF (приводим к диапазону [0, 1])
cdf = cdf/np.sum(hist)

# Строим график кумулятивной функции распределения
plt.figure(figsize=(12, 5))
plt.plot(cdf, linewidth=2, color='darkgreen')
plt.grid(True, alpha=0.3)
plt.xlabel('Уровень яркости пикселя', fontsize=11)
plt.ylabel('Кумулятивная функция распределения (CDF)', fontsize=11)
plt.title('Кумулятивная функция распределения\n(используется для эквализации)', fontsize=12, fontweight='bold')
plt.xlim([0, 255])
plt.ylim([0, 1])
plt.tight_layout()
plt.show()  # Отображаем графики

# Применяем эквализацию: каждый пиксель заменяем на значение из CDF, умноженное на 255
equalized = np.zeros((rows, cols), dtype=np.uint8)
for r in range(rows):
    for c in range(cols):
        equalized[r,c] = 255*cdf[img[r,c]]

# Сравниваем исходное и эквализированное изображение
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Исходное изображение', fontsize=11, fontweight='bold')
plt.axis('off')

plt.subplot(122)
plt.imshow(equalized, cmap='gray', vmin=0, vmax=255)
plt.title('Эквализированное изображение\n(улучшенный контраст)', fontsize=11, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()  # Отображаем графики

# ============================================================================
# ПОЛЕЗНЫЕ ФУНКЦИИ
# ============================================================================
# Вычисление гистограммы и эквализация доступны в библиотеках numpy и OpenCV.
# Эти функции работают намного быстрее, чем ручная реализация.

# Вычисляем гистограмму с помощью функции numpy.histogram
start = time()
hist, bins = np.histogram(img.ravel(), bins=256, range=(0,255))
print('Время выполнения (numpy.histogram):', time() - start)

# Строим график гистограммы, вычисленной с помощью numpy
plt.figure(figsize=(12, 5))
plt.plot(bins[0:-1]+0.5, hist, linewidth=2, color='steelblue')
plt.grid(True, alpha=0.3)
plt.xlabel('Уровень яркости пикселя', fontsize=11)
plt.ylabel('Количество пикселей', fontsize=11)
plt.title('Гистограмма изображения (numpy.histogram)\nБыстрое вычисление с помощью оптимизированных функций', 
          fontsize=12, fontweight='bold')
plt.xlim([0, 255])
plt.tight_layout()
plt.show()  # Отображаем графики

# Применяем эквализацию гистограммы с помощью функции OpenCV
dst = cv2.equalizeHist(img)

# Сравниваем результаты
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Исходное изображение', fontsize=11, fontweight='bold')
plt.axis('off')

plt.subplot(122)
plt.imshow(dst, cmap='gray', vmin=0, vmax=255)
plt.title('Эквализированное изображение (cv2.equalizeHist)\nАвтоматическое улучшение контраста', 
          fontsize=11, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()  # Отображаем графики

# ============================================================================
# АДАПТИВНАЯ ЭКВАЛИЗАЦИЯ ГИСТОГРАММЫ С ОГРАНИЧЕНИЕМ КОНТРАСТА (CLAHE)
# ============================================================================
# Обычная эквализация гистограммы предполагает, что изображение с хорошим контрастом
# должно иметь "плоскую" функцию плотности вероятности (PDF). Однако это не всегда верно.
# На изображении ниже показана тёмная шина, где обычная эквализация слишком агрессивна.

# Загружаем новое изображение для демонстрации проблем обычной эквализации
img = cv2.imread('data/tire.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Показываем исходное изображение и результат обычной эквализации
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Исходное изображение\n(тёмная область)', fontsize=11, fontweight='bold')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.equalizeHist(img), cmap='gray', vmin=0, vmax=255)
plt.title('Обычная эквализация\n(слишком агрессивная)', fontsize=11, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()  # Отображаем графики

# Вычисляем CDF для исходного изображения
hist, bins = np.histogram(img.ravel(), bins=256, range=(0,255))
cdf = np.cumsum(hist/np.sum(hist))

# Строим график функции преобразования (Input -> Output)
plt.figure(figsize=(8, 8))
plt.plot(255*cdf, linewidth=2, color='purple')
plt.axis('square')
plt.grid(True, alpha=0.3)
plt.xlabel('Входная яркость (Input)', fontsize=11)
plt.ylabel('Выходная яркость (Output)', fontsize=11)
plt.title('Функция преобразования для эквализации\n(Input → Output mapping)', 
          fontsize=12, fontweight='bold')
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.tight_layout()
plt.show()  # Отображаем графики

# Создаём адаптивный эквализатор с ограничением контраста (CLAHE)
# clipLimit: ограничение контраста (2.0 - умеренное значение)
# tileGridSize: размер блока для локальной эквализации (8x8 пикселей)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Сравниваем три метода: исходное, обычная эквализация и CLAHE
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Исходное изображение', fontsize=11, fontweight='bold')
plt.axis('off')

plt.subplot(132)
plt.imshow(cv2.equalizeHist(img), cmap='gray', vmin=0, vmax=255)
plt.title('Обычная эквализация\n(агрессивная обработка)', fontsize=11, fontweight='bold')
plt.axis('off')

plt.subplot(133)
plt.imshow(clahe.apply(img), cmap='gray', vmin=0, vmax=255)
plt.title('CLAHE (Адаптивная эквализация)\n(сбалансированное улучшение)', fontsize=11, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()  # Отображаем графики

# ============================================================================
# ЭКВАЛИЗАЦИЯ ГИСТОГРАММЫ ДЛЯ ЦВЕТНЫХ ИЗОБРАЖЕНИЙ
# ============================================================================
# Как применить эквализацию гистограммы к цветным изображениям?
# 
# Вариант 1: Эквализировать каждый канал RGB отдельно.
# Проблема: это может изменить цветовую палитру изображения и привести к появлению
# неестественных цветов.
# 
# Вариант 2: Преобразовать изображение в цветовое пространство HSV и эквализировать
# только канал яркости (Value/Intensity). Это сохраняет цветовую информацию,
# изменяя только яркость изображения.

from urllib.request import urlopen, Request

def read_image_from_url(url):
    """
    Загружает изображение из URL и преобразует в RGB формат для OpenCV.
    
    Параметры:
        url: URL адрес изображения
        
    Возвращает:
        Изображение в формате RGB
    """
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urlopen(req)
    image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # OpenCV использует BGR, преобразуем в RGB для matplotlib
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Загружаем изображения из интернета для демонстрации
urls = ['https://upload.wikimedia.org/wikipedia/commons/c/c8/Common_errors_-_underexposed.jpg',
        'https://www.nikonforums.com/forums/uploads/monthly_05_2016/post-13788-0-32083700-1464045099.jpg']

images = [read_image_from_url(url) for url in urls]

# Отображаем исходные изображения
plt.figure(figsize=(12, 5))
for idx, image in enumerate(images):
    plt.subplot(1, 2, idx+1)
    plt.imshow(image)
    plt.title(f'Исходное изображение {idx+1}\n(недоэкспонированное)', 
              fontsize=11, fontweight='bold')
    plt.axis(False)

plt.tight_layout()
plt.show()  # Отображаем изображения

def equalize(x, clip):
    """
    Применяет эквализацию к каналу изображения.
    
    Параметры:
        x: одноканальное изображение (grayscale)
        clip: если True, использует CLAHE, иначе обычную эквализацию
        
    Возвращает:
        Эквализированное изображение
    """
    if clip:
        return clahe.apply(x)
    else:
        return cv2.equalizeHist(x)

clip = False  # Используем обычную эквализацию (можно изменить на True для CLAHE)

# Применяем различные методы эквализации к каждому изображению
plt.figure(figsize=(18, 10))

for idx, image in enumerate(images):
    # Метод 1: Эквализация каждого RGB канала отдельно
    red, green, blue = cv2.split(image)

    red = equalize(red, clip)
    blue = equalize(blue, clip)
    green = equalize(green, clip)

    # Отображаем результаты
    plt.subplot(2, 3, 3*idx+1)
    plt.imshow(image)
    plt.title(f'Исходное изображение {idx+1}', fontsize=10, fontweight='bold')
    plt.axis(False)

    plt.subplot(2, 3, 3*idx+2)
    plt.imshow(cv2.merge([red, green, blue]))
    plt.title(f'Эквализация RGB каналов\n(может изменить цвета)', 
              fontsize=10, fontweight='bold')
    plt.axis(False)

    # Метод 2: Эквализация только канала яркости в HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(image_hsv)

    # Эквализируем только канал яркости (Value), сохраняя цвет и насыщенность
    value = equalize(value, clip)
    out = cv2.cvtColor(cv2.merge([hue, saturation, value]), cv2.COLOR_HSV2RGB)

    plt.subplot(2, 3, 3*idx+3)
    plt.imshow(out)
    plt.title(f'Эквализация только яркости (HSV)\n(сохраняет цвета)', 
              fontsize=10, fontweight='bold')
    plt.axis(False)

plt.tight_layout()
plt.show()  # Отображаем изображения