# Импорт необходимых библиотек
import cv2  # OpenCV для работы с изображениями
import numpy as np  # NumPy для математических операций с массивами
from matplotlib import pyplot as plt  # Matplotlib для отображения графиков

# Установка размера фигуры для лучшего отображения
plt.rcParams['figure.figsize'] = [15, 10]

# Загрузка изображения
img = cv2.imread('test_img.png')

# Проверка успешной загрузки изображения
if img is None:
	raise SystemExit("Could not read 'test_img.png'. Make sure the file exists in the script directory.")

try:
	# Конвертация из BGR (OpenCV) в RGB (Matplotlib) для правильного отображения цветов
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	# Отображение оригинального изображения
	plt.imshow(img_rgb)
	plt.title('Original Image')
	plt.show()

	# Разделение изображения на отдельные цветовые каналы
	b, g, r = cv2.split(img)
	
	# Обнуление синего канала (все пиксели становятся равными 0)
	b[:] = 0
	
	# Обнуление зеленого канала (все пиксели становятся равными 0)
	g[:] = 0
	
	# Объединение каналов обратно в изображение (только красный канал остается)
	img_red = cv2.merge((r, g, b))  # Исправлен порядок каналов
	
	# Отображение только красным каналом
	plt.imshow(img_red)
	plt.title('Difference: Original - Red Channel Only')
	plt.show()

	# Разделение изображения на три цветовых канала
	red, green, blue = cv2.split(img)

	# Составление изображения в цветовом пространстве RGB
	img1 = cv2.merge([red, green, blue])

	# Составление изображения в цветовом пространстве RBG
	img2 = cv2.merge([red, blue, green])

	# Составление изображения в цветовом пространстве GRB
	img3 = cv2.merge([green, red, blue])

	# Составление изображения в цветовом пространстве BGR
	img4 = cv2.merge([blue, green, red])

	# Горизонтальное объединение изображений (рядом друг с другом)
	out1 = np.hstack([img1, img2])
	out2 = np.hstack([img3, img4])
	
	# Вертикальное объединение изображений (одно под другим)
	out3 = np.vstack([out1, out2])
	
	# Отображение всех четырех вариантов цветовых пространств
	plt.imshow(out3)
	plt.title('Combined Images')
	plt.show()

	# Горизонтальное отражение изображения (зеркало по вертикальной оси)
	img_horizontal_invert = img_rgb[:, ::-1, :]
	
	# Объединение оригинального и отраженного изображения горизонтально
	img_face_to_face = np.concatenate((img_rgb,img_horizontal_invert) , axis=1)
	
	# Отображение "лицом к лицу" (горизонтальное отражение)
	plt.imshow(img_face_to_face)
	plt.title('Face to Face')
	plt.show()

	# Вертикальное отражение изображения "лицом к лицу" (зеркало по горизонтальной оси)
	img_face_to_face_vertical_invert = img_face_to_face[::-1, :, :]
	
	# Отображение вертикально отраженного изображения
	plt.imshow(img_face_to_face_vertical_invert)
	plt.title('Face to Face vertical invert')
	plt.show()

	# Объединение горизонтального и вертикального отражений (создание полного отражения)
	img_face_to_face_vertical_invert_face_to_face = np.concatenate((img_face_to_face,img_face_to_face_vertical_invert) , axis=0)
	
	# Отображение полного отражения (4 копии изображения в разных ориентациях)
	plt.imshow(img_face_to_face_vertical_invert_face_to_face)
	plt.title('Face to Face full invert')
	plt.show()

except Exception as e:
	print('Error:', e)