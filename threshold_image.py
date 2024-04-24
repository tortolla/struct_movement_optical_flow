import os
from PIL import Image

def threshold_images(input_folder, output_folder, threshold):
    # Проверяем, существует ли указанная папка вывода, если нет, то создаем ее
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Проходим по всем файлам в указанной папке ввода
    for filename in os.listdir(input_folder):
        # Проверяем, является ли файл изображением
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # Формируем пути к входному и выходному файлам
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Открываем изображение
            image = Image.open(input_path)

            # Преобразуем изображение в черно-белое
            bw_image = image.convert("L")

            # Получаем размеры изображения
            width, height = bw_image.size

            # Проходим по каждому пикселю изображения
            for y in range(height):
                for x in range(width):
                    # Получаем яркость пикселя
                    pixel = bw_image.getpixel((x, y))
                    # Если яркость пикселя меньше порога, устанавливаем ее в 0
                    if pixel < threshold:
                        bw_image.putpixel((x, y), 0)
                    # Если яркость пикселя больше или равна порогу, устанавливаем ее в 255
                    else:
                        bw_image.putpixel((x, y), 255)

            # Сохраняем измененное изображение в формате PNG
            bw_image.save(output_path, format="PNG")

            print(f"Processed image '{filename}' saved to '{output_folder}'")

# Пример использования функции
threshold_images("/Users/tortolla/Desktop/resized", "/Users/tortolla/Desktop/check_it", 128)






# from PIL import Image

# def threshold_image(image_path, threshold):
#     # Открываем изображение
#     image = Image.open(image_path)

#     # Преобразуем изображение в черно-белое
#     bw_image = image.convert("L")

#     # Получаем размеры изображения
#     width, height = bw_image.size

#     # Проходим по каждому пикселю изображения
#     for y in range(height):
#         for x in range(width):
#             # Получаем яркость пикселя
#             pixel = bw_image.getpixel((x, y))
#             # Если яркость пикселя меньше порога, устанавливаем ее в 0
#             if pixel < threshold:
#                 bw_image.putpixel((x, y), 0)
#             # Если яркость пикселя больше или равна порогу, устанавливаем ее в 255
#             else:
#                 bw_image.putpixel((x, y), 255)

#     # Сохраняем измененное изображение
#     bw_image.save("/Users/tortolla/Desktop/check_it/0.png")

# # Пример использования функции
# threshold_image("/Users/tortolla/Desktop/resized/0.jpg", 80)  # Порог 128
