import re
from PIL import Image
import numpy as np
import os

def copy_images(base_path, output_folder):
    # Составляем паттерн для идентификации нужных папок
    pattern = re.compile(r'^image(\d+)$')
    
    # Создание папки для результатов, если она ещё не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Получение списка подходящих папок
    folders = [folder for folder in os.listdir(base_path) if pattern.match(folder)]
    
    # Перебор всех подходящих папок
    for folder in folders:
        # Формирование полного пути к папке
        folder_path = os.path.join(base_path, folder)
        
        # Получение списка файлов в папке
        files = os.listdir(folder_path)
        
        # Копирование изображений
        for file in files:
            # Полный путь к исходному изображению
            img_path = os.path.join(folder_path, file)
            
            # Загрузка изображения
            img = Image.open(img_path)
            
            # Формирование пути для сохранения копии изображения
            output_filename = f'{folder}_{file}'
            output_path = os.path.join(output_folder, output_filename)
            
            # Сохранение изображения
            img.save(output_path)
            #print(f"Saved: {output_path}")




def process_image_folders(base_path, output_folder):
    # Составляем паттерн для идентификации и извлечения чисел из нужных папок
    pattern = re.compile(r'^image(\d+)$')
    
    # Получение и сортировка папок, которые соответствуют шаблону 'image*'
    folders = sorted(
        [folder for folder in os.listdir(base_path) if pattern.match(folder)],
        key=lambda x: int(pattern.match(x).group(1))
    )

    # Создание папки для результатов, если она ещё не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(folders)
    
    # Перебор папок по парам
    for i in range(len(folders) - 1):
        folder1 = folders[i]
        folder2 = folders[i + 1]

        # Формирование путей к изображениям в этих папках
        img_path1 = os.path.join(base_path, folder1, os.listdir(os.path.join(base_path, folder1))[0])
        img_path2 = os.path.join(base_path, folder2, os.listdir(os.path.join(base_path, folder2))[0])
        print(img_path1, img_path2)
        # Расчёт модуля разности пикселей
        difference_array = calculate_pixel_difference(img_path1, img_path2)

        # Формирование пути сохранения нового изображения
        output_filename = f'dif_{i + 1}_{i + 2}.png'
        output_path = os.path.join(output_folder, output_filename)

        # Сохранение нового изображения
        Image.fromarray(difference_array).save(output_path)
        #print(f"Processed and saved: {output_path}")

def calculate_pixel_difference(img_path1, img_path2):
    # Загрузка и преобразование изображений в черно-белый формат
    img1 = Image.open(img_path1).convert('L')
    img2 = Image.open(img_path2).convert('L')

    # Конвертация изображений в массивы numpy
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    # Проверка на одинаковый размер изображений
    if img1_array.shape != img2_array.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Вычисление модуля разности яркости
    difference = np.abs(img2_array - img1_array)
    print(difference[difference!=0])
    
    # Возвращение массива с модулем разности
    return difference

# Пример использования:
base_folder_path = '/Users/tortolla/Desktop/res_of_simulation'  # Путь к базовой папке с папками image1, image2, ...
output_results_folder = '/Users/tortolla/Desktop/dif'  # Путь к папке, куда будут сохраняться результаты
process_image_folders(base_folder_path, output_results_folder)


# Пример использования:
#base_folder_path = '/Users/tortolla/Desktop/res_of_simulation'  # Путь к базовой папке с папками image1, image2, ...
output_results_folder_1 = '/Users/tortolla/Desktop/optical_flow'  # Путь к папке, куда будут сохраняться копии изображений
copy_images(base_folder_path, output_results_folder_1)