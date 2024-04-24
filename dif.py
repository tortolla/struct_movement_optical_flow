from PIL import Image
import numpy as np

def calculate_pixel_difference(img_path1, img_path2, output_path):
    # Загрузка изображений
    img1 = Image.open(img_path1).convert('L')  # Преобразование в черно-белое
    img2 = Image.open(img_path2).convert('L')  # Преобразование в черно-белое
    
    # Конвертация изображений в массивы numpy
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    # Проверка на одинаковый размер изображений
    if img1_array.shape != img2_array.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Вычисление модуля разности яркости
    difference = np.abs(img2_array - img1_array)
    
    # Сохранение результата как изображения
    Image.fromarray(difference.astype(np.uint8)).save(output_path)
    
    # Возвращение массива с модулем разности
    return difference


if __name__ == '__main__':

    img_path1 = '/Users/tortolla/Desktop/resized/1.jpg'
    img_path2 = '/Users/tortolla/Desktop/resized/2.jpg'
    output_path = '/Users/tortolla/Desktop/dif/dif.png'
    
    calculate_pixel_difference(img_path1, img_path2, output_path)