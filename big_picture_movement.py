from impulse import *
from object import *
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import time
import os
import numpy as np
from PIL import Image
import time
import cv2
import re


file1 = "/Users/tortolla/Desktop/proga/model_for_serv - 1/calibration.txt"
file2 = "/Users/tortolla/Desktop/proga/model_for_serv - 1/calibration_relax.txt"

array1_imp, array2_imp = read_arrays_from_file(file1, 1)
array1_relax, array2_relax = read_arrays_from_file(file2, 0)

p1 = 255 / array1_imp[-1]
p2 = 255 / array1_relax[0]

array1_imp = array1_imp * p1
array1_relax = array1_relax * p2

start = time.perf_counter()

files_dict = []

def resize_images_in_directory(directory_path, output_directory, target_size=(30, 30)):
    """
    Resizes all images in the specified directory to the target size and saves them
    to the output directory.
    
    :param directory_path: Path to the directory containing images.
    :param output_directory: Path to the directory where resized images will be saved.
    :param target_size: Tuple (width, height) representing the target image size.
    :return: List of paths to the resized images.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # List to store the paths of the resized images
    resized_images_paths = []
    
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Path to the original image
            original_image_path = os.path.join(directory_path, filename)
            # Path to save the resized image
            resized_image_path = os.path.join(output_directory, filename)
            
            try:
                # Open the image
                with Image.open(original_image_path) as img:
                    # Resize the image
                    img_resized = img.resize(target_size)
                    # Save the resized image
                    img_resized.save(resized_image_path)
                    # Add the path of the resized image to the list
                    resized_images_paths.append(resized_image_path)
            except Exception as e:
                print(f"Error resizing image {filename}: {e}")
    
    return resized_images_paths

# Example usage: resizing all images in the 'images_to_resize' folder
# and saving them to the 'resized_images' folder.
# You should replace 'images_to_resize' with the path to your specific folder.







def create_alternating_array(length):

    pattern = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    repeats = length // len(pattern) + 1
    array = np.tile(pattern, repeats)[:length]
    array = np.insert(array, 0, 1, )
    array = np.insert(array, 0, 1, )
    array = np.insert(array, 0, 1, )
    array = np.insert(array, 0, 1, )

    return array

def start(name, folder_path, x_start, y_start, a, b, radius):

    pic = np.zeros((a, b))
    circle = Figure(pic, a, b)
    circle.add_round_shape_full(radius, x_start, y_start)

    circle.save(name, folder_path)

    return circle

def image_to_figure(name, folder_path, image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    a, b = image.shape
    
    image = Figure(image, a, b)

    image.save(name, folder_path)

    return image

def image_start(name, folder_path, image_path):
    """странно работающая версия""" 
    # pic = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # height, width = pic.shape
    # image = Figure(pic, width, height)
    # image.save(name, folder_path)

    # return image
    pic = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = pic.shape

    image = Figure(pic, width, height)
    # image.save(name, folder_path)
    img = Image.fromarray(pic, 'L')  # 'L' означает режим оттенков серого

    if isinstance(name, int):
        name = str(name)

    # Проверяем наличие расширения у файла
    if not os.path.splitext(name)[1]:
        name += '.png'  # Добавляем расширение PNG по умолчанию
    # Сохраняем изображение
    img.save(f"{folder_path}/{name}")

    return image

def get_sorted_image_paths(directory_path):

    image_paths = []
    
    for filename in os.listdir(directory_path):
        print(filename)

        if filename.endswith('.jpg'):

            image_paths.append(os.path.join(directory_path, filename))
    
    image_paths.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
    
    return image_paths


def start_with_gauss(name, folder_path, x_start, y_start, a, b, radius, average, sigma):

    pic = np.zeros((a, b))
    circle = Figure(pic, a, b)
    circle.add_round_shape_and_gauss_field(radius, x_start, y_start, average, sigma)

    circle.save(name, folder_path)

    return circle

def start_dot(name, folder_path, x_start, y_start, a, b, radius, average, sigma):

    pic = np.zeros((a, b))
    pic[0][2] = 255
    dot = Figure(pic, a, b)
    dot.save(name, folder_path)

    return dot

"Задаем начальное положение обьекта и создаем его"
def shift_image_right(image, a, b):

    shifted_image = np.zeros_like(image.pic)

    # Сдвигаем строки вниз
    shifted_image[1:, :] = image.pic[:-1, :]

    # Сдвигаем первую строку в последнюю
    shifted_image[0, :] = image.pic[-1, :]

    shifted_object = Figure(shifted_image, a, b)

    return shifted_object

def shift_image_right_with_gauss(image, a, b, shift_amount, average, sigma):
    """
    Shifts the image to the right and adds Gaussian noise to non-255 pixels.

    Parameters:
    image - Instance of Figure with an image to be shifted.
    shift_amount - Number of pixels to shift the image to the right.
    average, sigma - Parameters for the normal distribution of the Gaussian noise.
    """
    # Создаем пустую матрицу такого же размера, как исходное изображение
    shifted_image = np.zeros_like(image.pic)

    # Сдвигаем изображение вправо
    shifted_image[:, shift_amount:] = image.pic[:, :-shift_amount]

    # Сдвигаем крайние левые столбцы в крайние правые позиции
    shifted_image[:, :shift_amount] = image.pic[:, -shift_amount:]

    # Проходим по всем пикселям сдвинутого изображения
    for y in range(shifted_image.shape[0]):
        for x in range(shifted_image.shape[1]):
            # Добавляем Гауссов шум к пикселям, не равным 255
            if shifted_image[y, x] != 255:
                shifted_image[y, x] = np.clip(np.random.normal(average, sigma), 0, 255)

    # Создаем новый объект Figure с измененным изображением
    shifted_object = Figure(shifted_image, a, b)

    return shifted_object


def shift_image_left(image, a, b):

    shifted_image = np.zeros_like(image.pic)

    # Сдвигаем строки вверх
    shifted_image[:-1, :] = image.pic[1:, :]

    # Сдвигаем последнюю строку в первую
    shifted_image[-1, :] = image.pic[0, :]

    shifted_object = Figure(shifted_image, a, b)

    return shifted_object


def shift_image_up(image, a, b):

    shifted_image = np.zeros_like(image.pic)

    # Сдвигаем столбцы вправо
    shifted_image[:, :-1] = image.pic[:, 1:]

    # Сдвигаем последний столбец в первый
    shifted_image[:, -1] = image.pic[:, 0]

    shifted_object = Figure(shifted_image, a, b)

    return shifted_object


def create_folder_with_name(parent_folder_path, folder_name1, folder_name2):

    new_folder_path = os.path.join(parent_folder_path, folder_name1)
    new_folder_path_1 = os.path.join(parent_folder_path, folder_name2)

    try:
        os.mkdir(new_folder_path)
        os.mkdir(new_folder_path_1)

        return new_folder_path, new_folder_path_1

    except FileExistsError:

        return new_folder_path, new_folder_path_1


    except Exception as e:

        print(f"Произошла ошибка при создании папки: {e}")


def shift_image_down(image, a, b):

    shifted_image = np.zeros_like(image.pic)

    # Сдвигаем столбцы вправо
    shifted_image[:, 1:] = image.pic[:, :-1]
    shifted_image[:, 0] = image.pic[:, -1]

    shifted_image = np.transpose(shifted_image)
    shifted_object = Figure(shifted_image, a, b)

    return shifted_object



def init_structures_worker(args):

    time_array, folder_path, pow1, coef, name, one_event_time, height, width, start_row, end_row, x_start, y_start = args


    big_structures = {}
    "создаем структуры"

    for i in range(start_row, end_row, 1):
        for j in range(0, width, 1):

            color = 0
            big_struct = Structure.__new__(Structure)
            big_struct.__init__(f"{j}_{i}", i, j, color, pow1, coef, time_array, one_event_time) #как в матрицах - первое число номер строки, второе число номер столбца
            big_structures[big_struct.name] = big_struct
    

    return big_structures


def save_object_image(all_struct, height, width, folder_path, name):
    # Создаем массив изображения
    image = np.zeros((height, width), dtype=np.uint8)  # Установим тип данных, чтобы соответствовать требованиям PIL

    # Заполняем массив значениями Conductivity
    for i in range(height):
        for j in range(width):
            # Обратите внимание на исправление порядка индексов
            image[i][j] = all_struct[f"{j}_{i}"].Conductivity
            print(all_struct[f"{j}_{i}"].Conductivity)
    

    #print(image)

    # Создаем объект изображения из массива
    img = Image.fromarray(image, 'L')  # 'L' означает режим оттенков серого

    if isinstance(name, int):
        name = str(name)

    # Проверяем наличие расширения у файла
    if not os.path.splitext(name)[1]:
        name += '.png'  # Добавляем расширение PNG по умолчанию
    # Сохраняем изображение
    img.save(f"{folder_path}/{name}")



def save_object_image_with_gauss(all_struct, a, b, folder_path, name, mean, std):

    image = np.zeros((a, b))


    for j in range(0, a, 1):
        for i in range(0, b, 1):
            image[i][j] = all_struct[f"{i}_{j}"].Conductivity
    

    image = image.transpose()

    # Добавляем Гауссов шум к изображению
    noise = np.random.normal(mean, std, image.shape)
    image_with_noise = image + noise

    im = Figure(image_with_noise, a, b)

    im.save(f"{name}", folder_path)


def process_image_worker(args):

    width, height, start_row, end_row, object_figure_path, one_event_time, big_structures = args

    im = Image.open(object_figure_path).convert('L')
    im = np.array(im)

    "теперь нужно соответствующим образом засветить структуры"
    "засветка определяется параметром при создании самих структур"
    for i in range(start_row, end_row, 1):
        for j in range(0, width, 1):

            cond  = big_structures[f"{i}_{j}"].Conductivity
            "определяем это обьект или фон - если обект - потенциация, если фон - релаксация"
            # print(f"Before: {i}_{j} = {big_structures[f'{i}_{j}'].Conductivity}")
            # Изменяем значение big_structures напрямую

            if im[j][i] != 0:
                # print("First")
                # print(big_structures[f"{i}_{j}"].Conductivity)
                # cond  = big_structures[f"{i}_{j}"].Conductivity
                # print("Cond_start")
                # print(cond)
                cond1 = big_structures[f"{i}_{j}"].impulse(cond)
                # print(cond1)
                # print("After light")
                # print(cond1)
                big_structures[f"{i}_{j}"].Conductivity = cond1
                # print(big_structures[f"{i}_{j}"].Conductivity)
                # print ("Saved in Structure")
                # print(big_structures[f"{i}_{j}"].Conductivity)
                

            else:
                cond1 = big_structures[f"{i}_{j}"].relax(cond, one_event_time)
                big_structures[f"{i}_{j}"].Conductivity = cond1

                #big_structures[f"{i}_{j}"].Conductivity = 0
                # cond = big_structures[f"{i}_{j}"].Conductivity
                # cond1 = big_structures[f"{i}_{j}"].relax(cond, one_event_time)
                # big_structures[f"{i}_{j}"].Conductivity = cond1
    #print("lets check struct")
    # for struct in big_structures.values():
    #     print(struct.Conductivity)


    return big_structures


if __name__ == '__main__':

    start_time = time.time()
    #resize_images_in_directory('/Users/tortolla/Desktop/image_movement', '/Users/tortolla/Desktop/resized')

    #первое число  - ширина, второе число  - высота

    dir_path = '/Users/tortolla/Desktop/resized_with_treshold'

    time_b = int(input("Input time between moves: "))
    path_array = get_sorted_image_paths(dir_path)
    shape_1 = Image.open(path_array[0]).convert('L')
    shape_1 = np.array(shape_1)

    height, width = shape_1.shape

    flag = 0
    pow1 = 8
    coef = 1.7
    time_array = np.arange(5, 51)

    name = 'start'

    x_start = 3
    y_start = 3

    n_processes = 1
    rows_per_process = height // n_processes

    flag = 0

    folder_path_image, folder_path_figure = create_folder_with_name('/Users/tortolla/Desktop/res_of_simulation', f'image:event_time{flag}', f'figure:event_time{flag}')


    all_struct = {}

    radius = 2


    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        args_list = [
            (time_array, folder_path_figure, pow1, coef, name,
             time_b, height, width,
             i * rows_per_process, (i + 1) * rows_per_process if i < n_processes - 1 else height, x_start, y_start)
            for i in range(1)
        ]

        all_struct = {}

        futures = []
        for args in args_list:
            future = executor.submit(init_structures_worker, args)
            futures.append(future)

        for future in futures:
            result_data = future.result()
            all_struct.update(result_data)

    all_keys = list(all_struct.keys())
    #save_object_image(all_struct, height, width, folder_path_image, flag)

    figure_now = image_start('start', folder_path_figure, path_array[0])
    n_processes = 1

    change_time = 0
    index = 0


    for path in path_array:

        flag+=1
        folder_path_image, folder_path_figure = create_folder_with_name('/Users/tortolla/Desktop/res_of_simulation', f'image{flag}', f'figure:event{flag}')
        figure_now = image_start(f'{flag}', folder_path_figure, path)
        figure_now_name = f"{flag}"
        figure_now_name = figure_now_name + '.png'
        figure_now_path = os.path.join(folder_path_figure, figure_now_name)

        with concurrent.futures.ProcessPoolExecutor() as executor:

            args_list = [
                (height, width,
                    i * rows_per_process, (i + 1) * rows_per_process if i < n_processes - 1 else width, figure_now_path, time_b , all_struct)
                for i in range(1)
            ]

            for args in args_list:
                future = executor.submit(process_image_worker, args)
                futures.append(future)

            for future in futures:
                result_data = future.result()
                all_struct.update(result_data)
        
        save_object_image(all_struct, height, width, folder_path_image, flag)
        print(f"{flag}")

    end_time = time.time()
    elapsed_time = end_time - start_time


    print(f"Время выполнения: {elapsed_time} секунд")














