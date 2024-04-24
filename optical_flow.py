import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def draw_optical_flow(img, flow, step=4):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Создаём изображение и рисуем на нём стрелки
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x0, y0), (fx, fy) in zip(np.stack((x, y), axis=-1), np.stack((fx, fy), axis=-1)):
        cv2.arrowedLine(vis, (x0, y0), (x0+int(fx), y0+int(fy)), (0, 255, 0), 1, tipLength=0.3)
    return vis

def compute_optical_flow(img_path1, img_path2, output_folder, output_filename):
    # Загрузка изображений
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    # Проверка на успешность загрузки
    if img1 is None or img2 is None:
        raise ValueError("One of the images didn't load correctly.")

    # Вычисление оптического потока
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Визуализация результатов
    vis = draw_optical_flow(img1, flow)

    # Сохранение результата в файл
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, output_filename)
    plt.figure(figsize=(10, 10))
    plt.imshow(vis)
    plt.axis('off')  # Убираем оси координат
    plt.title('Optical Flow')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved optical flow visualization to {output_path}")

# Пример использования:
compute_optical_flow('/Users/tortolla/Desktop/sensor/image1_1.png', '/Users/tortolla/Desktop/sensor/image2_2.png', '/Users/tortolla/Desktop/optical_flow', 'sensor_1.png')
compute_optical_flow('/Users/tortolla/Desktop/sensor/image2_2.png', '/Users/tortolla/Desktop/sensor/image3_3.png', '/Users/tortolla/Desktop/optical_flow', 'sensor_2.png')
compute_optical_flow('/Users/tortolla/Desktop/sensor/image3_3.png', '/Users/tortolla/Desktop/sensor/image4_4.png', '/Users/tortolla/Desktop/optical_flow', 'sensor_3.png')
compute_optical_flow('/Users/tortolla/Desktop/sensor/image5_5.png', '/Users/tortolla/Desktop/sensor/image6_6.png', '/Users/tortolla/Desktop/optical_flow', 'sensor_4.png')
