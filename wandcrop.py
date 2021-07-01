import numpy as np
import wand.image
from cv2 import cv2
from PIL import Image
from time import process_time
from pympler import asizeof

def print_res(img, crop, text, t_start, t_stop):
    print(f'{text } time:', t_stop - t_start)
    print('img size', asizeof.asizeof(img))
    print('crop size', asizeof.asizeof(crop))
    print()

def main():
    f = '/home/rudolfs/Desktop/panos/pan-19-05-2021/CAM00022P1-PR0364-PAN06_hdrnet.jpg'
    x, y = 10, 10
    W, H = 256, 256

    # OpenCV
    t_start = process_time()
    img = cv2.imread(f)
    crop = img[y:y+H, x:x + W].copy()
    t_stop = process_time()
    print_res(img, crop, 'OpenCV', t_start, t_stop)

    # Crop by PIL
    t_start = process_time()
    img = Image.open(f)
    crop = img.crop((x, y, x+W, x+H))
    t_stop = process_time()
    print_res(img, crop, 'PIL', t_start, t_stop)

    # ImageMagick
    t_start = process_time()
    img = wand.image.Image(filename=f)
    crop = img.clone()
    crop.crop(x, y, width=W, height=H)
    t_stop = process_time()
    print_res(img, crop, 'ImageMagick', t_start, t_stop)


if __name__ == "__main__":
    main()
