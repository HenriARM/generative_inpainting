import numpy as np
from wand.image import Image

def main():
    f = '/home/rudolfs/Desktop/panos/pan-19-05-2021/CAM00022P1-PR0364-PAN06_hdrnet.jpg'
    with Image(filename=f) as img:
        print(img.width, img.height)
        img.crop(10, 20, width=256, height=256)
        print(img.width, img.height)

if __name__ == "__main__":
    main()
