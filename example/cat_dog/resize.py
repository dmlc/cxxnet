from PIL import Image
import numpy as np
import os
from multiprocessing import Pool

input = "./train/"
output = "./resized_train/"


def proc(name):
    img = Image.open(input + name)
    w, h = img.size
    size = (256, 'auto') if h > w else ('auto', 256)
    img.thumbnail(size, Image.ANTIALIAS)
    w, h = img.size
    left = (w - 256)/2
    top = (h - 256)/2
    right = (w + 256)/2
    bottom = (h + 256)/2
    img = img.crop((left, top, right, bottom))
    # array = np.asarray(img)
    # a = array[:,:,0] - np.mean(array[:,:,0])
    # b = array[:,:,1] - np.mean(array[:,:,1])
    # c = array[:,:,2] - np.mean(array[:,:,2])
    # array = np.dstack((a, b, c))
    # better to have npy iterator, for jpg doesn't support neg number
    img.save(output + name, "JPEG")

if __name__ == '__main__':
    lst = os.listdir(input)
    p = Pool(4)
    p.map(proc, lst)



