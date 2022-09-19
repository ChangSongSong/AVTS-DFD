import os
from PIL import Image


def pil_loader(img_path, grayscale=False):
    if grayscale:
        return Image.open(img_path).convert('L')
    else:
        return Image.open(img_path).convert('RGB')

'''
Borrow from https://github.com/chail/patch-forensics/blob/master/data/dataset_util.py
'''

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(img_dir, max_dataset_size=float("inf")):
    images = []
    labels = []
    assert os.path.isdir(img_dir), '%s is not a valid directory' % img_dir
    for root, _, fnames in sorted(os.walk(img_dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                label = path.split('/')[-3]
                images.append(path)
                labels.append(label)
    image_list = images[:min(max_dataset_size, len(images))]
    label_list = labels[:min(max_dataset_size, len(labels))]
    return image_list, label_list
