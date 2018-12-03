import os
import sys
import time
import shutil
import PIL.Image as pimg

cwd = os.path.dirname(sys.argv(0))
data_path = os.path.join(cwd, 'data')
img_path = os.path.join(data_path, 'train')
csv_path = os.path.join(data_path, 'gt_train.csv')

obj_ids = {
    'articulated_truck': '0',
    'bicycle': '1',
    'bus': '2',
    'car': '3',
    'motorcycle': '4',
    'motorized_vehicle': '5',
    'non-motorized_vehicle': '6',
    'pedestrian': '7',
    'pickup_truck': '8',
    'single_unit_truck': '9',
    'work_van': '10'
}

start = time.time()

# Delete all existing labels
for txt in os.listdir(img_path):
    if txt.endswith('.txt'):
        os.remove(os.path.join(img_path, txt))

# Create train text file with image paths
img_names = sorted(os.listdir(img_path))
num_images = len(img_names)
num_train = int(num_images * 0.8)

with open(os.path.join(data_path, 'train.txt'), 'w') as train:
    for img in img_names[:num_train]:
        train.write(os.path.join('data/train', img) + '\n')

with open(os.path.join(data_path, 'test.txt'), 'w') as test:
    for img in img_names[num_train:]:
        test.write(os.path.join('data/train', img) + '\n')

# Create label text files
with open(csv_path, 'r') as csv:
    for line in csv:
        name, obj, x1, y1, x2, y2 = line.strip().split(',')
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        name_txt = f'{name}.txt'
        name_jpg = f'{name}.jpg'

        jpg_path = os.path.join(img_path, name_jpg)
        txt_path = os.path.join(img_path, name_txt)

        img_w, img_h = pimg.open(jpg_path).size

        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h

        with open(txt_path, 'a') as txt:
            txt.write(f'{obj_ids[obj]} {x_center} {y_center} {w} {h}\n')

print(f'Execution time: {time.time() - start} s')
