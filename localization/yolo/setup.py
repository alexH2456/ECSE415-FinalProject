import os
import shutil
import PIL.Image as pimg

mio_path = 'MIO-TCD-Localization'
img_path = os.path.join(mio_path, 'train')
csv_path = os.path.join(mio_path, 'gt_train.csv')

# TODO: IDK WHICH FOLDER TO OUTPUT TO
dn_path = 'darknet'
data_path = os.path.join(dn_path, 'data')

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

# Delete all label text files
for txt in os.listdir(img_path):
    if txt.endswith('.txt'):
        os.remove(os.path.join(img_path, txt))

# Create label text files
with open(csv_path, 'r') as csv:
    for line in csv:
        name, obj, x1, y1, x2, y2 = line.strip().split(',')
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        name_txt = f'{name}.txt'
        name_jpg = f'{name}.jpg'

        img_w, img_h = pimg.open(os.path.join(img_path, name_jpg)).size

        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h

        with open(os.path.join(img_path, name_txt), 'a') as txt:
            txt.write(f'{obj_ids[obj]} {x_center} {y_center} {w} {h}\n')

# Create train text file with image paths
img_names = sorted(os.listdir(img_path))
with open(f'{img_path}.txt', 'w') as train:
    for img in img_names:
        train.write(os.path.join(img_path, img) + '\n')
