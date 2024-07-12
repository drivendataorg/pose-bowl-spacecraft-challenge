import pandas as pd 
import numpy as np 
import os
from tqdm import tqdm
import random 
from multiprocessing import Pool
import json
from utils import Item, randmix, randmix_small, randmix_big, randmix_max_size, add_antenna

if __name__ == '__main__':
    bg_dict = {}
    bg_image_ids = []

    train_labels_df = pd.read_csv('../dataset/drivendata/train_labels.csv')
    for _, row in tqdm(train_labels_df.iterrows(), total=len(train_labels_df)):
        bg_image_ids.append(row['image_id'])
        image_path = '../dataset/drivendata/images/{}.png'.format(row['image_id'])
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        bg_dict[row['image_id']] = {
            'image_path': image_path,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        } 

    f = open('../dataset/satellite_external/all_bbox.txt', 'r')
    content = f.readline()
    f.close()
    data = json.loads(content)
    for mode in ['train', 'val']:
        for rdir,_, files in os.walk('../dataset/satellite_external/images/{}'.format(mode)):
            for file in tqdm(files):
                image_id = file.split('_')[-1].replace('.png','')
                boxes = np.array(data[image_id], dtype=int)
                if len(boxes) != 1:
                    continue
                image_path = os.path.join(rdir, file)
                x2, y2, x1, y1 = boxes[0]
                bg_image_ids.append('{}_{}_ext'.format(mode, image_id))
                bg_dict['{}_{}_ext'.format(mode, image_id)] = {
                    'image_path': image_path,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                } 

    sc_no_bg_dict = {}
    sc_image_ids = []
    sc_big_100_image_ids = []
    sc_big_200_image_ids = []

    no_bg_df = pd.read_csv('../dataset/drivendata/no_background/no_background.csv')
    for _, row in tqdm(no_bg_df.iterrows(), total=len(no_bg_df)):
        sc_image_ids.append(row['image_id'])
        image_path = '../dataset/drivendata/no_background/images/{}.png'.format(row['image_id'])
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        sc_no_bg_dict[row['image_id']] = {
            'image_path': image_path,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
        if max(x2-x1, y2-y1) > 100:
            sc_big_100_image_ids.append(row['image_id'])
        if max(x2-x1, y2-y1) > 200:
            sc_big_200_image_ids.append(row['image_id'])

    no_bg_ext_df = pd.read_csv('../dataset/satellite_external/no_background/data.csv')
    for _, row in tqdm(no_bg_ext_df.iterrows(), total=len(no_bg_ext_df)):
        image_path = row['image_path']
        image_id = image_path.split('/')[-1].replace('.png','')
        sc_image_ids.append(image_id)
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        sc_no_bg_dict[image_id] = {
            'image_path': image_path,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
        if max(x2-x1, y2-y1) > 100:
            sc_big_100_image_ids.append(image_id)
        if max(x2-x1, y2-y1) > 200:
            sc_big_200_image_ids.append(image_id)

    os.makedirs('../dataset/synthetic_data/randmix/images', exist_ok=True)
    os.makedirs('../dataset/synthetic_data/randmix/labels', exist_ok=True)
    items = []
    total = 2*len(train_labels_df)
    for cnt in range(total):
        bg_image_id = random.choice(bg_image_ids)
        bg_it = bg_dict[bg_image_id]
        bg_image_path = bg_it['image_path']
        bg_box = [bg_it['x1'], bg_it['y1'], bg_it['x2'], bg_it['y2']]

        sc_image_id = random.choice(sc_image_ids)
        sc_it = sc_no_bg_dict[sc_image_id]
        sc_image_path = sc_it['image_path']
        sc_box = [sc_it['x1'], sc_it['y1'], sc_it['x2'], sc_it['y2']]
        
        new_image_path = '../dataset/synthetic_data/randmix/images/{}_{}_{}.png'.format(bg_image_id, sc_image_id, cnt)
        new_label_path = '../dataset/synthetic_data/randmix/labels/{}_{}_{}.txt'.format(bg_image_id, sc_image_id, cnt)

        items.append(Item(cnt, total, bg_image_id, bg_image_path, bg_box, sc_image_id, sc_image_path, sc_box, new_image_path, new_label_path))

    p = Pool(32)
    results = p.map(func=randmix, iterable = items)
    p.close()
    train_file = open('../dataset/synthetic_data/randmix/train.txt', 'w')
    for r in results:
        if r is not None:
            train_file.write(r+'\n')
    train_file.close()

    os.makedirs('../dataset/synthetic_data/randmix_small/images', exist_ok=True)
    os.makedirs('../dataset/synthetic_data/randmix_small/labels', exist_ok=True)
    items = []
    total = len(train_labels_df)
    for cnt in range(total):
        bg_image_id = random.choice(bg_image_ids)
        bg_it = bg_dict[bg_image_id]
        bg_image_path = bg_it['image_path']
        bg_box = [bg_it['x1'], bg_it['y1'], bg_it['x2'], bg_it['y2']]

        sc_image_id = random.choice(sc_image_ids)
        sc_it = sc_no_bg_dict[sc_image_id]
        sc_image_path = sc_it['image_path']
        sc_box = [sc_it['x1'], sc_it['y1'], sc_it['x2'], sc_it['y2']]
        
        new_image_path = '../dataset/synthetic_data/randmix_small/images/{}_{}_{}.png'.format(bg_image_id, sc_image_id, cnt)
        new_label_path = '../dataset/synthetic_data/randmix_small/labels/{}_{}_{}.txt'.format(bg_image_id, sc_image_id, cnt)

        items.append(Item(cnt, total, bg_image_id, bg_image_path, bg_box, sc_image_id, sc_image_path, sc_box, new_image_path, new_label_path))

    p = Pool(32)
    results = p.map(func=randmix_small, iterable = items)
    p.close()
    train_file = open('../dataset/synthetic_data/randmix_small/train.txt', 'w')
    for r in results:
        if r is not None:
            train_file.write(r+'\n')
    train_file.close()

    os.makedirs('../dataset/synthetic_data/randmix_big/images', exist_ok=True)
    os.makedirs('../dataset/synthetic_data/randmix_big/labels', exist_ok=True)
    items = []
    total = int(0.8*len(train_labels_df))
    for cnt in range(total):
        bg_image_id = random.choice(bg_image_ids)
        bg_it = bg_dict[bg_image_id]
        bg_image_path = bg_it['image_path']
        bg_box = [bg_it['x1'], bg_it['y1'], bg_it['x2'], bg_it['y2']]

        sc_image_id = random.choice(sc_big_100_image_ids)
        sc_it = sc_no_bg_dict[sc_image_id]
        sc_image_path = sc_it['image_path']
        sc_box = [sc_it['x1'], sc_it['y1'], sc_it['x2'], sc_it['y2']]
        
        new_image_path = '../dataset/synthetic_data/randmix_big/images/{}_{}_{}.png'.format(bg_image_id, sc_image_id, cnt)
        new_label_path = '../dataset/synthetic_data/randmix_big/labels/{}_{}_{}.txt'.format(bg_image_id, sc_image_id, cnt)

        items.append(Item(cnt, total, bg_image_id, bg_image_path, bg_box, sc_image_id, sc_image_path, sc_box, new_image_path, new_label_path))

    p = Pool(32)
    results = p.map(func=randmix_big, iterable = items)
    p.close()
    train_file = open('../dataset/synthetic_data/randmix_big/train.txt', 'w')
    for r in results:
        if r is not None:
            train_file.write(r+'\n')
    train_file.close()


    os.makedirs('../dataset/synthetic_data/randmix_max_size/images', exist_ok=True)
    os.makedirs('../dataset/synthetic_data/randmix_max_size/labels', exist_ok=True)
    items = []
    total = int(0.2*len(train_labels_df))
    for cnt in range(total):
        bg_image_id = random.choice(bg_image_ids)
        bg_it = bg_dict[bg_image_id]
        bg_image_path = bg_it['image_path']
        bg_box = [bg_it['x1'], bg_it['y1'], bg_it['x2'], bg_it['y2']]

        sc_image_id = random.choice(sc_big_200_image_ids)
        sc_it = sc_no_bg_dict[sc_image_id]
        sc_image_path = sc_it['image_path']
        sc_box = [sc_it['x1'], sc_it['y1'], sc_it['x2'], sc_it['y2']]
        
        new_image_path = '../dataset/synthetic_data/randmix_max_size/images/{}_{}_{}.png'.format(bg_image_id, sc_image_id, cnt)
        new_label_path = '../dataset/synthetic_data/randmix_max_size/labels/{}_{}_{}.txt'.format(bg_image_id, sc_image_id, cnt)

        items.append(Item(cnt, total, bg_image_id, bg_image_path, bg_box, sc_image_id, sc_image_path, sc_box, new_image_path, new_label_path))

    p = Pool(32)
    results = p.map(func=randmix_max_size, iterable = items)
    p.close()
    train_file = open('../dataset/synthetic_data/randmix_max_size/train.txt', 'w')
    for r in results:
        if r is not None:
            train_file.write(r+'\n')
    train_file.close()

    os.makedirs('../dataset/synthetic_data/add_antenna/images', exist_ok=True)
    os.makedirs('../dataset/synthetic_data/add_antenna/labels', exist_ok=True)
    items = []
    total = 2500
    for cnt in range(total):
        bg_image_id = random.choice(bg_image_ids)
        bg_it = bg_dict[bg_image_id]
        bg_image_path = bg_it['image_path']
        bg_box = [bg_it['x1'], bg_it['y1'], bg_it['x2'], bg_it['y2']]

        sc_image_id = random.choice(sc_image_ids)
        sc_it = sc_no_bg_dict[sc_image_id]
        sc_image_path = sc_it['image_path']
        sc_box = [sc_it['x1'], sc_it['y1'], sc_it['x2'], sc_it['y2']]

        new_image_path = '../dataset/synthetic_data/add_antenna/images/{}_{}_{}.png'.format(bg_image_id, sc_image_id, cnt)
        new_label_path = '../dataset/synthetic_data/add_antenna/labels/{}_{}_{}.txt'.format(bg_image_id, sc_image_id, cnt)

        items.append(Item(cnt, total, bg_image_id, bg_image_path, bg_box, sc_image_id, sc_image_path, sc_box, new_image_path, new_label_path))

    p = Pool(32)
    results = p.map(func=add_antenna, iterable = items)
    p.close()

    train_file = open('../dataset/synthetic_data/add_antenna/train.txt', 'w')
    for r in results:
        if r is not None:
            train_file.write(r+'\n')
    train_file.close()
