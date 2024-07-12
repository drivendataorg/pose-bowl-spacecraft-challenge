import pandas as pd 
import numpy as np 
import os 
from tqdm import tqdm
import cv2 

if __name__ == "__main__":
    ### train val split
    df = pd.read_csv('../dataset/drivendata/train_metadata.csv')
    val_df = df.loc[df.spacecraft_id.isin([30,13,24,2])]
    train_df = df.loc[~df.spacecraft_id.isin([30,13,24,2])]

    train_df = train_df.sample(frac=1).reset_index(drop=True)

    train_file = open('../dataset/drivendata/train.txt', 'w')
    for image_id in np.unique(train_df.image_id.values):
        train_file.write('../dataset/drivendata/images/{}.png\n'.format(image_id))
    train_file.close()

    val_file = open('../dataset/drivendata/val.txt', 'w')
    for image_id in np.unique(val_df.image_id.values):
        val_file.write('../dataset/drivendata/images/{}.png\n'.format(image_id))
    val_file.close()

    ### convert to yolo annotation
    train_label_df = pd.read_csv('../dataset/drivendata/train_labels.csv')
    os.makedirs('../dataset/drivendata/labels', exist_ok=True)
    for image_id, grp in tqdm(train_label_df.groupby('image_id')):
        if len(grp) != 1:
            print(image_id, len(grp))
        image_path = '../dataset/drivendata/images/{}.png'.format(image_id)
        if os.path.isfile(image_path) == False:
            print(image_path, 'not found')
            continue
        image = cv2.imread(image_path)
        label_file = open('../dataset/drivendata/labels/{}.txt'.format(image_id), 'w')

        height, width = image.shape[0:2]

        for _, row in grp.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            if x1 >= x2 or y1 >= y2:
                print(image_id, "x1 >= x2 or y1 >= y2")
            xc = 0.5*(x1+x2)/width
            yc = 0.5*(y1+y2)/height
            w = (x2-x1)/width
            h = (y2-y1)/height
            label_file.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
        label_file.close()