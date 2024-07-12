import pandas as pd 
import numpy as np 
import os 
import cv2 
from tqdm import tqdm
import json

def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    iou = interArea / boxAArea
    return iou

if __name__ == '__main__':
    f = open('../dataset/satellite_external/all_bbox.txt', 'r')
    content = f.readline()
    f.close()
    data = json.loads(content)
    
    train_file = open('../dataset/satellite_external/train_val.txt', 'w')
    for mode in ['train', 'val']:
        os.makedirs('../dataset/satellite_external/labels/{}'.format(mode), exist_ok=True)
        for rdir,_, files in os.walk('../dataset/satellite_external/images/{}'.format(mode)):
            for file in tqdm(files):
                image_id = file.split('_')[-1].replace('.png','')
                boxes = np.array(data[image_id], dtype=int)
                
                label_path = '../dataset/satellite_external/labels/{}/{}'.format(mode, file.replace('.png','.txt'))
                label_file = open(label_path, 'w')

                image_path = os.path.join(rdir, file)
                image = cv2.imread(image_path)
                height, width = image.shape[0:2]
                for box in boxes:
                    x2, y2, x1, y1 = box
                    x1 = min(max(0, x1), width)
                    x2 = min(max(0, x2), width)
                    y1 = min(max(0, y1), height)
                    y2 = min(max(0, y2), height)
                    if x1 >= x2 or y1 >= y2:
                        continue

                    xc = 0.5*(x1+x2)/width
                    yc = 0.5*(y1+y2)/height
                    w = (x2-x1)/width
                    h = (y2-y1)/height
                    label_file.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
                label_file.close()
                train_file.write(image_path + "\n")
    train_file.close()

    f = open('../dataset/satellite_external/all_bbox.txt', 'r')
    content = f.readline()
    f.close()
    data = json.loads(content)
    
    no_background_meta = []
   
    for mode in ['train', 'val']:
        os.makedirs('../dataset/satellite_external/no_background/{}'.format(mode), exist_ok=True)
        for rdir,_, files in os.walk('../dataset/satellite_external/images/{}'.format(mode)):
            for file in tqdm(files):
                image_id = file.split('_')[-1].replace('.png','')

                image_path = os.path.join(rdir, file)
                image = cv2.imread(image_path)
                height, width = image.shape[0:2]

                mask_path = image_path.replace('images', 'mask').replace('.png','_mask.png')
                mask = cv2.imread(mask_path).mean(-1)
                mask = np.where(mask>0,1,0)
                mask = mask[..., None]

                boxes = np.array(data[image_id], dtype=int)
                
                for i in range(len(boxes)):
                    x2, y2, x1, y1 = boxes[i]

                    x1 = min(max(0, x1), width)
                    x2 = min(max(0, x2), width)
                    y1 = min(max(0, y1), height)
                    y2 = min(max(0, y2), height)
                    if x1 >= x2 or y1 >= y2:
                        continue

                    check = True
                    for j in range(len(boxes)):
                        if j == i:
                            continue
                        x22, y22, x12, y12 = boxes[j]
                        iou = bb_iou([x1, y1, x2, y2],[x12, y12, x22, y22])
                        if iou > 0 and iou < 1:
                            check = False
                    if check == False:
                        continue
                    
                    mask_tmp = np.zeros((height, width, 1), dtype=np.uint8)
                    mask_tmp[y1:y2,x1:x2,:] = np.ones((y2-y1,x2-x1,1), dtype=np.uint8)

                    image_i = image.copy()
                    image_i = image_i*mask_tmp

                    mask_i = mask.copy()
                    mask_i = mask_i*mask_tmp

                    image_i = image_i*mask_i

                    image_i = image_i.astype(np.uint8)
                    mask_i = (mask_i*255).astype(np.uint8)
                    image_i = np.concatenate((image_i, mask_i), -1)

                    new_image_path = '../dataset/satellite_external/no_background/{}/{}_{}.png'.format(mode, image_id, i)
                    cv2.imwrite(new_image_path, image_i)

                    no_background_meta.append([new_image_path, x1, y1, x2, y2])

    no_background_df = pd.DataFrame(data=np.array(no_background_meta), columns=['image_path', 'x1', 'y1', 'x2', 'y2'])
    no_background_df[['x1','y1','x2','y2']] = no_background_df[['x1','y1','x2','y2']].astype(int)
    no_background_df.to_csv('../dataset/satellite_external/no_background/data.csv', index=False)
