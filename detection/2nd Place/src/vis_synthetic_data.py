import os 
import cv2 
import random

if __name__ == '__main__':
    num_images = 10
    for gen_type in ['add_antenna','randmix','randmix_big','randmix_max_size','randmix_small']:
        os.makedirs('vis/' + gen_type, exist_ok=True)
        f = open('../dataset/synthetic_data/{}/train.txt'.format(gen_type), 'r')
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        f.close()
        lines = random.sample(lines, num_images)
        cnt = 0
        for image_path in lines:
            label_path = image_path.replace('images','labels').replace('.png','.txt')
            image = cv2.imread(image_path)
            height, width = image.shape[0:2]
            f = open(label_path, 'r')
            lines = f.readlines()
            lines = [x.strip() for x in lines]
            for idx, line in enumerate(lines):
                arr = line.split(' ')
                xc, yc, w, h = float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4])
                x1 = int((xc-0.5*w)*width)
                x2 = int((xc+0.5*w)*width)
                y1 = int((yc-0.5*h)*height)
                y2 = int((yc+0.5*h)*height)
                if idx == 0:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 1)
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 1)
            f.close()
            cv2.imwrite('vis/{}/{}.jpg'.format(gen_type,cnt), image)
            cnt += 1