import numpy as np 
import cv2 
import random 
import albumentations as albu

albu_transform_sc = albu.Compose([
    albu.Affine(rotate=(-90, 90), shear=(-30, 30), p=1),
    albu.RandomBrightnessContrast(p=1),
])

def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def bb_iom(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iom = interArea / min(boxAArea, boxBArea)
    return iom

def move_sc_to_center(image, x1, y1, x2, y2):
    height, width = image.shape[0:2]
    xc = 0.5*(x1+x2)
    yc = 0.5*(y1+y2)
    bw, bh = x2-x1, y2-y1 

    new_image = np.zeros((height, width, 4), dtype=np.uint8)
    n_x1 = int(width/2-bw/2)
    n_x2 = n_x1 + bw
    n_y1 = int(height/2-bh/2)
    n_y2 = n_y1 + bh
    new_image[n_y1:n_y2,n_x1:n_x2,:] = image[y1:y2,x1:x2,:]
    return new_image, n_x1, n_y1, n_x2, n_y2

def rand_crop_sc(sc_image, sc_x1, sc_y1, sc_x2, sc_y2):
    sc_image, sc_x1, sc_y1, sc_x2, sc_y2 = move_sc_to_center(sc_image, sc_x1, sc_y1, sc_x2, sc_y2)
    sc_bgr = sc_image[:,:,0:3].astype(np.uint8)
    sc_alpha = sc_image[:,:,3].astype(np.float32)

    transformed = albu_transform_sc(image=sc_bgr, mask=sc_alpha)
    new_sc_bgr = transformed["image"].astype(np.uint8)
    new_sc_alpha = transformed["mask"].astype(np.uint8)
    thresh = cv2.threshold(new_sc_alpha, 0, 255, cv2.THRESH_OTSU)[1]
    x,y,w,h = cv2.boundingRect(thresh)
    if w>0 and h>0:
        sc_x1, sc_y1, sc_x2, sc_y2 = x, y, x+w, y+h
        new_sc_alpha = new_sc_alpha[..., np.newaxis]
        sc_image = np.concatenate((new_sc_bgr, new_sc_alpha), -1)
    sc_image = sc_image[sc_y1:sc_y2, sc_x1:sc_x2,:].astype(np.uint8)
    return sc_image

class Item:
    def __init__(self, index, total, bg_image_id, bg_image_path, bg_box, sc_image_id, sc_image_path, sc_box, new_image_path, new_label_path):
        self.index = index
        self.total = total
        self.bg_image_id = bg_image_id
        self.bg_image_path = bg_image_path
        self.bg_box = bg_box
        self.sc_image_id = sc_image_id
        self.sc_image_path = sc_image_path
        self.sc_box = sc_box
        self.new_image_path = new_image_path
        self.new_label_path = new_label_path

def randmix(it):
    print('*'*10, 'randmix', it.index, it.total, int(100*it.index/it.total), '*'*10)
    bg_image = cv2.imread(it.bg_image_path)
    bg_height, bg_width = bg_image.shape[0:2]
    gen_boxes = [it.bg_box]

    sc_image = cv2.imread(it.sc_image_path, cv2.IMREAD_UNCHANGED)
    sc_x1, sc_y1, sc_x2, sc_y2 = it.sc_box
    sc_image = rand_crop_sc(sc_image, sc_x1, sc_y1, sc_x2, sc_y2)
    sc_height, sc_width = sc_image.shape[0:2]
    sc_bgr = sc_image[:,:,0:3].astype(np.uint8)
    sc_alpha = sc_image[:,:,3].astype(np.float32)

    cnt = 0
    while True:
        try:
            if bg_width > sc_width:
                ran_px = random.randint(0, bg_width-sc_width)
            else:
                ran_px = 0
            if bg_height > sc_height:
                ran_py = random.randint(0, bg_height-sc_height)
            else:
                ran_py = 0
            if bb_iom([ran_px, ran_py, ran_px+sc_width, ran_py+sc_height], it.bg_box) < 0.1:
                bg_crop = bg_image.copy()[ran_py:ran_py+sc_height, ran_px:ran_px+sc_width]

                mask = sc_alpha/255.0
                mask = mask[..., None]
                composite = sc_bgr.astype(np.float32) * mask + bg_crop.astype(np.float32) * (1-mask)
                composite = composite.astype(np.uint8)
                                
                bg_image[ran_py:ran_py+sc_height, ran_px:ran_px+sc_width] = composite

                gen_boxes.append([ran_px, ran_py, ran_px+sc_width, ran_py+sc_height])

                label_file = open(it.new_label_path, 'w')

                for k, gen_box in enumerate(gen_boxes):
                    x1, y1, x2, y2 = gen_box
                    # if k == 0:
                    #     cv2.rectangle(bg_image, (x1, y1), (x2, y2), (255,0,0), 2)
                    # else:
                    #     cv2.rectangle(bg_image, (x1, y1), (x2, y2), (0,255,0), 1)

                    xc = 0.5*(x1+x2)/bg_width
                    yc = 0.5*(y1+y2)/bg_height
                    w = (x2-x1)/bg_width
                    h = (y2-y1)/bg_height
                    label_file.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
                label_file.close()
                
                cv2.imwrite(it.new_image_path, bg_image)
                return it.new_image_path
            else:
                cnt += 1
        except:
            cnt += 1
        
        if cnt > 50:
            break
    return None


def randmix_small(it):
    print('*'*10, 'randmix_small', it.index, it.total, int(100*it.index/it.total), '*'*10)
    bg_image = cv2.imread(it.bg_image_path)
    bg_height, bg_width = bg_image.shape[0:2]
    gen_boxes = [it.bg_box]

    sc_image = cv2.imread(it.sc_image_path, cv2.IMREAD_UNCHANGED)
    sc_x1, sc_y1, sc_x2, sc_y2 = it.sc_box
    sc_image = rand_crop_sc(sc_image, sc_x1, sc_y1, sc_x2, sc_y2)

    new_size = random.randint(10, 64)

    albu_transform_resize = albu.LongestMaxSize(max_size=new_size, p=1)
    sc_bgr = sc_image[:,:,0:3].astype(np.uint8)
    sc_alpha = sc_image[:,:,3].astype(np.float32)
    transformed = albu_transform_resize(image=sc_bgr, mask=sc_alpha)
    sc_bgr = transformed["image"].astype(np.uint8)
    sc_alpha = transformed["mask"].astype(np.float32)
    sc_height, sc_width = sc_bgr.shape[0:2]

    cnt = 0
    while True:
        try:
            if bg_width > sc_width:
                ran_px = random.randint(0, bg_width-sc_width)
            else:
                ran_px = 0
            if bg_height > sc_height:
                ran_py = random.randint(0, bg_height-sc_height)
            else:
                ran_py = 0
            if bb_iom([ran_px, ran_py, ran_px+sc_width, ran_py+sc_height], it.bg_box) < 0.1:
                bg_crop = bg_image.copy()[ran_py:ran_py+sc_height, ran_px:ran_px+sc_width]

                mask = sc_alpha/255.0
                mask = mask[..., None]
                composite = sc_bgr.astype(np.float32) * mask + bg_crop.astype(np.float32) * (1-mask)
                composite = composite.astype(np.uint8)
                                
                bg_image[ran_py:ran_py+sc_height, ran_px:ran_px+sc_width] = composite

                gen_boxes.append([ran_px, ran_py, ran_px+sc_width, ran_py+sc_height])

                label_file = open(it.new_label_path, 'w')

                for k, gen_box in enumerate(gen_boxes):
                    x1, y1, x2, y2 = gen_box
                    # if k == 0:
                    #     cv2.rectangle(bg_image, (x1, y1), (x2, y2), (255,0,0), 2)
                    # else:
                    #     cv2.rectangle(bg_image, (x1, y1), (x2, y2), (0,255,0), 1)

                    xc = 0.5*(x1+x2)/bg_width
                    yc = 0.5*(y1+y2)/bg_height
                    w = (x2-x1)/bg_width
                    h = (y2-y1)/bg_height
                    label_file.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
                label_file.close()
                
                cv2.imwrite(it.new_image_path, bg_image)
                return it.new_image_path
            else:
                cnt += 1
        except:
            cnt += 1
        if cnt > 50:
            break
    return None



def randmix_big(it):
    print('*'*10, 'randmix_big', it.index, it.total, int(100*it.index/it.total), '*'*10)
    bg_image = cv2.imread(it.bg_image_path)
    bg_height, bg_width = bg_image.shape[0:2]
    gen_boxes = [it.bg_box]

    sc_image = cv2.imread(it.sc_image_path, cv2.IMREAD_UNCHANGED)
    sc_x1, sc_y1, sc_x2, sc_y2 = it.sc_box
    sc_image = rand_crop_sc(sc_image, sc_x1, sc_y1, sc_x2, sc_y2)
    sc_height, sc_width = sc_image.shape[0:2]

    if sc_height > sc_width:
        new_size = random.randint(350, bg_height)
    else:
        new_size = random.randint(350, bg_width)
    if new_size > 4*max(sc_width, sc_height):
        new_size = 4*max(sc_width, sc_height)
    
    albu_transform_resize = albu.LongestMaxSize(max_size=new_size, p=1)
    sc_bgr = sc_image[:,:,0:3].astype(np.uint8)
    sc_alpha = sc_image[:,:,3].astype(np.float32)
    transformed = albu_transform_resize(image=sc_bgr, mask=sc_alpha)
    sc_bgr = transformed["image"].astype(np.uint8)
    sc_alpha = transformed["mask"].astype(np.float32)
    sc_height, sc_width = sc_bgr.shape[0:2]

    cnt = 0
    while True:
        try:
            if bg_width > sc_width:
                ran_px = random.randint(0, bg_width-sc_width)
            else:
                ran_px = 0
            if bg_height > sc_height:
                ran_py = random.randint(0, bg_height-sc_height)
            else:
                ran_py = 0
            if bb_iom([ran_px, ran_py, ran_px+sc_width, ran_py+sc_height], it.bg_box) < 0.1:
                bg_crop = bg_image.copy()[ran_py:ran_py+sc_height, ran_px:ran_px+sc_width]

                mask = sc_alpha/255.0
                mask = mask[..., None]
                composite = sc_bgr.astype(np.float32) * mask + bg_crop.astype(np.float32) * (1-mask)
                composite = composite.astype(np.uint8)
                                
                bg_image[ran_py:ran_py+sc_height, ran_px:ran_px+sc_width] = composite

                gen_boxes.append([ran_px, ran_py, ran_px+sc_width, ran_py+sc_height])

                label_file = open(it.new_label_path, 'w')

                for k, gen_box in enumerate(gen_boxes):
                    x1, y1, x2, y2 = gen_box
                    # if k == 0:
                    #     cv2.rectangle(bg_image, (x1, y1), (x2, y2), (255,0,0), 2)
                    # else:
                    #     cv2.rectangle(bg_image, (x1, y1), (x2, y2), (0,255,0), 1)

                    xc = 0.5*(x1+x2)/bg_width
                    yc = 0.5*(y1+y2)/bg_height
                    w = (x2-x1)/bg_width
                    h = (y2-y1)/bg_height
                    label_file.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
                label_file.close()
                
                cv2.imwrite(it.new_image_path, bg_image)
                return it.new_image_path
            else:
                cnt += 1
        except:
            cnt += 1
        if cnt > 50:
            break
    return None


def randmix_max_size(it):
    print('*'*10, 'randmix_max_size', it.index, it.total, int(100*it.index/it.total), '*'*10)
    bg_image = cv2.imread(it.bg_image_path)
    bg_height, bg_width = bg_image.shape[0:2]
    gen_boxes = [it.bg_box]

    sc_image = cv2.imread(it.sc_image_path, cv2.IMREAD_UNCHANGED)
    sc_x1, sc_y1, sc_x2, sc_y2 = it.sc_box
    sc_image = rand_crop_sc(sc_image, sc_x1, sc_y1, sc_x2, sc_y2)
    sc_height, sc_width = sc_image.shape[0:2]

    if sc_height > sc_width:
        new_size = bg_height
    else:
        new_size = bg_width
    
    albu_transform_resize = albu.LongestMaxSize(max_size=new_size, p=1)
    sc_bgr = sc_image[:,:,0:3].astype(np.uint8)
    sc_alpha = sc_image[:,:,3].astype(np.float32)
    transformed = albu_transform_resize(image=sc_bgr, mask=sc_alpha)
    sc_bgr = transformed["image"].astype(np.uint8)
    sc_alpha = transformed["mask"].astype(np.float32)
    sc_height, sc_width = sc_bgr.shape[0:2]

    cnt = 0
    while True:
        try:
            if bg_width > sc_width:
                ran_px = random.randint(0, bg_width-sc_width)
            else:
                ran_px = 0
            if bg_height > sc_height:
                ran_py = random.randint(0, bg_height-sc_height)
            else:
                ran_py = 0
            if bb_iom([ran_px, ran_py, ran_px+sc_width, ran_py+sc_height], it.bg_box) < 0.1:
                bg_crop = bg_image.copy()[ran_py:ran_py+sc_height, ran_px:ran_px+sc_width]

                mask = sc_alpha/255.0
                mask = mask[..., None]
                composite = sc_bgr.astype(np.float32) * mask + bg_crop.astype(np.float32) * (1-mask)
                composite = composite.astype(np.uint8)
                                
                bg_image[ran_py:ran_py+sc_height, ran_px:ran_px+sc_width] = composite

                gen_boxes.append([ran_px, ran_py, ran_px+sc_width, ran_py+sc_height])

                label_file = open(it.new_label_path, 'w')

                for k, gen_box in enumerate(gen_boxes):
                    x1, y1, x2, y2 = gen_box
                    # if k == 0:
                    #     cv2.rectangle(bg_image, (x1, y1), (x2, y2), (255,0,0), 2)
                    # else:
                    #     cv2.rectangle(bg_image, (x1, y1), (x2, y2), (0,255,0), 1)

                    xc = 0.5*(x1+x2)/bg_width
                    yc = 0.5*(y1+y2)/bg_height
                    w = (x2-x1)/bg_width
                    h = (y2-y1)/bg_height
                    label_file.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
                label_file.close()
                
                cv2.imwrite(it.new_image_path, bg_image)
                return it.new_image_path
            else:
                cnt += 1
        except:
            cnt += 1
        if cnt > 50:
            break
    return None


def add_antenna(it):
    print('*'*10, 'add_antenna', it.index, it.total, int(100*it.index/it.total), '*'*10)
    bg_image = cv2.imread(it.bg_image_path)
    bg_height, bg_width = bg_image.shape[0:2]
    gen_boxes = [it.bg_box]

    sc_image = cv2.imread(it.sc_image_path, cv2.IMREAD_UNCHANGED)
    sc_x1, sc_y1, sc_x2, sc_y2 = it.sc_box
    sc_image = rand_crop_sc(sc_image, sc_x1, sc_y1, sc_x2, sc_y2)
    
    albu_transform_resize = albu.LongestMaxSize(max_size=200, p=1)
    sc_bgr = sc_image[:,:,0:3].astype(np.uint8)
    sc_alpha = sc_image[:,:,3].astype(np.float32)
    transformed = albu_transform_resize(image=sc_bgr, mask=sc_alpha)
    sc_bgr = transformed["image"].astype(np.uint8)
    sc_alpha = transformed["mask"].astype(np.float32)
    sc_height, sc_width = sc_bgr.shape[0:2]

    cnt = 0
    while True:
        try:
            sc_nw = min(random.randint(2*sc_width, 3*sc_width), bg_width)
            sc_nh = min(random.randint(2*sc_height, 3*sc_height), bg_height)

            sc_lx1 = random.randint(0, bg_width-sc_nw)
            sc_ly1 = random.randint(0, bg_height-sc_nh)
            
            sc_lx2 = min(max(0, sc_lx1+sc_nw), bg_width)
            sc_ly2 = min(max(0, sc_ly1+sc_nh), bg_height)
            sc_xc = int(0.5*sc_lx1+0.5*sc_lx2)
            sc_yc = int(0.5*sc_ly1+0.5*sc_ly2)

            if np.mean(bg_image[sc_ly1:sc_ly2,sc_lx1:sc_lx2,:]) < 128:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)
            
            ran_px = int(sc_xc-0.5*sc_width)
            ran_py = int(sc_yc-0.5*sc_height)

            if bb_iom([ran_px, ran_py, ran_px+sc_width, ran_py+sc_height], it.bg_box) < 0.1:
                if random.random() > 0.5:
                    cv2.line(bg_image, (sc_lx1, sc_ly1), (sc_lx2, sc_ly2), color, thickness=1)
                else:
                    cv2.line(bg_image, (sc_lx2, sc_ly1), (sc_lx1, sc_ly2), color, thickness=1)

                if random.random() > 0.5:
                    sc_lx11, sc_ly11 = random.choice([(sc_lx1, sc_ly1), (sc_lx1, sc_ly2), (sc_lx2, sc_ly1), (sc_lx2, sc_ly2)])
                    cv2.line(bg_image, (sc_xc, sc_yc), (sc_lx11, sc_ly11), color, thickness=1)

                bg_crop = bg_image.copy()[ran_py:ran_py+sc_height, ran_px:ran_px+sc_width]

                mask = sc_alpha/255.0
                mask = mask[..., None]
                composite = sc_bgr.astype(np.float32) * mask + bg_crop.astype(np.float32) * (1-mask)
                composite = composite.astype(np.uint8)
                            
                bg_image[ran_py:ran_py+sc_height, ran_px:ran_px+sc_width] = composite

                gen_boxes.append([min(sc_lx1, ran_px),min(sc_ly1, ran_py),max(sc_lx2, ran_px+sc_width),max(sc_ly2, ran_py+sc_height)])

                label_file = open(it.new_label_path, 'w')

                for k, gen_box in enumerate(gen_boxes):
                    x1, y1, x2, y2 = gen_box
                    # if k == 0:
                    #     cv2.rectangle(bg_image, (x1, y1), (x2, y2), (255,0,0), 2)
                    # else:
                    #     cv2.rectangle(bg_image, (x1, y1), (x2, y2), (0,255,0), 1)

                    xc = 0.5*(x1+x2)/bg_width
                    yc = 0.5*(y1+y2)/bg_height
                    w = (x2-x1)/bg_width
                    h = (y2-y1)/bg_height
                    label_file.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
                label_file.close()
                
                cv2.imwrite(it.new_image_path, bg_image)
                return it.new_image_path
            else:
                cnt += 1
        except:
            cnt += 1
        if cnt > 50:
            break
    return None