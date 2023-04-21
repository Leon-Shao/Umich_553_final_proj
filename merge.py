import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

fg_path = 'Foreground_reshape/'
bg_path = 'Background_reshape/'
gt_path = 'Alpha_reshape/'
tri_path = 'Trimap_reshape/'
training_path = 'Training_Dataset/'
eval_path = "Evaluation_Dataset/"
img_dir = "Image/"
alpha_dir = "Alpha/"
trimap_dir = "Trimap/"


def read_picture_name(path):
    picture_list = os.listdir(path)
    return picture_list

def composite4(fg, bg, gt, tri, a, w, h):
    fg = cv2.resize(fg, (w, h))
    gt = cv2.resize(gt, (w, h))
    tri = cv2.resize(tri, (w, h))
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    alpha = fg != 0
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, a, fg, bg, gt, tri

def pic_padding(pic, w, h, x, y):
    pic_h, pic_w, c = pic.shape
    im = np.zeros((h, w, c))
    im[x:x+pic_h, y:y+pic_w, :] = pic
    return im

if not os.path.exists(training_path):
    os.makedirs(training_path)

if not os.path.exists(eval_path):
    os.makedirs(eval_path)

fg_list = read_picture_name(fg_path)
bg_list = read_picture_name(bg_path)

fg_training_list, fg_eval_list = train_test_split(fg_list, test_size=0.1, random_state=25)
bg_training_list, bg_eval_list = train_test_split(bg_list, test_size=0.1, random_state=25)


count = 0
for fg_name in fg_training_list:
    for bg_name in bg_training_list:
        fg_pic = os.path.join(fg_path, fg_name)
        bg_pic = os.path.join(bg_path, bg_name)
        alpha_pic = os.path.join(gt_path, fg_name)
        tri_pic = os.path.join(tri_path, fg_name)
        
        fg = cv2.imread(fg_pic)
        bg = cv2.imread(bg_pic)
        gt = cv2.imread(alpha_pic)
        tri = cv2.imread(tri_pic)
        a = 128
        if type(bg) != type(None):
            h, w, c = bg.shape
            im, a, fg, bg, gt, tri = composite4(fg, bg, gt, tri, a, w, h)
            pic_name = str(count) + '.png'
            alpha_name = str(count) + '.png'
            tri_name = str(count) + '.png'
            data_path = os.path.join(training_path, img_dir , pic_name)
            alpha_path = os.path.join(training_path, alpha_dir, alpha_name)
            trimap_path = os.path.join(training_path,trimap_dir, tri_name)
            print(data_path)
            print(alpha_path)
            print(trimap_path)
            cv2.imwrite(data_path, im)
            cv2.imwrite(alpha_path, gt)
            cv2.imwrite(trimap_path, tri)
            count += 1
            if count%500==0:
                print(f"training picture {count+1} has been generated")
count = 0
for fg_name in fg_eval_list:
    for bg_name in bg_eval_list:
        fg_pic = os.path.join(fg_path, fg_name)
        bg_pic = os.path.join(bg_path, bg_name)
        alpha_pic = os.path.join(gt_path, fg_name)
        tri_pic = os.path.join(tri_path, fg_name)
        
        fg = cv2.imread(fg_pic)
        bg = cv2.imread(bg_pic)
        gt = cv2.imread(alpha_pic)
        tri = cv2.imread(tri_pic)
        a = 128
        if type(bg) != type(None):
            h, w, c = bg.shape
            im, a, fg, bg, gt, tri = composite4(fg, bg, gt, tri, a, w, h)
            pic_name = str(count) + '.png'
            alpha_name = str(count) + '.png'
            tri_name = str(count) + '.png'
            data_path = os.path.join(eval_path, img_dir, pic_name)
            alpha_path = os.path.join(eval_path,alpha_dir,alpha_name)
            trimap_path = os.path.join(eval_path,trimap_dir,tri_name)
            cv2.imwrite(data_path, im)
            cv2.imwrite(alpha_path, gt)
            cv2.imwrite(trimap_path, tri)
            count += 1
            if count%500==0:
                print(f"training picture {count+1} has been generated")