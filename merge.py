import os
import cv2
import numpy as np

fg_path = 'Foreground/'
bg_path = 'Background/'
gt_path = 'Trimap/'
output_path = 'Dataset/'

def read_picture_name(path):
    picture_list = os.listdir(path)
    return picture_list

def composite4(fg, bg, gt, a, w, h):
    fg = cv2.resize(fg, (w, h))
    gt = cv2.resize(gt, (w, h))
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
    return im, a, fg, bg, gt

def pic_padding(pic, w, h, x, y):
    pic_h, pic_w, c = pic.shape
    im = np.zeros((h, w, c))
    im[x:x+pic_h, y:y+pic_w, :] = pic
    return im

if not os.path.exists(output_path):
    os.makedirs(output_path)

fg_list = read_picture_name(fg_path)
bg_list = read_picture_name(bg_path)

count = 0
for fg_name in fg_list:
    for bg_name in bg_list:
        fg_pic = os.path.join(fg_path, fg_name)
        bg_pic = os.path.join(bg_path, bg_name)
        tr_pic = os.path.join(gt_path, fg_name)
        fg = cv2.imread(fg_pic)
        bg = cv2.imread(bg_pic)
        gt = cv2.imread(tr_pic)
        a = 128
        if type(bg) != type(None):
            h, w, c = bg.shape
            im, a, fg, bg, gt = composite4(fg, bg, gt, a, w, h)
            pic_name = str(count) + '.png'
            tri_name = str(count) + '_tri.png'
            data_path = os.path.join(output_path, pic_name)
            tri_path = os.path.join(output_path, tri_name)
            cv2.imwrite(data_path, im)
            cv2.imwrite(tri_path, gt)
            count += 1
