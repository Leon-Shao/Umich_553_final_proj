import os
import cv2
import numpy as np

fg_path = 'Foreground/'
bg_path = 'Background/'


def composite4(fg, bg, a, w, h):
    fg = cv2.resize(fg, (w, h))
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
    return im, a, fg, bg

def pic_padding(pic, w, h, x, y):
    pic_h, pic_w, c = pic.shape
    im = np.zeros((h, w, c))
    im[x:x+pic_h, y:y+pic_w, :] = pic
    return im

fg_pic = os.path.join(fg_path, '2007_000032.png')
bg_pic = os.path.join(bg_path, '5_new_bg.png')

fg = cv2.imread(fg_pic)
bg = cv2.imread(bg_pic)

a = 128

h, w, c = bg.shape

im, a, fg, bg = composite4(fg, bg, a, w, h)

cv2.imwrite('test.png', im)
