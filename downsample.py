import cv2 as cv
import os

path_list = []

time = 4

def read_picture_name(path):
    picture_list = os.listdir(path)
    return picture_list

def downsample(pic, time):
    h, w, c = pic.shape
    new_pic = cv.resize(pic, (w//time, h//time))
    h, w, c = new_pic.shape
    return new_pic

for path in path_list:
    new_path = path+'_reshape'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    pic_list = read_picture_name(path)
    for pic_name in pic_list:
        pic_path = os.path.join(path, pic_name)
        pic = cv.imread(pic_path)
        if type(pic) != type(None):
            new_pic = downsample(pic, time)
            new_pic_path = os.path.join(new_path, pic_name)
            cv.imwrite(new_pic_path, new_pic)