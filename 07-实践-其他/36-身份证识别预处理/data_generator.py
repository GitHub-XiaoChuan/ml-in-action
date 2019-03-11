import os
import glob
import random
import codecs
import pathlib
import math
import PIL
from PIL import Image, ImageDraw
import numpy as np
import cv2
import random
import tqdm
from PIL import Image, ImageFont, ImageDraw
from basic.aug import Aug_Operations as aug

target_path = '/Users/xingoo/Documents/dataset/单字符数据集2/'
dict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']

def getBackgroundListFromDir(background_dir):
    images = glob.glob('/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/36-身份证识别预处理/bak/*.jpg')
    print('Load background image: %d' % len(images))
    return images


def getFontListFromDir(font_dir):
    fonts = glob.glob('/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/36-身份证识别预处理/fonts/*.[tT][tT][fCF]')
    print('Load font files: %d' % len(fonts))
    return fonts


def getRandomOneFromList(list):
    return list[random.randint(0, len(list) - 1)]


def rotate_img(image, degree):
    img = np.array(image)

    height, width = img.shape[:2]

    heightNew = int(
        width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(
        height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255, 0))

    w = width
    h = height

    points = np.matrix([[-w / 2, -h / 2, 1], [-w / 2, h / 2, 1], [w / 2, h / 2, 1], [w / 2, -h / 2, 1]])

    matRotation = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)

    matRotation[0, 2] = widthNew / 2
    matRotation[1, 2] = heightNew / 2

    p = matRotation * points.T

    for row in imgRotation:
        for element in row:
            if element[3] == 0:
                for i in range(3):
                    element[i] = 0
    image = Image.fromarray(imgRotation)
    points = np.array(p.T, int)
    return image, points

def augmentImage(txt_img, points):
    # Augment rate for eatch type
    rot = random.uniform(0, 1)
    skew_rate = random.uniform(0, 1)
    shear_rate = random.uniform(0, 1)
    distort_rate = random.uniform(0, 1)

    if rot < 1:
        rot_degree = random.randint(-5, 5)
        txt_img, points = rotate_img(txt_img, rot_degree)
    elif skew_rate < 0:  # 平行四边形形变
        skew = aug.Skew(1, 'RANDOM', 0.3)
        txt_img, points = skew.perform_operation(txt_img)
    elif shear_rate < 0:  # 剪切形变
        shear = aug.Shear(1., 5, 5)
        txt_img, points = shear.perform_operation(txt_img)
    elif distort_rate < 0:  # 扭曲变形
        distort = aug.Distort(1.0, 4, 4, 1)
        txt_img = distort.perform_operation(txt_img)
    return txt_img, points

def mergeBgimgAndTxtimgPoints(center, points):
    center_x, center_y = center
    width, height = points[2]

    return np.array([[center_x - width // 2, center_y - height // 2],
                    [center_x + width // 2, center_y - height // 2],
                    [center_x + width // 2, center_y + height // 2],
                    [center_x - width // 2, center_y + height // 2]])

def pltImage2Array(image):
    image = image.convert('RGB')
    image = np.array(image)
    # Convert RGB to BGR
    image = image[:, :, ::-1].copy()
    return image

def setColor():
    c = random.randint(0, 150)
    return np.array([c, c, c], dtype='float64')

def mergeImageAtPoint(image, txt_img, left_top_point, color):
    left, top = left_top_point
    image = pltImage2Array(image)

    w, h = txt_img.size
    res_img = np.array(image)
    txt_img = np.array(txt_img)

    # 获取最后一位，作为mask
    mask = txt_img[:, :, -1]
    mask1 = mask * 1.0 / 255

    for i in range(0, h - 1):
        for j in range(0, w - 1):
            if i + top < res_img.shape[0] and j + left < res_img.shape[1]:
                res_img[i + top, j + left, :] = color * mask1[i, j] + (1 - mask1[i, j]) * res_img[i + top, j + left, :]

    res_img = res_img[:, :, [2, 1, 0]]
    res_img = Image.fromarray(res_img)

    return res_img

def putContent2Image(background_image_path, font_path, height=32):
    # 加载背景图片
    image = cv2.imread(background_image_path)

    scale = 0.3 + 0.7 * random.random()
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # 随机截取图片
    h, w, _ = image.shape
    random_h = random.randint(0, h - height)
    back_image = image[random_h:random_h + height, :, :]

    # 设置字体大小
    font_size = 28

    # 设置中心点
    left_center_point = (height // 2, height // 2)

    #设置字体颜色
    c = random.randint(0, 150)
    color = np.array([c, c, c], dtype='float64')


    dict_len = len(dict)

    # 加载背景
    background = Image.fromarray(back_image, 'RGB')
    charlist = []
    widthes = []
    while left_center_point[0] < w and w - left_center_point[0] > height:

        # 随机字符
        character = dict[random.randint(0, dict_len-1)]
        charlist.append(character)

        background = background.convert('RGBA')

        font = ImageFont.truetype(font_path, font_size)
        font_w, font_h = font.getsize(character)
        widthes.append(font_w)

        txt = Image.new('RGBA', (font_w, font_h), (255, 255, 255, 0))

        # 得到四个点的坐标
        points_in_txt = np.array([[0, 0], [font_w, 0], [font_w, font_h], [0, font_h]])

        draw = ImageDraw.Draw(txt)
        draw.text((0, 0), character, font=font, fill=(255, 255, 255, 255))  # draw text, full opacity

        txt, points_in_txt = augmentImage(txt, points_in_txt)

        points = mergeBgimgAndTxtimgPoints(left_center_point, points_in_txt)

        out_image = mergeImageAtPoint(background, txt, tuple(points[0]), color)
        background = out_image.convert('RGB')

        left_center_point = (left_center_point[0] + height, left_center_point[1])

    background = cv2.cvtColor(np.asarray(background), cv2.COLOR_RGB2BGR)

    background = np.array(background*(0.3+0.7*random.random())).astype(np.uint8)
    return background, charlist, widthes


bg_img_list = getBackgroundListFromDir('./bak')
font_list = getFontListFromDir('./fonts')

# 随机背景和字体
number = 0
for i in range(0, 10000):
    background_image_path, font_path = map(getRandomOneFromList, [bg_img_list, font_list])
    #print('background: %s - font: %s' % (background_image_path, font_path))

    background, charlist, widthes = putContent2Image(background_image_path, font_path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    print('%d - %s'%(i, ''.join(charlist)))
    bh, bw = background.shape

    for index, char in enumerate(charlist):
        number += 1
        name = '%s%s_%d.jpg' % (target_path, char, number)
        center_x = 32*index+16
        char_width = widthes[index]

        char_begin = max(0, center_x - char_width//2-5)
        char_end = min((center_x + char_width//2+5), bw)

        cv2.imwrite(name, background[:, char_begin:char_end])
        # cv2.imshow(name, background[:, char_begin:char_end, :])
        # cv2.waitKey(0)

    # cv2.imshow('back', background)
    # cv2.waitKey(0)


