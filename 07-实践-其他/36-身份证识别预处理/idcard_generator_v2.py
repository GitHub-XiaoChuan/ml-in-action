import tqdm
import random
import cv2
import numpy as np
import basic.aug.utils as utils
from PIL import Image, ImageFont, ImageDraw

TOTAL = 10000
SAVE_PATH = '/Users/xingoo/Documents/dataset/idcard_xulie/'

AREA_CODE = {
    11: "北京", 12: "天津", 13: "河北", 14: "山西", 15: "内蒙古", 21: "辽宁", 22: "吉林",
    23: "黑龙江", 31: "上海", 32: "江苏", 33: "浙江", 34: "安徽", 35: "福建", 36: "江西",
    37: "山东", 41: "河南", 42: "湖北", 43: "湖南", 44: "广东", 45: "广西", 46: "海南",
    50: "重庆", 51: "四川", 52: "贵州", 53: "云南", 54: "西藏", 61: "陕西", 62: "甘肃",
    63: "青海", 64: "宁夏", 65: "新疆", 71: "台湾", 81: "香港", 82: "澳门", 91: "国外"
}

VALIDATE = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
LAST_KEY = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']


def putContent2Image(content, background_image_path, font_path, add_rectangle=0, resize_rate=2):
    background = cv2.imread(str(background_image_path))
    h, w, _ = background.shape

    top = random.randint(0, h - 32) + 16
    left = random.randint(0, 30) + 16

    font_size = random.randint(20, 32)

    #
    # image = Image.open(background_image_path)
    #
    # image = image.resize((280, 32), Image.ANTIALIAS)

    # 确定字体的大小

    # font_size = min(280//len(content), 28)

    # 获得文字起始点
    left_center_point = (left, top)

    # 计算文字颜色，我这里只需要灰白的，直接RGB相等然后随机就行了
    c = random.randint(0, 150)
    color = np.array([c, c, c], dtype='float64')

    for character in content:
        background, points = putOneCharacter2Image(character, background, font_path, font_size, left_center_point, color)
        left_center_point = (max(points[1][0], points[2][0]), left_center_point[1])

    box = [
        max(0, left - random.randint(16, 30)),
        max(0, top - random.randint(16, 32)),
        min(w, left_center_point[0] + random.randint(16, 30)),
        min(h, left_center_point[1] + random.randint(16, 32))
    ]

    new_image = background[box[1]:box[3], box[0]:box[2], :]
    new_image = np.array(new_image * (0.3 + 0.7 * random.random())).astype(np.uint8)

    return new_image


def putOneCharacter2Image(character, background_image, font_path, font_size, left_center_point, color=None):
    background_image = Image.fromarray(background_image, 'RGB')
    background = background_image.convert('RGBA')
    font = ImageFont.truetype(font_path, font_size)
    width, height = font.getsize(character)
    width += 3
    height += 3

    txt = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    # 得到四个点的坐标
    points_in_txt = utils.getPointsOfImageRectangle(width, height)
    draw = ImageDraw.Draw(txt)
    draw.text((0, 0), character, font=font, fill=(255, 255, 255, 255))  # draw text, full opacity

    txt, points_in_txt = utils.augmentImage(txt, points_in_txt)
    points = utils.mergeBgimgAndTxtimgPoints(left_center_point, points_in_txt)
    # assert points[0][0] >= 0 and points[0][1] >= 0
    # assert points[2][0] <= background.size[0] and points[2][1] <= background.size[1]
    # out_image = Image.alpha_composite(background, txt)
    out_image = utils.mergeImageAtPoint(background, txt, tuple(points[0]), color)
    out_image = out_image.convert('RGB')
    out_image = np.asarray(out_image)
    return out_image, points


def saveImage(image, image_index):
    image_save_dir = '/data1/ocr/ctc/images'
    utils.saveImage2Dir(image, image_save_dir, image_name=str(image_index))


def save_annotation(content, image_index):
    ann_name = ''.join([str(image_index), '.jpg'])
    ann_path = '/data1/ocr/ctc/label.txt'
    with open(ann_path, 'a+', encoding='utf-8') as file:
        file.write("%s %s\n" % (ann_name, content))


def generate_id_card():
    key = ''
    area_keys = list(AREA_CODE.keys())
    random.shuffle(area_keys)
    # 省份
    key += str(area_keys[0])
    # 城市
    key += str(random.randint(0, 9))
    key += str(random.randint(0, 9))
    # 区县
    key += str(random.randint(0, 9))
    key += str(random.randint(0, 9))
    # 年份
    key += str(random.randint(1900, 2900))
    # 月份
    month = random.randint(1, 12)
    if month < 10:
        key += '0' + str(month)
    else:
        key += str(month)
    # 日 不管大小月了
    day = random.randint(1, 31)
    if day < 10:
        key += '0' + str(day)
    else:
        key += str(day)

    # 派出所
    key += str(random.randint(0, 9))
    key += str(random.randint(0, 9))

    # 性别
    key += str(random.randint(0, 9))

    # 校验码
    sum = 0
    for index, c in enumerate(key):
        sum += int(c) * VALIDATE[index]
    last_key = sum % 11

    return key + LAST_KEY[last_key]


def main():
    bg_img_list = utils.getBackgroundListFromDir('./bak')
    font_list = utils.getFontListFromDir('./fonts')

    # 随机背景和字体
    for i in tqdm.tqdm(range(0, TOTAL)):
        idcard = generate_id_card()
        background_image_path, font_path = map(utils.getRandomOneFromList, [bg_img_list, font_list])
        image = putContent2Image(idcard, background_image_path, font_path)
        cv2.imwrite(SAVE_PATH + str(i) + '_' + idcard+'.jpg', image)


if __name__ == '__main__':
    main()
