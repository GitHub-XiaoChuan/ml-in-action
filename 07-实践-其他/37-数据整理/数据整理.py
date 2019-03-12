import glob
import cv2
import json
import os
import xlrd
import numpy as np

BASE_PATH = '/Users/xingoo/Documents/dataset/天狗-吊牌/'

EXCEL_PATH = '正式-审核文件/'
LABEL_PATH = '正式-标注文件/'
IMAGE_PATH = '正式-全部图片/'

RESULT_PATH = 'data/'

excel_files = glob.glob(BASE_PATH+EXCEL_PATH+'*.xls')

for excel_file_path in excel_files:
    # 获得文件名
    dir_name = excel_file_path.split('/')[-1].split('.')[0]

    sheet = xlrd.open_workbook(excel_file_path).sheet_by_index(0)
    nrows = sheet.nrows

    for j in range(1, nrows):
        file_name = sheet.cell(j, 0).value
        status = sheet.cell(j, 1).value

        if status == '正常':

            prefix = file_name.split('_')[0]

            json_list = glob.glob(BASE_PATH + LABEL_PATH + dir_name + '/' + prefix + '*.json')
            current_image_path = BASE_PATH + IMAGE_PATH + dir_name + '/' + file_name

            if len(json_list) > 0 and os.path.exists(current_image_path):

                lines = []
                with open(json_list[0], 'r', encoding='utf-8') as f:
                    jo = json.load(f)
                    if 'outputs' not in jo or 'object' not in jo['outputs']:
                        continue
                    bs = jo['outputs']['object']

                    for b in bs:
                        xmin = str(b['bndbox']['xmin'])
                        ymin = str(b['bndbox']['ymin'])
                        xmax = str(b['bndbox']['xmax'])
                        ymax = str(b['bndbox']['ymax'])

                        lines.append(' '.join([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, b['name']]) + '\n')

                with open(BASE_PATH + RESULT_PATH + prefix + '.txt', 'w+', encoding='utf-8') as f:
                    f.writelines(lines)
                    img = cv2.imread(current_image_path)
                    cv2.imwrite(BASE_PATH + RESULT_PATH + prefix + '.jpg', img)
            print('完成%s的解析' % current_image_path)