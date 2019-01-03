#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
from urllib import request
from urllib import parse
import json
import hashlib
import base64
import cv2
import re
import os


url = 'http://webapi.xfyun.cn/v1/service/v1/ocr/general'
x_appid = '5b4d9bbf'
api_key = 'a08332353b4df650842359129ffadb88'
param = {"language": "cn|en", "location": "true"}
x_param = str(base64.b64encode(bytes(json.dumps(param).replace(' ', ''),encoding='utf-8')),encoding = "utf8")
splitors = ['：', ':']
pricechars = ['元','￥']
dict = {}

def reg_format(str):
    return re.sub(r'[^\u4E00-\u9FA5a-zA-Z0-9]', "", str)

with open('kv.dict', encoding='utf-8') as f:
    for line in f:
        words = line.strip().split(' ')
        dict[words[0]] = words

def format(str):
    """
    格式化json，并提取其中有效的内容
    :param str:
    :return:
    """
    obj = json.loads(str)
    content = set()

    for x in obj['data']['block'][0]['line']:
        for y in x['word']:
            # 过滤长度小于2的内容
            if(len(y['content'])>2):
                content.add(y['content'])

    return content

def iskvStyle(s):
    """
    分割键值对
    :param s:
    :return:
    """
    for splitor in splitors:
        if s.find(splitor) != -1:
            return s.split(splitor)
    return [s]

def getDict(a):
    """
    查找字典里面的key
    :param a:
    :return:
    """
    for k, v in dict.items():
        for word in v:
            if len(a) <= 2:
                if a == word:
                    return k
            else:
                #todo  基于编辑举例
                if a == word:
                    return k

    return ''

def parseline(arr):
    """
    提取关键信息
    :param arr:
    :return:
    """
    for i,v in enumerate(arr):
        for char in pricechars:
            if char in v:
                return ('价格', v)

        if i != len(arr)-1:
            key = getDict(reg_format(v))
            if key != '':
                return (key,arr[i+1])

    None

def parseOCR(json):
    result = []

    for line in format(json):
        arr = iskvStyle(line)
        if len(arr) > 0:
            parsed = parseline(arr)
            if parsed != None:
                result.append(parsed)

    return result

def main():
    # 图片缩放
    #img = cv2.imread("151531891121_.pic_hd.jpg", -1)
    #img = cv2.imread("161531900793_.pic.jpg", -1)
    #img = cv2.imread("171531900794_.pic.jpg", -1)
    #img = cv2.imread("181531900795_.pic.jpg", -1)
    #img = cv2.imread("191531901915_.pic.jpg", -1)
    img = cv2.imread("001.jpg", -1)
    #img = cv2.imread("211531901916_.pic.jpg", -1)
    #img = cv2.imread("221531901917_.pic.jpg", -1)

    height, width = img.shape[:2]
    #size = (int(width * 0.5), int(height * 0.5))
    # 图像的等比例缩放
    shrink = cv2.resize(img, (int(500*width/width), int(500*height/width)), interpolation=cv2.INTER_AREA)
    cv2.imwrite('shrink.png', shrink)

    # 图片加载
    f = open("shrink.png", 'rb')
    file_content = f.read()
    body = parse.urlencode({'image': base64.b64encode(file_content)})

    x_time = int(int(round(time.time() * 1000)) / 1000)
    x_checksum = hashlib.md5((api_key + str(x_time) + x_param).encode("utf8")).hexdigest()

    x_header = {
        'X-Appid': x_appid,
        'X-CurTime': x_time,
        'X-Param': x_param,
        'X-CheckSum': x_checksum
    }

    req = request.Request(url, data=body.encode("utf-8"), headers=x_header)
    result = request.urlopen(req)
    result = result.read()
    print(str(result,encoding="utf-8"))
    print(parseOCR(str(result,encoding="utf-8")))

    return


if __name__ == '__main__':
    main()