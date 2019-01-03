#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
from urllib import request,parse
import json
import hashlib
import base64
import cv2


url = 'http://webapi.xfyun.cn/v1/service/v1/ocr/general'
x_appid = '5b4d9bbf'
api_key = 'a08332353b4df650842359129ffadb88'
param = {"language": "cn|en", "location": "true"}
x_param = str(base64.b64encode(bytes(json.dumps(param).replace(' ', ''),encoding='utf-8')),encoding = "utf8")

dict = {}

def main():
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
    result = request.urlopen(req).read()

    body = json.loads(result, encoding='utf-8')

    img = cv2.imread('shrink.png')
    for text_line in body['data']['block'][0]['line']:
        word = text_line['word'][0]
        x1 = word['location']['top_left']['x']
        y1 = word['location']['top_left']['y']
        x2 = word['location']['right_bottom']['x']
        y2 = word['location']['right_bottom']['y']

        # 绘制文本框
        cv2.line(img, (x1, y1), (x2, y1), (255, 0, 0), 2)
        cv2.line(img, (x1, y1), (x1, y2), (255, 0, 0), 2)
        cv2.line(img, (x1, y2), (x2, y2), (255, 0, 0), 2)
        cv2.line(img, (x2, y1), (x2, y2), (255, 0, 0), 2)

        # 输出对应文本
        text = word['content']
        print(text)

    cv2.imwrite('result.png',img)

if __name__ == '__main__':
    main()