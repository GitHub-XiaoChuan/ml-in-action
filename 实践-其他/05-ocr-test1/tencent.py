#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
from urllib import request
from urllib import parse
import json
import hashlib
import base64

def main():
    f = open("/Users/xingoo/PycharmProjects/ml-py/work-testing/05-ocr-test1/名片2.jpeg", 'rb')
    # f = open("/Users/xingoo/PycharmProjects/ml-py/work-testing/05-ocr-test1/名片.jpg", 'rb')
    file_content = f.read()
    base64_image = base64.b64encode(file_content)
    #body = parse.urlencode({'image': base64_image})
    body = {
        "appid":"1254146587",
        "bucket":"test",
        "url":"http://images.cnblogs.com/cnblogs_com/xing901022/1187174/o_Jietu20180327-182924.jpg"
    }

    url = 'http://recognition.image.myqcloud.com/ocr/general'

    x_header = {
        'host': 'recognition.image.myqcloud.com',
        'content-type': 'application/json',
        'authorization': ''
    }

    print(x_header)
    print(body)
    print({'image': base64_image})
    req = request.Request(url, data=body.encode("utf-8"), headers=x_header)
    result = request.urlopen(req)
    result = result.read()
    print(str(result,encoding="utf-8"))
    return


if __name__ == '__main__':
    main()