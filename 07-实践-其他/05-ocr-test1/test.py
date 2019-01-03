#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re

res = re.findall('^(?=.*\d)(?=.*[a-zA-Z])(?=.*[\u4E00-\u9FA5])[\u4E00-\u9FA5A-Za-z0-9]*$','小高jimmy9999')

print(re.sub(r'[^\u4E00-\u9FA5a-zA-Z0-9]', "", "-产品1名称a"))
print(re.sub(r'[^\u4E00-\u9FA5]|[^a-zA-Z0-9]', "", "e~ fda "))