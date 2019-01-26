import re
import requests
from urllib.parse import quote
from selenium import webdriver

search = "耐克 928890"

driver=webdriver.Chrome()   #声明浏览器对象
driver.get('https://list.tmall.com/search_product.htm?q='+quote(search)) #通过浏览器打开百度页面
print(driver.page_source) #得到网页html信息
driver.close()  #关闭浏览器


# response = requests.get('https://list.tmall.com/search_product.htm?q='+quote(search))
# print(response.status_code)# 响应的状态码
# print(response.content)  #返回字节信息
# print(response.text)  #返回文本内容

with open('a.html', 'w') as f:
    f.write(str(driver.page_source))

# urls=re.findall(r'class="items".*?href="(.*?)"',respose.text,re.S)  #re.S 把文本信息转换成1行匹配
# url=urls[5]
# result=requests.get(url)
# mp4_url=re.findall(r'id="media".*?src="(.*?)"',result.text,re.S)[0]

# video=requests.get(mp4_url)

# with open('D:\\a.mp4','wb') as f:
#     f.write(video.content)