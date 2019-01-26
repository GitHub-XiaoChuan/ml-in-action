from tornado.httpclient import HTTPClient

def synchronous_fetch(url):
    http_client = HTTPClient()
    response = http_client.fetch(url)
    return response.body

print(synchronous_fetch("http://www.baidu.com?wd=england"))