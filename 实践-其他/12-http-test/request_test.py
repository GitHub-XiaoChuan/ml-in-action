#!/user/bin/env python
# coding=utf-8
import requests
import datetime
import time
import threading


class url_request():
    times = []
    error = []

    def req(self):
        my_req = url_request()
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}
        # 正向图片
        # r = requests.get("http://127.0.0.1:8000/brand/IMG_20180814_101836.jpg", headers=headers)
        # 倾斜图片
        r = requests.get("http://127.0.0.1:8000/brand/9.jpg", headers=headers)
        # 横向图片
        #r = requests.get("http://127.0.0.1:8000/brand/20180917_01.jpeg", headers=headers)
        response_time = float(r.elapsed.total_seconds())  # 获取响应时间，单位ms
        my_req.times.append(response_time)  # 将响应时间写入数组
        if r.status_code != 200:
            my_req.error.append("0")


if __name__ == '__main__':
    myreq = url_request()
    threads = []
    starttime = datetime.datetime.now()

    print("request start time %s" % starttime)
    nub = 200  # 设置并发线程数
    ThinkTime = 0.2  # 设置思考时间
    for i in range(1, nub + 1):
        t = threading.Thread(target=myreq.req)
        threads.append(t)

    for t in threads:
        time.sleep(ThinkTime)
        # print "thread %s" %t #打印线程
        t.setDaemon(True)
        t.start()
    t.join()
    endtime = datetime.datetime.now()
    print("request end time %s" % endtime)
    time.sleep(3)
    AverageTime = "{:.3f}".format(float(sum(myreq.times)) / float(len(myreq.times)))  # 计算数组的平均值，保留3位小数
    print("Average Response Time %s s" % AverageTime)  # 打印平均响应时间
    usetime = str(endtime - starttime)
    hour = usetime.split(':').pop(0)
    minute = usetime.split(':').pop(1)
    second = usetime.split(':').pop(2)
    totaltime = float(hour) * 60 * 60 + float(minute) * 60 + float(second)  # 计算总的思考时间+请求时间
    print("Concurrent processing %s" % nub)  # 打印并发数
    print("use total time %s s" % totaltime)  # 打印总共消耗的时间
    print("fail request %s" % myreq.error.count("0"))  # 打印错误请求数
