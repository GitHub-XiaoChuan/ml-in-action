# coding: utf-8
from concurrent.futures import ThreadPoolExecutor as Pool
from concurrent.futures import wait
import requests

URLS = ['http://qq.com', 'http://sina.com', 'http://www.baidu.com', ]


def task(url, timeout=10):
    return requests.get(url, timeout=timeout)


with Pool(max_workers=3) as executor:
    future_tasks = [executor.submit(task, url) for url in URLS]

    for f in future_tasks:
        if f.running():
            print('%s is running' % str(f))

    results = wait(future_tasks, return_when='FIRST_COMPLETED')
    done = results[0]
    for x in done:
        print(x)