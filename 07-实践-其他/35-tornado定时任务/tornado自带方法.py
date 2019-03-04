import tornado.web
from tornado import web, ioloop
import datetime

"""
1 任务为阻塞，如果任务执行时间长，会导致其他的任务卡死
2 任务如果有延迟，那么本来应该执行的任务将会取消掉
"""

period = 5 * 1000  # every 5 s

class MainHandler(web.RequestHandler):
    def get(self):
        self.write('Hello xiaorui.cc')



def task2():
    print(datetime.datetime.now())
    import time
    time.sleep(4)
    print('call in task 2')


def task1():
    print(datetime.datetime.now())
    print('call in task 1')


if __name__ == '__main__':
    application = web.Application([
        (r'/', MainHandler),
    ])
    application.listen(8081)
    ioloop.PeriodicCallback(task1, 1000).start()  # start scheduler
    ioloop.PeriodicCallback(task2, 3000).start()  # 这里的时间是毫秒
    ioloop.IOLoop.instance().start()