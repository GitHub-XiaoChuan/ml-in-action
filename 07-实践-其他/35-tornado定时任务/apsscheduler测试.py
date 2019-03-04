from apscheduler.schedulers.blocking import BlockingScheduler
import time

# 实例化一个调度器
scheduler = BlockingScheduler()


def job1():
    print("job1 %s: 执行任务开始" % time.asctime())
    time.sleep(4)
    print("job1 %s: 执行任务结束" % time.asctime())

def job2():
    #time.sleep(4)
    print("job2 %s: 执行任务" % time.asctime())

# 添加任务并设置触发方式为3s一次
scheduler.add_job(job1, 'interval', seconds=3)
scheduler.add_job(job2, 'interval', seconds=1)

# 开始运行调度器
scheduler.start()

print("1")