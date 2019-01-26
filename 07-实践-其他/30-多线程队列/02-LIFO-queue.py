import queue
# LIFO 后进先出队列
q2 = queue.LifoQueue()

for i in range(5):
    q2.put(i)

while not q2.empty():
    print(q2.get())