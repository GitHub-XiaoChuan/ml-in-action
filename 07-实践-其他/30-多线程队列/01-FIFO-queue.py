import queue

# FIFO 先进先出队列
q = queue.Queue()

for i in range(5):
    q.put(i)

while not q.empty():
    print(q.get())

