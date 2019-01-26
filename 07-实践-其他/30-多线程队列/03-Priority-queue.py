import queue
import threading
from functools import cmp_to_key
import operator


class Job(object):
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description
        print('Job ' + description)
        return

    def __cmp__(self, other):
        print('比较')
        if self is None:
            return -1
        if other is None:
            return 1

        if self.priority > other.priority:
            return 1
        elif self.priority < other.priority:
            return -1
        else:
            return 0

q = queue.PriorityQueue()

# q.put(Job(3, 'level 3'))
# q.put(Job(10, 'level 10'))
# q.put(Job(1, 'level 1'))
# q.put(Job(2, 'level 2'))
#
# while not q.empty():
#     print(q.get())