from multiprocessing import Process, Queue

def f(q):
    q.put([42, None, 'hello'])

def f1(q):
    q.put(['ni hao', 32])

if __name__ == '__main__':
    q = Queue()
    p1 = Process(target=f, args=(q,))
    p1.start()

    p2 = Process(target = f1, args = (q,))
    p2.start()
    #print q.get()    # prints "[42, None, 'hello']"
    p1.join()
    p2.join()
    print(q.get())
    print(q.get())
    print(q.get())
