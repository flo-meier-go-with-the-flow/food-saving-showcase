import sched, time

def print_time(a='default'):
    print("From print_time", time.time(), a)

def print_some_times():
    print(time.time())
    s = sched.scheduler(time.time, time.sleep)
    s.enter(1, 1, print_time)
    s.enter(2, 1, print_time)
    s.enter(3, 1, print_time)
    s.run()
    print(time.time())

if __name__ == '__main__':

    print_some_times()