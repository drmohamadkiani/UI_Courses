import threading
import time

class Philosopher(threading.Thread):
    def __init__(self, index, left_fork, right_fork):
        threading.Thread.__init__(self)
        self.index = index
        self.left_fork = left_fork
        self.right_fork = right_fork

    def run(self):
        while True:
            print(f"Philosopher {self.index} is thinking.")
            time.sleep(1)
            print(f"Philosopher {self.index} is hungry.")
            self.dine()

    def dine(self):
        # Attempt to create a deadlock by changing the order of acquiring forks
        with self.left_fork:
            print(f"Philosopher {self.index} picked up left fork.")
            time.sleep(1)  # Added delay to increase the chance of deadlock
            with self.right_fork:
                print(f"Philosopher {self.index} picked up right fork.")
                print(f"Philosopher {self.index} is eating.")
                time.sleep(1)

def main():
    forks = [threading.Lock() for _ in range(5)]
    philosophers = [Philosopher(i, forks[i], forks[(i + 1) % 5]) for i in range(5)]

    for philosopher in philosophers:
        philosopher.start()

if __name__ == "__main__":
    main()