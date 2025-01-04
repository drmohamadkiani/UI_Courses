import threading
import time

# Shared Memory variables
CAPACITY = 10
buffer = [-1 for _ in range(CAPACITY)]
in_index = 0
out_index = 0

# Declaring Semaphores
mutex = threading.Semaphore(1)
empty = threading.Semaphore(CAPACITY)
full = threading.Semaphore(0)

# Additional semaphore to prioritize producers
producer_priority = threading.Semaphore(1)

# Producer Thread Class
class Producer(threading.Thread):
    def run(self):
        global CAPACITY, buffer, in_index, out_index
        global mutex, empty, full, producer_priority

        items_produced = 0
        counter = 0

        while items_produced < 20:
            producer_priority.acquire()  # Ensure producer gets priority
            empty.acquire()
            mutex.acquire()

            counter += 1
            buffer[in_index] = counter
            in_index = (in_index + 1) % CAPACITY
            print("Producer produced: ", counter)

            mutex.release()
            full.release()
            producer_priority.release()  # Allow other producers to proceed

            time.sleep(0.5)
            items_produced += 1

# Consumer Thread Class
class Consumer(threading.Thread):
    def run(self):
        global CAPACITY, buffer, in_index, out_index
        global mutex, empty, full, producer_priority

        items_consumed = 0

        while items_consumed < 20:
            full.acquire()
            mutex.acquire()

            item = buffer[out_index]
            out_index = (out_index + 1) % CAPACITY
            print("Consumer consumed item: ", item)

            mutex.release()
            empty.release()

            time.sleep(1)
            items_consumed += 1

# Creating Threads
producer = Producer()
consumer = Consumer()

# Starting Threads
consumer.start()
producer.start()

# Waiting for threads to complete
producer.join()
consumer.join()