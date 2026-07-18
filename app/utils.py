import time

from rag_assignment.app.config import TOP_K


def timer(func):

    def wrapper(*args, **kwargs):

        start = time.time()

        result = func(*args, **kwargs)

        end = time.time()

        print(
            f"Latency : {end-start:.3f} sec"
        )

        return result

    return wrapper

from app.utils import timer

@timer
def retrieve(query, k=TOP_K):
    ...
