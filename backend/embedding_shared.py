# backend/embedding_shared.py

from queue import Queue

# Shared across worker + parser
job_q = Queue(maxsize=32)
result_q = Queue()
STOP_TOKEN = None