# backend/embedding_shared.py

from queue import Queue

# Shared across worker + parser
job_q = Queue(maxsize=32)
STOP_TOKEN = None
