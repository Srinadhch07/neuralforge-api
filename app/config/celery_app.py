from celery import Celery
import ssl
# REDIS_URL = "redis://localhost:6379/0"
REDIS_URL="rediss://default:AfNRAAIncDJjMDQwNWY0ODExOWQ0NTQ2YTkyNjAxNGJhOWE5ODVhN3AyNjIyODk@new-adder-62289.upstash.io:6379/0"


celery = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery.conf.imports = (
    "app.tasks.dl_tasks",
)
celery.conf.update(
    broker_use_ssl={
        "ssl_cert_reqs": ssl.CERT_NONE
    },
    redis_backend_use_ssl={
        "ssl_cert_reqs": ssl.CERT_NONE
    },
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    broker_pool_limit=1,
    task_track_started=True,
    task_send_sent_event=True,
    result_expires = 3600
)
