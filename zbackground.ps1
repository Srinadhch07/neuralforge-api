# celery -A app.config.celery_app.celery worker --pool=solo --loglevel=info --prefetch-multiplier=1
celery -A app.config.celery_app.celery worker --pool=solo --loglevel=info
Write-Host "Started worker ..." -ForegroundColor Green
