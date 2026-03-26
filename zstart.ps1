Write-host "Starting project... in  5 secs" -ForegroundColor DarkBlue
Write-host "Started project on Production Mode" -ForegroundColor Green
# uvicorn main:app  --port 8000 --reload
uvicorn main:app  --port 8000
# uvicorn main:app --host 0.0.0.0 --port 8080 --log-level critical
