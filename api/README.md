Start container service with:  

```
docker run -d \
  --name resume_api \
  -p 8000:80 \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  resume_analysis_api:latest
```