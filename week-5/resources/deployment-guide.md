# MCP Server Deployment Guide

Complete guide for deploying HTTP-streamable MCP servers to production.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Platform Deployment](#cloud-platform-deployment)
6. [Monitoring & Observability](#monitoring--observability)
7. [Scaling & Performance](#scaling--performance)
8. [Troubleshooting](#troubleshooting)

---

## Pre-Deployment Checklist

### Code Quality
- [ ] All tests passing (unit, integration, e2e)
- [ ] Code coverage > 80%
- [ ] Security scan completed (no critical issues)
- [ ] Dependency audit passed
- [ ] Code review approved
- [ ] Documentation complete

### Configuration
- [ ] Environment variables documented
- [ ] Secrets stored in vault
- [ ] Configuration validated
- [ ] Feature flags configured
- [ ] Resource limits defined
- [ ] Timeout values set

### Security
- [ ] Security checklist completed
- [ ] Penetration testing done
- [ ] SSL/TLS certificates ready
- [ ] Authentication configured
- [ ] Authorization rules defined
- [ ] Audit logging enabled

### Infrastructure
- [ ] DNS records configured
- [ ] Load balancer ready
- [ ] Database migrations tested
- [ ] Backup strategy defined
- [ ] Monitoring configured
- [ ] Alerting rules set

---

## Local Development

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/mcp-server.git
cd mcp-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your local configuration

# Run database migrations (if applicable)
alembic upgrade head

# Start server
python server.py
```

### Development Workflow

```bash
# Run tests
pytest

# Run with auto-reload
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Run linting
black .
ruff check .

# Run security scan
bandit -r .

# Run type checking
mypy .
```

---

## Docker Deployment

### Dockerfile

```dockerfile
# Multi-stage build for smaller image
FROM python:3.11-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 mcpuser

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /home/mcpuser/.local

# Copy application code
COPY --chown=mcpuser:mcpuser . .

# Switch to non-root user
USER mcpuser

# Set PATH
ENV PATH=/home/mcpuser/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python", "server.py"]
```

### Build and Run

```bash
# Build image
docker build -t mcp-server:latest .

# Run container
docker run -d \
  --name mcp-server \
  -p 8000:8000 \
  -e DB_HOST=postgres \
  -e REDIS_HOST=redis \
  --restart unless-stopped \
  mcp-server:latest

# View logs
docker logs -f mcp-server

# Stop container
docker stop mcp-server
```

### Docker Compose

```yaml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=postgres
      - REDIS_HOST=redis
      - LOG_LEVEL=info
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
    networks:
      - mcp-network

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mcpdb
      - POSTGRES_USER=mcpuser
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - mcp-network

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    networks:
      - mcp-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - mcp-server
    networks:
      - mcp-network

volumes:
  postgres-data:
  redis-data:

networks:
  mcp-network:
    driver: bridge
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  namespace: production
  labels:
    app: mcp-server
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
        version: v1.0.0
    spec:
      serviceAccountName: mcp-server
      containers:
      - name: mcp-server
        image: your-registry/mcp-server:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: db-host
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: db-password
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
  namespace: production
spec:
  selector:
    app: mcp-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mcp-server
  namespace: production
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - mcp.example.com
    secretName: mcp-tls
  rules:
  - host: mcp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mcp-server
            port:
              number: 80
```

### HorizontalPodAutoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace production

# Create secrets
kubectl create secret generic mcp-secrets \
  --from-literal=db-host=postgres.example.com \
  --from-literal=db-password=your-password \
  -n production

# Apply manifests
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Check status
kubectl get pods -n production
kubectl get svc -n production
kubectl get ingress -n production

# View logs
kubectl logs -f deployment/mcp-server -n production

# Scale manually
kubectl scale deployment mcp-server --replicas=5 -n production
```

---

## Cloud Platform Deployment

### AWS (ECS/Fargate)

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker build -t mcp-server .
docker tag mcp-server:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/mcp-server:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/mcp-server:latest

# Create ECS task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create ECS service
aws ecs create-service \
  --cluster production \
  --service-name mcp-server \
  --task-definition mcp-server:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Google Cloud (Cloud Run)

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/your-project/mcp-server
gcloud run deploy mcp-server \
  --image gcr.io/your-project/mcp-server \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances 3 \
  --max-instances 10 \
  --memory 1Gi \
  --cpu 1 \
  --set-env-vars DB_HOST=postgres.example.com
```

### Azure (Container Instances)

```bash
# Create resource group
az group create --name mcp-rg --location eastus

# Create container
az container create \
  --resource-group mcp-rg \
  --name mcp-server \
  --image your-registry/mcp-server:latest \
  --cpu 1 \
  --memory 1 \
  --ports 8000 \
  --dns-name-label mcp-server \
  --environment-variables DB_HOST=postgres.example.com
```

---

## Monitoring & Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
request_count = Counter('mcp_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('mcp_request_duration_seconds', 'Request duration')
active_connections = Gauge('mcp_active_connections', 'Active connections')

# Expose metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "MCP Server Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(mcp_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, mcp_request_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(mcp_requests_total{status=~\"5..\"}[5m])"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info("tool_called", tool="add", user_id="user123", duration=0.05)
```

---

## Scaling & Performance

### Horizontal Scaling

```bash
# Kubernetes
kubectl scale deployment mcp-server --replicas=10

# Docker Swarm
docker service scale mcp-server=10

# AWS ECS
aws ecs update-service --cluster production --service mcp-server --desired-count 10
```

### Vertical Scaling

```yaml
# Increase resources
resources:
  requests:
    memory: "1Gi"
    cpu: "1000m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

### Performance Optimization

```python
# Connection pooling
from asyncpg import create_pool

pool = await create_pool(
    host='localhost',
    port=5432,
    user='user',
    password='password',
    database='db',
    min_size=10,
    max_size=50
)

# Caching
from aiocache import Cache
from aiocache.serializers import JsonSerializer

cache = Cache(Cache.REDIS, endpoint="localhost", port=6379, serializer=JsonSerializer())

@cache.cached(ttl=300)
async def get_data(key):
    return await fetch_from_db(key)

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/endpoint")
@limiter.limit("100/minute")
async def endpoint(request: Request):
    return {"status": "ok"}
```

---

## Troubleshooting

### Common Issues

#### Server Won't Start

```bash
# Check logs
docker logs mcp-server
kubectl logs deployment/mcp-server

# Check configuration
env | grep MCP_

# Test connectivity
curl http://localhost:8000/health
```

#### High Memory Usage

```bash
# Check memory usage
docker stats mcp-server
kubectl top pods

# Analyze memory leaks
python -m memory_profiler server.py

# Increase limits
kubectl set resources deployment mcp-server --limits=memory=2Gi
```

#### Slow Response Times

```bash
# Check database connections
SELECT count(*) FROM pg_stat_activity;

# Check cache hit rate
redis-cli info stats

# Enable query logging
SET log_min_duration_statement = 100;

# Profile code
python -m cProfile -o output.prof server.py
```

#### Connection Timeouts

```python
# Increase timeouts
httpx.AsyncClient(timeout=30.0)

# Add retries
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_external_api():
    # API call
    pass
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debugger
python -m pdb server.py

# Use breakpoints
import pdb; pdb.set_trace()
```

---

## Rollback Procedures

### Kubernetes

```bash
# View deployment history
kubectl rollout history deployment/mcp-server

# Rollback to previous version
kubectl rollout undo deployment/mcp-server

# Rollback to specific revision
kubectl rollout undo deployment/mcp-server --to-revision=2

# Check rollout status
kubectl rollout status deployment/mcp-server
```

### Docker

```bash
# Tag previous version
docker tag mcp-server:v1.0.0 mcp-server:latest

# Restart with previous version
docker-compose up -d
```

---

## Best Practices

1. **Always use health checks** - Kubernetes/Docker can restart unhealthy containers
2. **Implement graceful shutdown** - Handle SIGTERM properly
3. **Use connection pooling** - Reuse database connections
4. **Enable caching** - Reduce database load
5. **Monitor everything** - You can't fix what you can't see
6. **Test in staging first** - Never deploy directly to production
7. **Use blue-green deployments** - Zero-downtime deployments
8. **Automate everything** - CI/CD pipelines
9. **Document runbooks** - Help future you
10. **Regular backups** - Test restore procedures

---

## Checklist

- [ ] Code tested and reviewed
- [ ] Security scan passed
- [ ] Configuration validated
- [ ] Secrets in vault
- [ ] Monitoring configured
- [ ] Alerts set up
- [ ] Documentation updated
- [ ] Runbooks created
- [ ] Backup strategy defined
- [ ] Rollback plan tested
- [ ] Team trained
- [ ] Stakeholders notified

---

**Remember**: Deployment is not the end—it's the beginning of the operational phase!

**Last Updated**: January 2025
