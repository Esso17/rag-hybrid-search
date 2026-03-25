# RAG Hybrid Search on Minikube - Deployment Journey

A detailed account of deploying a production-ready RAG (Retrieval-Augmented Generation) application with Qdrant vector database and Ollama LLM in a local Kubernetes environment.

## 🎯 Project Overview

Built a hybrid search system combining semantic search (embeddings) with traditional keyword search (BM25) for Kubernetes and Cilium documentation. The system uses:
- **FastAPI** for the REST API
- **Qdrant** for vector storage
- **Ollama** with Mistral LLM for text generation
- **nomic-embed-text** for embeddings
- **Kubernetes** (Minikube) for orchestration

## 🚧 Challenges Encountered & Solutions

### Challenge 1: Image Pull Policy Error

**Problem:**
```
Error: ErrImageNeverPull
Container image "rag-hybrid-search:latest" is not present with pull policy of Never
```

**Root Cause:**
Minikube runs its own Docker daemon, separate from the host machine. The deployment was configured with `imagePullPolicy: Never`, expecting the image to exist locally, but it was built on the host machine's Docker, not Minikube's.

**Solution:**
Built the image directly in Minikube's Docker environment:
```bash
eval $(minikube docker-env)
docker build -t rag-hybrid-search:latest .
```

**Key Learning:** In Minikube, you must build images in the cluster's Docker daemon or use `imagePullPolicy: IfNotPresent` with a registry.

---

### Challenge 2: Qdrant Readiness Probe Failure

**Problem:**
```
Readiness probe failed: HTTP probe failed with statuscode: 404
```

**Root Cause:**
The Qdrant deployment was configured with a readiness probe checking `/readiness` endpoint, but Qdrant doesn't expose this endpoint. Only `/` is available.

**Solution:**
Updated `k8s/qdrant-deployment.yaml`:
```yaml
readinessProbe:
  httpGet:
    path: /  # Changed from /readiness
    port: http
```

**Key Learning:** Always verify probe endpoints against actual API documentation. Common convention doesn't guarantee endpoint availability.

---

### Challenge 3: Ollama LLM Connection Refused

**Problem:**
```
Failed to connect to Qdrant: [Errno 111] Connection refused
LLM connection error: [Errno -2] Name or service not known
```

**Root Cause:**
The RAG API pods were trying to connect to Ollama at `http://ollama-service:11434`, but no Ollama instance was deployed in the cluster.

**Initial Decision Point:**
Two options emerged:
1. Run Ollama on host machine (accessible via `host.minikube.internal`)
2. Deploy Ollama inside Minikube

**Solution Chosen:** Deploy Ollama inside Minikube for:
- Complete isolation and production-like setup
- Self-contained environment
- Automatic model management
- Better for demonstration/portfolio purposes

---

### Challenge 4: Automatic Model Loading

**Problem:**
Ollama deployment worked, but models (mistral: 4.4GB, nomic-embed-text: 274MB) needed to be pulled manually after each deployment.

**Failed Approach #1:** Kubernetes Job
- Required manual execution
- Needed separate orchestration
- Not idempotent across pod restarts

**Failed Approach #2:** Init script with `curl`
```bash
curl: command not found
```
The ollama/ollama image doesn't include curl.

**Solution:**
Created a smart initialization script using native Ollama CLI:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ollama-init-script
data:
  init-models.sh: |
    #!/bin/bash
    set -ex

    # Start Ollama server
    /bin/ollama serve &
    OLLAMA_PID=$!

    # Wait for Ollama to be ready using native CLI
    until ollama list > /dev/null 2>&1; do
      sleep 2
    done

    # Smart model pulling - checks if exists first
    if ollama list | grep -q "mistral"; then
      echo "✓ mistral model already exists"
    else
      ollama pull mistral
    fi

    if ollama list | grep -q "nomic-embed-text"; then
      echo "✓ nomic-embed-text model already exists"
    else
      ollama pull nomic-embed-text
    fi

    # Keep Ollama running
    wait $OLLAMA_PID
```

**Key Features:**
- ✅ Idempotent: Checks if models exist before pulling
- ✅ Automatic: Runs on pod startup
- ✅ Persistent: Models stored in PVC, survive pod restarts
- ✅ Verbose logging: Easy troubleshooting with `set -ex`
- ✅ No external dependencies: Uses only native tools

---

## 📐 Architecture Decisions

### 1. Storage Strategy
**PersistentVolumeClaim for Ollama (20GB)**
- Models persist across pod restarts
- Prevents re-downloading 4.7GB on every deployment
- Smart init script leverages this for idempotency

### 2. Resource Allocation
```yaml
resources:
  requests:
    memory: "4Gi"    # Minimum for LLM operation
    cpu: "1000m"
  limits:
    memory: "8Gi"    # Prevent OOM kills
    cpu: "2000m"
```

### 3. Probe Configuration
```yaml
livenessProbe:
  initialDelaySeconds: 120  # Allow time for model loading
  periodSeconds: 30
  failureThreshold: 5

readinessProbe:
  initialDelaySeconds: 90   # Models must be loaded first
  failureThreshold: 10
```

Increased delays account for initial model pull time (~3-5 minutes).

### 4. Deployment Strategy
```yaml
strategy:
  type: Recreate  # PVC can only be mounted by one pod
```
Rolling updates aren't possible with ReadWriteOnce PVCs.

---

## 🎓 Technical Insights

### Minikube Networking
- `host.minikube.internal` provides secure host access
- Only accessible from within the cluster
- Resolves to `192.168.65.254` (gateway to host)
- Perfect for development without compromising security

### Init Containers vs Init Scripts
**Why init script in main container vs init container?**
- Ollama server must keep running (not one-time initialization)
- Init containers exit after completion
- Main container runs as PID 1 with custom entrypoint
- Allows Ollama to serve while models are pulled

### Kubernetes Resource Management
**Why not use Jobs for model pulling?**
- Jobs require external orchestration
- Not automatically triggered on pod recreation
- Init script is self-contained and atomic

---

## 📊 Final Architecture

```
┌─────────────────────────────────────────────┐
│           Minikube Cluster                  │
│                                             │
│  ┌──────────────┐      ┌──────────────┐   │
│  │   RAG API    │────▶ │   Qdrant     │   │
│  │  (FastAPI)   │      │ (Vector DB)  │   │
│  │  2 replicas  │      │   Ready ✓    │   │
│  └──────┬───────┘      └──────────────┘   │
│         │                                   │
│         │                                   │
│         ▼                                   │
│  ┌──────────────┐      ┌──────────────┐   │
│  │   Ollama     │◀─────│  Init Script │   │
│  │   Service    │      │  ConfigMap   │   │
│  └──────┬───────┘      └──────────────┘   │
│         │                                   │
│         ▼                                   │
│  ┌──────────────┐                          │
│  │ Ollama Pod   │                          │
│  │ ┌──────────┐ │                          │
│  │ │ mistral  │ │  4.4 GB                  │
│  │ │  (LLM)   │ │                          │
│  │ └──────────┘ │                          │
│  │ ┌──────────┐ │                          │
│  │ │  nomic   │ │  274 MB                  │
│  │ │  embed   │ │                          │
│  │ └──────────┘ │                          │
│  └──────┬───────┘                          │
│         │                                   │
│         ▼                                   │
│  ┌──────────────┐                          │
│  │  PVC 20GB    │  Models persist here     │
│  │ (Storage)    │                          │
│  └──────────────┘                          │
└─────────────────────────────────────────────┘
```

---

## ✅ Final Status

| Component | Status | Details |
|-----------|--------|---------|
| **Qdrant** | ✅ Running | Vector database ready |
| **Ollama** | ✅ Running | Both models loaded automatically |
| **RAG API** | ✅ Running | 2 replicas, connected to both services |
| **Models** | ✅ Loaded | mistral (4.4GB), nomic-embed-text (274MB) |
| **Health** | ✅ Passing | All probes green |

### Verification
```bash
# Check all pods
kubectl get pods -n rag-hybrid-search

# Verify models loaded
kubectl exec -n rag-hybrid-search deployment/ollama -- ollama list

# Check API health
kubectl logs -n rag-hybrid-search deployment/rag-api --tail=10
```

---

## 🔄 Alternative Approach: Host-based Ollama

Documented a second deployment option using `host.minikube.internal` for:
- Resource-constrained environments
- Development with GPU acceleration
- Multi-project Ollama sharing

See `k8s/OLLAMA_OPTIONS.md` for switching instructions.

---

## 💡 Key Takeaways

1. **Understand Your Environment**: Minikube's Docker daemon isolation requires different build strategies than typical Docker workflows

2. **Validate Assumptions**: API endpoints, probe paths, and image availability should be verified, not assumed

3. **Design for Automation**: Init scripts that are idempotent and self-checking reduce operational overhead

4. **Resource Planning**: LLMs need substantial resources - plan probe delays and limits accordingly

5. **Document Alternatives**: Production and development environments have different optimization targets

6. **Debugging Strategy**:
   - Start with `kubectl describe pod` for events
   - Use `kubectl logs` for application errors
   - Exec into pods for direct inspection
   - Check service connectivity with `kubectl exec -- curl`

---

## 📚 Repository Structure

```
k8s/
├── app-deployment.yaml         # RAG API deployment
├── qdrant-deployment.yaml      # Vector database
├── qdrant-pvc.yaml            # Storage for Qdrant
├── ollama-deployment.yaml     # LLM deployment ✨
├── ollama-service.yaml        # LLM service ✨
├── ollama-pvc.yaml           # Model storage (20GB) ✨
├── ollama-init-script.yaml   # Auto-pull script ✨
├── configmap.yaml             # App configuration
├── service.yaml               # API service (NodePort)
├── namespace.yaml             # Isolation
├── OLLAMA_OPTIONS.md          # Deployment alternatives ✨
└── all-in-one.yaml           # Complete manifest

✨ = Created during this deployment journey
```

---

## 🚀 Next Steps

1. **Performance Optimization**: Profile model loading times, consider model quantization
2. **Monitoring**: Add Prometheus metrics for Ollama request latency
3. **Scaling**: Explore Ollama model caching strategies for multiple replicas
4. **Security**: Implement NetworkPolicies to restrict inter-pod communication
5. **CI/CD**: Automate image building in Minikube for seamless deployments

---

## 🔗 Technologies Used

- **Kubernetes (Minikube)**: Container orchestration
- **Qdrant**: Vector database for semantic search
- **Ollama**: Local LLM inference (Mistral 7B)
- **FastAPI**: Python web framework
- **Docker**: Containerization
- **Bash**: Initialization scripting

---

**Duration**: ~4 hours from initial deployment to fully automated setup
**Lines of YAML**: ~300 (including init scripts)
**Coffee consumed**: ☕☕☕

---

*This deployment journey demonstrates troubleshooting real-world Kubernetes issues, designing for automation, and balancing development convenience with production-readiness.*
