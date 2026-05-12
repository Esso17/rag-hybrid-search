# ── RAG Hybrid Search — Minikube deployment ────────────────────────────────────
# Prerequisites: minikube, kubectl, docker, python 3.11+, ollama
# Quick start:  make setup && make deploy && make ingest && make test

API_URL    ?= http://localhost:8000
DOCS_K8S   ?= data/docs/kubernetes
BATCH_SIZE ?= 5
WORKERS    ?= 1
CONCURRENT ?= 2

# ── Local / Docker Compose ─────────────────────────────────────────────────────

.PHONY: install
install:                          ## Install Python dependencies
	pip install -r requirements.txt

.PHONY: models
models:                           ## Pull required Ollama models (run once)
	ollama pull phi3.5:3.8b
	ollama pull nomic-embed-text

.PHONY: up
up:                               ## Start the API with Docker Compose
	docker compose up -d

.PHONY: down
down:                             ## Stop the API
	docker compose down

.PHONY: logs-compose
logs-compose:                     ## Tail Docker Compose logs
	docker compose logs -f

# ── Minikube ───────────────────────────────────────────────────────────────────

.PHONY: minikube-start
minikube-start:                   ## Start Minikube (4 CPU, 8 GB RAM)
	minikube start --cpus=4 --memory=8192 --disk-size=40g

.PHONY: build
build:                            ## Build Docker image inside Minikube
	eval $$(minikube docker-env) && docker build -t rag-hybrid-search:latest .

.PHONY: deploy
deploy: build                     ## Deploy app + monitoring to Minikube
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/monitoring/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/pvc.yaml
	kubectl apply -f k8s/app-deployment.yaml
	kubectl apply -f k8s/app-service.yaml
	kubectl apply -f k8s/monitoring/
	kubectl rollout status deployment/rag-api -n rag-hybrid-search --timeout=120s

.PHONY: port-forward
port-forward:                     ## Forward API + monitoring ports (runs in background)
	kubectl port-forward -n rag-hybrid-search svc/rag-api-service 8000:8000 &
	kubectl port-forward -n monitoring svc/prometheus 9090:9090 &
	kubectl port-forward -n monitoring svc/grafana 3000:3000 &
	@echo "API      → http://localhost:8000/docs"
	@echo "Prometheus → http://localhost:9090"
	@echo "Grafana  → http://localhost:3000"

.PHONY: logs
logs:                             ## Tail rag-api pod logs
	kubectl logs -n rag-hybrid-search -l app=rag-api -f --tail=50

.PHONY: status
status:                           ## Show pod + service status
	kubectl get pods,svc -n rag-hybrid-search
	kubectl get pods,svc -n monitoring

# ── Ingestion ──────────────────────────────────────────────────────────────────

.PHONY: ingest
ingest:                           ## Ingest Kubernetes docs via live API
	python scripts/ingest_docs.py \
		--source $(DOCS_K8S) --name Kubernetes \
		--api --api-url $(API_URL) \
		--batch-size $(BATCH_SIZE) --workers $(WORKERS) --concurrent $(CONCURRENT)

# ── Tests ──────────────────────────────────────────────────────────────────────

.PHONY: test
test:                             ## Run all tests (unit + API via TestClient; skips slow/LLM tests)
	python -m pytest tests/test_core_modules.py tests/test_api.py -v --tb=short

.PHONY: test-slow
test-slow:                        ## Run all tests including slow tests (requires live Ollama)
	python -m pytest tests/test_core_modules.py tests/test_api.py -v --tb=short --slow

.PHONY: test-unit
test-unit:                        ## Run unit tests only (no live server needed)
	python -m pytest tests/test_core_modules.py -v --tb=short

.PHONY: test-api
test-api:                         ## Run API tests via FastAPI TestClient
	python -m pytest tests/test_api.py -v --tb=short

# ── Checks ─────────────────────────────────────────────────────────────────────

.PHONY: lint
lint:                             ## Run ruff linter (F + E rules)
	ruff check app/ tests/ --select F,E --ignore E501

.PHONY: health
health:                           ## Quick health check against running API
	curl -s $(API_URL)/health | python3 -m json.tool

.PHONY: stats
stats:                            ## Show current index + cache stats
	curl -s $(API_URL)/stats | python3 -m json.tool

# ── Cleanup ────────────────────────────────────────────────────────────────────

.PHONY: teardown
teardown:
	kubectl delete namespace rag-hybrid-search --ignore-not-found
	kubectl delete namespace monitoring --ignore-not-found

.PHONY: minikube-stop
minikube-stop:
	minikube stop

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*##"}{printf "  \033[36m%-20s\033[0m %s\n",$$1,$$2}'

.DEFAULT_GOAL := help
