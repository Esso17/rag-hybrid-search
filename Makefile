.PHONY: help build run stop clean deploy-minikube undeploy-minikube test logs shell

# Colors for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
NC=\033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)RAG Hybrid Search - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# Local Development
build: ## Build Docker image locally
	@echo "$(YELLOW)Building Docker image...$(NC)"
	docker build -t rag-hybrid-search:latest .
	@echo "$(GREEN)✓ Build complete$(NC)"

run: ## Run with docker-compose
	@echo "$(YELLOW)Starting services with docker-compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"

stop: ## Stop docker-compose services
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

clean: ## Clean up docker resources
	@echo "$(YELLOW)Cleaning up...$(NC)"
	docker-compose down -v
	docker rmi rag-hybrid-search:latest 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

logs: ## View docker-compose logs
	docker-compose logs -f rag-api

restart: stop run ## Restart docker-compose services

rebuild: ## Rebuild and restart
	docker-compose up -d --build

# Minikube Deployment
minikube-start: ## Start Minikube
	@echo "$(YELLOW)Starting Minikube...$(NC)"
	minikube start --memory=4096 --cpus=2
	@echo "$(GREEN)✓ Minikube started$(NC)"

deploy-minikube: ## Deploy to Minikube
	@echo "$(YELLOW)Deploying to Minikube...$(NC)"
	cd k8s && ./deploy.sh

undeploy-minikube: ## Remove from Minikube
	@echo "$(YELLOW)Removing from Minikube...$(NC)"
	cd k8s && ./undeploy.sh

minikube-status: ## Show Minikube deployment status
	@echo "$(GREEN)Minikube Resources:$(NC)"
	kubectl get all -n rag-hybrid-search

minikube-logs: ## View Minikube pod logs
	kubectl logs -f deployment/rag-api -n rag-hybrid-search

minikube-url: ## Get Minikube service URL
	@echo "$(GREEN)Service URL:$(NC)"
	@minikube service rag-api-service -n rag-hybrid-search --url

minikube-dashboard: ## Open Kubernetes dashboard
	minikube dashboard

# Development
setup: ## Install Python dependencies
	@echo "$(YELLOW)Installing dependencies...$(NC}"
	pip3 install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

test: ## Run tests
	@echo "$(YELLOW)Running tests...$(NC)"
	pytest tests/ -v

lint: ## Run linting
	@echo "$(YELLOW)Running linters...$(NC)"
	ruff check app/
	black --check app/

format: ## Format code
	@echo "$(YELLOW)Formatting code...$(NC)"
	black app/
	ruff check --fix app/

shell: ## Open shell in running container
	docker-compose exec rag-api /bin/bash

# Docker operations
docker-build-minikube: ## Build image in Minikube's Docker
	@echo "$(YELLOW)Building in Minikube Docker...$(NC)"
	eval $$(minikube docker-env) && docker build -t rag-hybrid-search:latest .
	@echo "$(GREEN)✓ Build complete$(NC)"

# Kubernetes operations
k8s-apply: ## Apply all Kubernetes manifests
	kubectl apply -f k8s/all-in-one.yaml

k8s-delete: ## Delete all Kubernetes resources
	kubectl delete -f k8s/all-in-one.yaml

k8s-restart: ## Restart deployment
	kubectl rollout restart deployment/rag-api -n rag-hybrid-search

k8s-scale: ## Scale deployment (usage: make k8s-scale REPLICAS=3)
	kubectl scale deployment/rag-api --replicas=$(REPLICAS) -n rag-hybrid-search

# Monitoring
watch: ## Watch pod status
	watch kubectl get pods -n rag-hybrid-search

events: ## Show recent events
	kubectl get events -n rag-hybrid-search --sort-by='.lastTimestamp'

describe: ## Describe deployment
	kubectl describe deployment/rag-api -n rag-hybrid-search

# Documentation download
download-k8s: ## Download Kubernetes documentation (usage: make download-k8s K8S_VERSION=v1.29)
	@echo "$(YELLOW)Downloading Kubernetes documentation...$(NC)"
	bash scripts/download_k8s_docs.sh $(if $(K8S_VERSION),--version $(K8S_VERSION),)

download-cilium: ## Download Cilium documentation (usage: make download-cilium CILIUM_VERSION=v1.15)
	@echo "$(YELLOW)Downloading Cilium documentation...$(NC)"
	bash scripts/download_cilium_docs.sh $(if $(CILIUM_VERSION),--version $(CILIUM_VERSION),)

download-all: ## Download both K8s and Cilium docs
	@echo "$(YELLOW)Downloading all documentation...$(NC)"
	bash scripts/download_all_docs.sh

download-and-ingest: ## Download and automatically ingest all docs
	@echo "$(YELLOW)Downloading and ingesting all documentation...$(NC)"
	bash scripts/download_all_docs.sh --ingest

# Data ingestion
ingest-k8s: ## Ingest Kubernetes docs (usage: make ingest-k8s [SOURCE=/path/to/docs])
	python3 scripts/ingest_k8s_docs.py --source $(or $(SOURCE),data/docs/kubernetes)

ingest-cilium: ## Ingest Cilium docs (usage: make ingest-cilium [SOURCE=/path/to/docs])
	python3 scripts/ingest_cilium_docs.py --source $(or $(SOURCE),data/docs/cilium)
