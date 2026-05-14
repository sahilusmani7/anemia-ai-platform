# AI-Powered Medical Inference Platform with CI/CD, Monitoring, and Cloud-Native Deployment

## Overview

An end-to-end AI-powered medical inference platform for non-invasive anemia detection using conjunctival eye images. The platform combines deep learning-based feature extraction with a Random Forest classifier and operationalizes the system using modern DevOps and SRE practices.

The project was transformed from a standalone machine learning application into a production-style deployment platform featuring:

* Dockerized deployment
* CI/CD automation
* Multi-container orchestration
* Observability and monitoring
* Infrastructure-style architecture
* Automated build workflows
* Real-time metrics collection

---

# Key Features

## AI/ML Features

* CNN-based feature extraction
* Random Forest classification
* Image preprocessing pipeline
* Real-time inference predictions
* Medical image classification workflow

## DevOps Features

* Dockerized application runtime
* Docker Compose multi-service orchestration
* GitHub Actions CI/CD automation
* Reproducible deployment environments
* Automated Docker image builds

## Observability & Monitoring

* Prometheus metrics scraping
* Grafana dashboard visualization
* Request monitoring
* Application telemetry exposure
* Production-style monitoring architecture

---

# System Architecture

```text
                    ┌────────────────────┐
                    │    User / Client   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   Flask Web App    │
                    │  AI Inference API  │
                    └─────────┬──────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
        ┌────────────────┐      ┌──────────────────┐
        │ CNN + RF Model │      │ Metrics Endpoint │
        │   Prediction   │      │    /metrics      │
        └────────────────┘      └─────────┬────────┘
                                           │
                                           ▼
                               ┌────────────────────┐
                               │    Prometheus      │
                               │ Metrics Collection │
                               └─────────┬──────────┘
                                         │
                                         ▼
                               ┌────────────────────┐
                               │      Grafana       │
                               │ Visualization UI   │
                               └────────────────────┘


                CI/CD AUTOMATION PIPELINE

      GitHub Push
            │
            ▼
   GitHub Actions Workflow
            │
   ┌────────┴────────┐
   ▼                 ▼
Install Dependencies  Build Docker Image
```

---

# Technology Stack

| Category              | Technology         |
| --------------------- | ------------------ |
| Backend Framework     | Flask              |
| Machine Learning      | TensorFlow / Keras |
| ML Classifier         | Random Forest      |
| Containerization      | Docker             |
| Orchestration         | Docker Compose     |
| CI/CD                 | GitHub Actions     |
| Monitoring            | Prometheus         |
| Visualization         | Grafana            |
| Version Control       | Git + GitHub       |
| Language              | Python             |
| Operating Environment | Linux Containers   |

---

# CI/CD Workflow

The platform uses GitHub Actions to automate:

* Dependency installation
* Docker image builds
* CI validation workflows
* Deployment pipeline verification

Every push to the main branch automatically triggers the CI pipeline.

---

# Monitoring & Observability

The application exposes a Prometheus-compatible metrics endpoint:

```text
/metrics
```

Prometheus scrapes telemetry data from the Flask application and Grafana visualizes the collected metrics through dashboards.

Current observability capabilities include:

* Request monitoring
* Application telemetry
* Service health visibility
* Infrastructure-style metrics collection

---

# Dockerized Deployment

The application was containerized using Docker to provide:

* Reproducible environments
* Portable deployments
* Dependency isolation
* Cross-platform execution
* Cloud deployment readiness

The complete stack is orchestrated using Docker Compose.

---

# Running the Application

## Clone Repository

```bash
git clone https://github.com/sahilusmani7/anemia-ai-platform.git
cd anemia-ai-platform
```

---

## Create Virtual Environment

```bash
py -3.11 -m venv venv
```

---

## Activate Environment

```bash
.\venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Flask Application

```bash
python app.py
```

---

## Run Dockerized Stack

```bash
docker compose up --build
```

---

# Service Ports

| Service           | Port |
| ----------------- | ---- |
| Flask Application | 5000 |
| Prometheus        | 9090 |
| Grafana           | 3000 |

---

# Future Improvements

* Kubernetes deployment
* Terraform infrastructure automation
* NGINX reverse proxy
* Cloud deployment on AWS/Render
* HTTPS configuration
* Load balancing
* Distributed monitoring

---

# Resume Highlights

* Developed a containerized AI-powered medical inference platform using Flask, TensorFlow, and Docker
* Implemented CI/CD automation using GitHub Actions for Docker-based deployment workflows
* Built observability stack using Prometheus and Grafana for metrics collection and monitoring
* Orchestrated multi-container services using Docker Compose
* Designed production-style deployment architecture for reproducible ML inference workflows

---

# Project Outcome

This project demonstrates the integration of:

* Machine Learning
* Backend Engineering
* DevOps Automation
* Observability
* Infrastructure-Oriented Deployment

into a unified production-style AI platform suitable for DevOps, SRE, Cloud, and MLOps-oriented engineering roles.
