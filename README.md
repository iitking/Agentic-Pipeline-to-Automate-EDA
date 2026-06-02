# Agentic Pipeline to Automate EDA

An AI-powered, agentic Exploratory Data Analysis (EDA) system that automatically profiles datasets, detects data-quality issues, selects relevant analyses, generates visualizations, validates findings, and produces a comprehensive report with actionable insights.

The goal is to combine **reliable Python-based analytics** with **LLM-powered planning and interpretation**. Python performs calculations; the LLM helps decide what to investigate and explains verified results.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. High-Level Architecture](#2-high-level-architecture)
- [3. Quick Start & Installation](#3-quick-start--installation)
- [4. Usage Guide](#4-usage-guide)
  - [4.1 Command Line Interface (CLI)](#41-command-line-interface-cli)
  - [4.2 Streamlit Interactive Web Studio](#42-streamlit-interactive-web-studio)
  - [4.3 FastAPI REST API](#43-fastapi-rest-api)
  - [4.4 Docker & Docker Compose](#44-docker--docker-compose)
- [5. Agent Responsibilities](#5-agent-responsibilities)
- [6. Agentic Workflow & Feedback Loop](#6-agentic-workflow--feedback-loop)
- [7. Project Structure](#7-project-structure)
- [8. Shared State Design](#8-shared-state-design)
- [9. Safety, Sandboxing & Reliability](#9-safety-sandboxing--reliability)
- [10. Evaluation Strategy](#10-evaluation-strategy)
- [11. Testing & CI/CD](#11-testing--cicd)

---

## 1. Project Overview

Traditional EDA notebooks generally follow a fixed sequence of operations. This project builds a system that **adapts its analysis plan to the dataset it receives**.

### Core Capabilities

- **Multi-format Ingestion:** Accepts CSV, Excel (`.xlsx`, `.xls`), and Parquet (`.parquet`) datasets.
