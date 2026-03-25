# Documentation Index

Complete guide to RAG Hybrid Search.

---

## Getting Started

- **[Quick Start Guide](QUICKSTART.md)** ⭐ *Start here!*
  - 5-minute setup
  - First query example
  - Basic usage

- **[Main README](../README.md)**
  - Project overview
  - Key features
  - Installation

---

## Deep Dives

- **[Medium Article](../MEDIUM_ARTICLE.md)** 📖
  - Project journey
  - Technical decisions
  - Performance insights
  - Lessons learned

- **[Enhanced Score Fusion](../app/core/search/README.md)** 🔬
  - RRF explained
  - Heuristics deep dive
  - Benchmarks
  - Configuration examples

- **[Code Reorganization](../REORGANIZATION.md)** 🏗️
  - Module structure
  - Migration guide
  - Import changes
  - Design decisions

---

## Configuration

- **[Configuration Guide](CONFIGURATION.md)** ⚙️
  - All settings explained
  - Environment variables
  - Use case examples
  - Performance tuning

---

## Examples

- **[Enhanced Fusion Demo](../examples/enhanced_fusion_demo.py)** 💡
  - Compare fusion methods
  - Metadata boosting
  - Quality scoring
  - Interactive examples

---

## Archived

Historical documentation (may be outdated):

- [Deployment Journey](archive/DEPLOYMENT_JOURNEY.md)
- [Optimization Summary](archive/OPTIMIZATION_SUMMARY.md)
- [Phase 3 Optimizations](archive/PHASE3_OPTIMIZATIONS.md)
- [Final Test Summary](archive/FINAL_TEST_SUMMARY.md)
- [Operations Guide](archive/OPERATIONS_GUIDE.md) *(outdated)*
- [Technical Architecture](archive/TECHNICAL_ARCHITECTURE.md) *(outdated)*

---

## Quick Links

| Topic | Document |
|-------|----------|
| 🚀 Get started | [Quick Start](QUICKSTART.md) |
| ⚡ Performance | [Medium Article](../MEDIUM_ARTICLE.md) |
| 🔧 Configuration | [Configuration](CONFIGURATION.md) |
| 🧩 Code structure | [Reorganization](../REORGANIZATION.md) |
| 🔍 Score fusion | [Enhanced Fusion](../app/core/search/README.md) |
| 💻 Examples | [Demo Script](../examples/enhanced_fusion_demo.py) |

---

## Project Structure

```
rag-hybrid-search/
├── README.md                    # Project overview
├── MEDIUM_ARTICLE.md            # Project story & lessons
├── REORGANIZATION.md            # Code structure
│
├── docs/
│   ├── INDEX.md                 # This file
│   ├── QUICKSTART.md            # 5-minute guide
│   ├── CONFIGURATION.md         # Settings guide
│   └── archive/                 # Historical docs
│
├── app/core/
│   ├── search/README.md         # Enhanced fusion guide
│   └── ... (module docs)
│
└── examples/
    └── enhanced_fusion_demo.py  # Interactive examples
```

---

**Can't find what you're looking for?** Check the [main README](../README.md) or browse the [code structure guide](../REORGANIZATION.md).
