# Documentation Index

## 📖 Main Documentation (START HERE)

### For Users & Operators
**[OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md)** - Complete operational guide
- Quick start
- Installation
- Configuration
- Running ingestion (sequential & parallel)
- Deployment (Docker, Kubernetes)
- Monitoring & troubleshooting
- Performance tuning
- API usage
- Maintenance

**👉 Read this first if you want to:** Install, configure, run, or deploy the system

---

### For Developers & Architects
**[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)** - Technical deep dive
- System architecture
- Performance bottleneck analysis
- Parallel processing architecture
- Error tracking system
- Benchmarking framework
- Code architecture
- Performance optimizations
- Technical decisions

**👉 Read this if you want to:** Understand internals, optimize performance, or contribute code

---

## 📁 Additional Documentation

### Quick References
- **[README.md](README.md)** - Project overview and quick start
- **[QUICKSTART.md](QUICKSTART.md)** - Minimal setup guide
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment-specific details

### API & Configuration
- **[docs/API.md](docs/API.md)** - API endpoints and usage
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** - Configuration options
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues

### Scripts & Tools
- **[scripts/README.md](scripts/README.md)** - Ingestion scripts overview
- **[scripts/analyze_errors.py](scripts/analyze_errors.py)** - Error analysis tool
- **[scripts/compare_benchmarks.py](scripts/compare_benchmarks.py)** - Performance comparison

### Examples
- **[examples/README.md](examples/README.md)** - Usage examples

---

## 🗂️ Archived Documentation

The following files have been consolidated into the main guides above:

**Location:** [docs/archive/](docs/archive/)
- `ARCHITECTURE_IMPROVEMENTS.md` → Merged into TECHNICAL_ARCHITECTURE.md
- `COMPLETE_SUMMARY.md` → Split into OPERATIONS_GUIDE.md + TECHNICAL_ARCHITECTURE.md
- `FIXES_SUMMARY.md` → Incorporated into TECHNICAL_ARCHITECTURE.md
- `MERGE_SUMMARY.md` → Incorporated into TECHNICAL_ARCHITECTURE.md
- `README_ENHANCEMENTS.md` → Merged into OPERATIONS_GUIDE.md

These files are kept for historical reference but are no longer maintained.

---

## 🎯 Documentation by Task

### I want to...

**Install and run the system**
→ Start with [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) § Quick Start

**Deploy to production**
→ [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) § Deployment

**Optimize performance**
→ [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) § Performance Tuning
→ [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) § Performance Bottleneck Analysis

**Troubleshoot errors**
→ [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) § Monitoring & Troubleshooting
→ Use `scripts/analyze_errors.py`

**Understand the architecture**
→ [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) § System Architecture

**Contribute code**
→ [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) § Code Architecture

**Configure the system**
→ [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) § Configuration
→ [docs/CONFIGURATION.md](docs/CONFIGURATION.md)

**Use the API**
→ [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) § API Usage
→ [docs/API.md](docs/API.md)

---

## 📊 Quick Stats

**Main Guides:**
- OPERATIONS_GUIDE.md: ~2,500 lines (comprehensive operational guide)
- TECHNICAL_ARCHITECTURE.md: ~1,800 lines (technical deep dive)

**Total Documentation:** 4,300+ lines of consolidated, organized content

**Archived Files:** 5 files moved to `docs/archive/` (kept for reference)

---

## 🔄 Documentation Maintenance

**Primary Maintainers Should Update:**
1. OPERATIONS_GUIDE.md - For user-facing changes
2. TECHNICAL_ARCHITECTURE.md - For technical changes

**Deprecated:**
- Individual feature READMEs (consolidated)
- Scattered architecture docs (consolidated)
- Multiple getting started guides (unified)

**Active:**
- Main 2 guides (comprehensive)
- Specific API/config docs (detailed)
- Script-specific READMEs (targeted)

---

## 📝 Documentation Philosophy

**Consolidated Approach:**
✅ 2 comprehensive guides instead of 10+ scattered files
✅ Clear separation: Operations vs Technical
✅ Complete information in one place
✅ Easy to navigate and search
✅ Reduced maintenance burden

**Old Approach (Deprecated):**
❌ Multiple overlapping guides
❌ Information scattered across files
❌ Unclear which doc to read first
❌ Duplicated content
❌ Hard to keep in sync

---

## 🚀 Getting Started Paths

### Path 1: I just want to use it
1. Read [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) § Quick Start
2. Follow installation steps
3. Run parallel ingestion
4. Start querying

**Time:** 15-30 minutes

### Path 2: I want to deploy it
1. Read [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) § Quick Start
2. Read [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) § Deployment
3. Choose deployment method (Docker/K8s)
4. Follow deployment steps

**Time:** 1-2 hours

### Path 3: I want to understand it
1. Read [README.md](README.md) for overview
2. Read [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) § System Architecture
3. Read [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) § Performance Bottleneck Analysis
4. Explore code with architecture understanding

**Time:** 2-3 hours

### Path 4: I want to optimize it
1. Read [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) § Performance Bottleneck Analysis
2. Read [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) § Performance Tuning
3. Run benchmarks: `scripts/compare_benchmarks.py`
4. Adjust configuration based on results

**Time:** 1-2 hours

---

## 📞 Support

**For operational questions:** See OPERATIONS_GUIDE.md § Troubleshooting
**For technical questions:** See TECHNICAL_ARCHITECTURE.md
**For bugs/issues:** Check error reports in `error_reports/`
**For performance:** Use `scripts/compare_benchmarks.py`

---

## ✅ Checklist for New Users

- [ ] Read OPERATIONS_GUIDE.md § Quick Start
- [ ] Install dependencies
- [ ] Configure `.env` file
- [ ] Run test ingestion with `--parallel`
- [ ] Check benchmark results
- [ ] Review error reports (should be 0)
- [ ] Test API endpoints
- [ ] Deploy to production (if needed)

---

**Last Updated:** 2026-03-22
**Documentation Version:** 2.0 (Consolidated)
