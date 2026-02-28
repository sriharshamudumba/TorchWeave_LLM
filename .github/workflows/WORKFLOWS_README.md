# TorchWeave LLM – GitHub Actions Workflows

Place these files in `.github/workflows/` in your repository.

## Workflow Overview

| File | Trigger | Purpose |
|------|---------|---------|
| `ci.yml` | Push / PR to `main`, `develop` | Lint → Test → Docker build validation |
| `cd.yml` | Push to `main` or version tag `v*.*.*` | Build & push Docker images to GHCR, create GitHub Release |
| `security.yml` | Push / PR + weekly cron | pip-audit, Bandit SAST, Trivy container scan, Gitleaks secret detection |
| `benchmark.yml` | Push to `main` / PR / manual | Inference latency benchmarks, PR comment with results |
| `stale.yml` | Daily cron | Auto-label and close stale issues (30d) and PRs (14d) |

---

## Setup Steps

### 1. Copy files
```
.github/
  workflows/
    ci.yml
    cd.yml
    security.yml
    benchmark.yml
    stale.yml
```

### 2. Enable GitHub Container Registry
- Go to **Settings → Packages** and ensure GHCR is enabled.
- The `GITHUB_TOKEN` is used automatically — no extra secret needed for GHCR pushes.

### 3. Optional secrets
Add these under **Settings → Secrets and variables → Actions** if needed:

| Secret | Used by | Notes |
|--------|---------|-------|
| `CODECOV_TOKEN` | `ci.yml` | Only required for private repos |

### 4. GPU self-hosted runner (for real benchmarks)
The benchmark workflow runs on `ubuntu-latest` by default (CPU only).  
For actual GPU inference benchmarks, change the `runs-on` key in `benchmark.yml`:
```yaml
runs-on: [self-hosted, gpu]
```

### 5. Create your first release
```bash
git tag v0.1.0
git push origin v0.1.0
```
This triggers `cd.yml` → builds all 4 service images → creates a GitHub Release.

---

## CI Job Flow

```
Push / PR
    │
    ├── lint (black, isort, flake8, mypy)
    │       │
    │       └── test (pytest + coverage, matrix: py3.11 / py3.12)
    │               │
    │               └── docker-build (compose build --parallel)
    │
    └── security (parallel: pip-audit, bandit, trivy, gitleaks)
```

---

## Notes
- `mypy` is `continue-on-error: true` until type stubs are complete.
- The benchmark script (`scripts/benchmark.py`) is referenced but not yet created — add it when ready.
- Trivy scans the `server` image; extend to other services as needed.
