# CI/CD Pipeline

Automated testing, linting, and deployment workflows for ML-Monitor.

## Workflows

### CI Pipeline (`.github/workflows/ci.yml`)

Runs on every push and pull request to main:

**Jobs:**
1. **Lint and Test**
   - Code formatting check (Black)
   - Import sorting (isort)
   - Linting (flake8)
   - Unit and integration tests
   - Coverage reporting

2. **Docker Build**
   - Builds API Docker image
   - Tests image functionality
   - Runs only after tests pass

3. **Security Scan**
   - Trivy vulnerability scanning
   - Uploads results to GitHub Security
   - Checks for critical/high severity issues

### Model Training Pipeline (`.github/workflows/model-training.yml`)

Manual workflow for training models:

**Features:**
- Configurable hyperparameters
- MLflow experiment tracking
- Artifact upload for reproducibility
- Can be triggered manually or on schedule

**Usage:**
```bash
gh workflow run model-training.yml \
  -f n_estimators=100 \
  -f max_depth=10
```

## Pre-commit Hooks

Install hooks locally:
```bash
pip install pre-commit
pre-commit install
```

**Checks performed:**
- Trailing whitespace
- End of file fixes
- YAML/JSON validation
- Large file detection
- Private key detection
- Black formatting
- isort import sorting
- flake8 linting
- mypy type checking

## Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

Run only integration tests:
```bash
pytest -m integration
```

## Code Quality Standards

- **Line length**: 120 characters
- **Formatting**: Black
- **Import sorting**: isort (Black profile)
- **Linting**: flake8 (ignores: E203, W503)
- **Type hints**: Checked with mypy

## Security

- Trivy scans for vulnerabilities in code and dependencies
- Results uploaded to GitHub Security tab
- Fails on critical/high severity issues

## Branch Protection

Recommended branch protection rules for `main`:
- Require pull request reviews
- Require status checks to pass (CI pipeline)
- Require branches to be up to date
- No force pushes
- No deletions

## Secrets Configuration

Required secrets for full pipeline:
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `CODECOV_TOKEN`: For coverage reporting (optional)

## Future Enhancements

- Automatic model deployment on successful training
- Performance regression testing
- Load testing with Locust
- Automated API documentation deployment
- Scheduled data quality checks
- Slack/email notifications for failures
