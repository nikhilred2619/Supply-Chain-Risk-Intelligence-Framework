# Contributing

Thank you for your interest in the LLM-FMEA Supply Chain Risk Assessment framework.

## How to Contribute

### Reporting Issues

Please open a GitHub Issue with:
- Python version and OS
- Full traceback
- Minimal reproduction script

### Pull Requests

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Follow the code style (PEP 8, max line 100 chars)
4. Add tests for new functionality
5. Ensure all 27 tests pass: `python -m pytest tests/ -v`
6. Submit PR with clear description of changes

### Research Extensions

Priority areas for contribution:
- Additional enterprise connector implementations (Microsoft Dynamics, NetSuite)
- Real-time streaming mode (Kafka / Kinesis integration)
- Reinforcement learning for adaptive threshold optimization
- Federated learning for cross-organizational risk model training
- Additional statistical validation methods

## Code Standards

```bash
# Format
black . --max-line-length 100

# Lint
flake8 models/ services/ api/ evaluation/ utils/ --max-line-length 100

# Test
python -m pytest tests/ -v --cov=models --cov=services
```

## Author

Nikhil Reddy Donapati · [ORCID 0009-0006-7699-3928](https://orcid.org/0009-0006-7699-3928)
