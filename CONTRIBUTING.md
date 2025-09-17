# Contributing to TRader Panel

## Development Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd TRader-Panel-ASYNC
```

### 2. Set up Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### 4. Set up Pre-commit Hooks
```bash
# Automated setup
./setup-precommit.sh

# Or manual setup
pip install pre-commit
pre-commit install
```

## Code Quality Standards

### Pre-commit Hooks
The following hooks run automatically on every commit:

- **Black**: Code formatting (line length: 88)
- **isort**: Import sorting
- **flake8**: Linting and style checking
- **mypy**: Type checking
- **bandit**: Security scanning
- **pydocstyle**: Documentation style

### Running Hooks Manually
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black
pre-commit run mypy

# Update hook versions
pre-commit autoupdate
```

### Code Style Guidelines

#### Python Code Formatting
- Use **Black** for code formatting (88 character line limit)
- Use **isort** for import sorting
- Follow **Google docstring** convention

#### Type Hints
- Add type hints to all function signatures
- Use `Optional[Type]` for nullable parameters
- Import types from `typing` module

#### Documentation
- Write docstrings for all public functions and classes
- Use Google-style docstrings
- Include parameter and return type descriptions

#### Testing
- Write tests for new features
- Use `pytest` for testing
- Aim for >80% code coverage

## Git Workflow

### Branch Naming
- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Critical fixes
- `refactor/description` - Code refactoring

### Commit Messages
Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(signals): add supersede logic for duplicate signals
fix(database): resolve connection pooling issue
docs(readme): update installation instructions
```

### Pull Request Process
1. Create feature branch from `develop`
2. Make changes with proper tests
3. Ensure all pre-commit hooks pass
4. Create PR to `develop` branch
5. Wait for CI/CD pipeline to pass
6. Request code review

## CI/CD Pipeline

### GitHub Actions
The CI/CD pipeline runs on every push and PR:

1. **Lint and Test** (Python 3.11, 3.12)
   - Pre-commit hooks
   - Unit tests with coverage
   - Upload coverage reports

2. **Security Scan**
   - Bandit security analysis
   - Safety dependency check

3. **Type Check**
   - MyPy static type checking

### Local Testing
```bash
# Run tests
pytest tests/ -v --cov=.

# Run security scan
bandit -r . -f json

# Run type checking
mypy . --ignore-missing-imports
```

## Project Structure

```
TRader-Panel-ASYNC/
├── .github/workflows/     # GitHub Actions
├── database/             # Database models and operations
├── signals/              # Signal processing logic
├── utils/                # Utility functions
├── indicators/           # Technical indicators
├── tests/                # Test files
├── docs/                 # Documentation
├── .pre-commit-config.yaml
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
└── setup-precommit.sh   # Pre-commit setup script
```

## Getting Help

- Check existing issues and documentation
- Create detailed bug reports with reproduction steps
- Ask questions in discussions or issues
- Follow the code style and contribution guidelines

## License

This project is licensed under the MIT License - see the LICENSE file for details.
