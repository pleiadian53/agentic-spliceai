# Contributing to Agentic-SpliceAI

## Documentation Structure

We follow a clear convention for organizing documentation:

### 1. Internal Development (`dev/`)

**Purpose**: Internal notes, scripts, and development-specific documentation  
**Visibility**: NOT pushed to GitHub (in `.gitignore`)  
**Audience**: Project developers only

**Structure**:
```
dev/
├── data/              # Data setup scripts and notes
├── experiments/       # Experimental code and results
├── notes/             # Development notes and ideas
└── scratch/           # Temporary work
```

**Examples**:
- `dev/notes/architecture_ideas.md` - Brainstorming notes
- `dev/experiments/new_feature_test.py` - Experimental code
- `dev/scratch/temp_analysis.ipynb` - Temporary notebooks

### 2. Public Documentation (`docs/`)

**Purpose**: Public-facing documentation, tutorials, and guides  
**Visibility**: Pushed to GitHub  
**Audience**: External users and contributors

**Structure**:
```
docs/
├── tutorials/         # Step-by-step guides
├── guides/            # How-to guides
├── architecture/      # System architecture docs
└── api/               # API documentation
```

**Examples**:
- `docs/tutorials/getting_started.md` - User onboarding
- `docs/guides/splice_site_analysis.md` - Feature guide
- `docs/architecture/multi_agent_system.md` - System design

### 3. Scripts (`scripts/`)

**Purpose**: Public automation scripts and utilities  
**Visibility**: Pushed to GitHub  
**Audience**: All users

**Structure**:
```
scripts/
├── deployment/        # Deployment automation
├── utils/             # Utility scripts
└── ci/                # CI/CD scripts
```

**Examples**:
- `scripts/deployment/deploy.sh` - Deployment automation
- `scripts/utils/format_code.sh` - Code formatting

### 4. Package Documentation (`<package>/docs/`)

**Purpose**: Package-specific documentation  
**Visibility**: Pushed to GitHub  
**Audience**: Users of specific packages/modules

**Structure**:
```
server/docs/           # Server package documentation
agentic_spliceai/docs/ # Core package documentation
```

**Examples**:
- `server/docs/API.md` - REST API documentation
- `server/docs/QUICKSTART.md` - Server setup guide

### 5. Data Documentation (`data/`)

**Purpose**: Dataset descriptions and usage  
**Visibility**: Pushed to GitHub  
**Audience**: Users working with datasets

**Examples**:
- `data/README.md` - Data structure and access guide

## When to Use Each

### Use `dev/` for:
- ✅ Development notes and brainstorming
- ✅ Experimental code
- ✅ Personal notes and TODOs
- ✅ Temporary files and scratch work
- ✅ Work-in-progress that's not ready to share

### Use `scripts/` for:
- ✅ Public automation scripts (deployment, CI/CD)
- ✅ Utility scripts for all users
- ✅ Build automation
- ✅ Code formatting and linting

### Use `tests/` for:
- ✅ Private development tests
- ✅ Setup scripts with local paths (e.g., `tests/data/`)
- ✅ Test data and fixtures
- ✅ Development-specific scripts
- ✅ Anything that shouldn't be shared publicly

### Use `docs/` for:
- ✅ User-facing tutorials
- ✅ Feature documentation
- ✅ Architecture and design docs
- ✅ Contributing guidelines
- ✅ API references

### Use `<package>/docs/` for:
- ✅ Package-specific guides
- ✅ Module API documentation
- ✅ Package quickstart guides

## File Organization Best Practices

### Naming Conventions

**Development files** (in `dev/`):
- Use descriptive names: `setup_data_symlinks.sh`
- Include dates for time-sensitive notes: `2025-11-26_meeting_notes.md`
- Use prefixes for categories: `exp_new_feature.py`, `note_architecture.md`

**Public documentation** (in `docs/`):
- Use clear, user-friendly names: `getting_started.md`
- Follow consistent naming: `tutorial_*.md`, `guide_*.md`
- Use lowercase with underscores: `api_reference.md`

### Documentation Standards

**All public docs should include**:
1. Clear title and purpose
2. Table of contents (for long docs)
3. Code examples with syntax highlighting
4. Links to related documentation
5. Last updated date (optional)

**Internal dev docs can be**:
- More informal and concise
- Work-in-progress
- Personal notes style

## Git Workflow

### What Gets Committed

**Always commit**:
- Public documentation (`docs/`, `<package>/docs/`)
- README files
- Contributing guidelines
- Data documentation (`data/README.md`)

**Never commit**:
- Internal development files (`dev/`)
- Personal notes
- Experimental code (unless ready for review)
- Temporary files

### `.gitignore` Configuration

The `dev/` and `tests/` directories are configured in `.gitignore`:

```gitignore
# Development directory (NOT for public)
dev/
dev/**/*

# Tests directory (NOT for public - development only)
tests/
tests/**/*
```

## Examples from This Project

### Private Test Scripts
- `tests/data/setup_data_symlinks.sh` - Data symlink setup (private paths)
- `tests/data/README.md` - Private script documentation

### Public Documentation
- `README.md` - Project overview
- `QUICKSTART.md` - Quick start guide
- `SETUP.md` - Environment setup
- `data/README.md` - Data structure guide

### Package Documentation
- `server/README.md` - Server package overview
- `server/QUICKSTART.md` - Server quick start

## Contributing Documentation

When adding new documentation:

1. **Determine visibility**: Internal or public?
2. **Choose location**: `dev/`, `docs/`, or `<package>/docs/`
3. **Follow conventions**: Use appropriate naming and structure
4. **Link related docs**: Cross-reference where helpful
5. **Keep it updated**: Update docs when code changes

## Questions?

If you're unsure where to put something:
- **Will external users need this?** → `docs/` or `<package>/docs/`
- **Is this a useful public script?** → `scripts/` or `scripts/<topic>/`
- **Is this a private script or test?** → `tests/`
- **Is this internal/temporary?** → `dev/`
- **Is this about datasets?** → `data/README.md`

When in doubt, start in `dev/` and promote to appropriate location when ready.
