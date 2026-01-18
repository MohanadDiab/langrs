# Release Checklist

Use this checklist before releasing a new version of LangRS.

## Pre-Release

### Code Quality
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] No linter errors (`read_lints` or `mypy langrs/`)
- [ ] Code follows style guidelines (black, isort)
- [ ] All TODOs addressed or documented
- [ ] No debug code or print statements

### Documentation
- [ ] README.md is up to date
- [ ] CHANGELOG.md is updated
- [ ] Migration guide is current (if breaking changes)
- [ ] Examples are tested and working
- [ ] API documentation is complete

### Testing
- [ ] Unit tests pass (100% coverage for new code)
- [ ] Integration tests pass
- [ ] Examples run without errors
- [ ] Tested on Python 3.10, 3.11, 3.12
- [ ] Tested on both CPU and GPU (if applicable)

### Dependencies
- [ ] `requirements.txt` is up to date
- [ ] `setup.py` dependencies match `requirements.txt`
- [ ] All dependencies have version pins (or ranges)
- [ ] No conflicting dependencies

### Version Management
- [ ] Version bumped in `setup.py`
- [ ] Version bumped in `__init__.py` (if applicable)
- [ ] CHANGELOG.md updated with version number
- [ ] Git tag created (if applicable)

## Release Process

### 1. Final Checks
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Run examples: `python examples/basic_usage.py`
- [ ] Check imports: `python -c "from langrs import create_pipeline; print('OK')"`
- [ ] Verify no sensitive data in code

### 2. Build Package
- [ ] Clean build directories: `rm -rf build/ dist/ *.egg-info`
- [ ] Build source distribution: `python setup.py sdist`
- [ ] Build wheel: `python setup.py bdist_wheel`
- [ ] Test installation: `pip install dist/langrs-*.whl`

### 3. Documentation
- [ ] Update version in README if needed
- [ ] Update CHANGELOG with release date
- [ ] Create release notes (if GitHub release)

### 4. Git Operations
- [ ] All changes committed
- [ ] Commit message follows conventions
- [ ] Create release branch (if needed)
- [ ] Tag release: `git tag -a v2.0.0 -m "Release v2.0.0"`
- [ ] Push tags: `git push --tags`

### 5. PyPI Release (if applicable)
- [ ] Test upload to TestPyPI first
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Verify package on PyPI

### 6. Post-Release
- [ ] Update documentation site (if applicable)
- [ ] Announce release (if applicable)
- [ ] Monitor for issues
- [ ] Update roadmap/backlog

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (2.0.0): Breaking changes
- **MINOR** (2.1.0): New features, backward compatible
- **PATCH** (2.0.1): Bug fixes, backward compatible

## Current Release: v2.0.0

### Major Changes
- Complete refactoring with modern architecture
- Removed dependency on `samgeo`
- New pipeline API with dependency injection
- Comprehensive test suite
- Full documentation

### Breaking Changes
- Old `LangRS` API still supported but new API recommended
- Model loading now explicit (`load_weights()`)
- Configuration system changed

### Migration
- See `MIGRATION_GUIDE.md` for details

---

**Last Updated**: Phase 6 completion
**Next Review**: Before v2.1.0 release
