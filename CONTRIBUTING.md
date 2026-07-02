# Contributing to BathDiff

Thanks for your interest in improving BathDiff! 🌊

## 🛠️ Development setup

```bash
git clone https://github.com/<your-username>/bathymetric-diffusion.git
cd bathymetric-diffusion
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## 🧭 Workflow

1. **Fork & branch** — create a feature branch off `main`:
   ```bash
   git checkout -b feat/my-new-feature
   ```
2. **Write code** — follow the existing style (Black + Ruff, line-length 100).
3. **Write tests** — anything in `src/bathdiff/` that has logic should have a test in `tests/`.
4. **Run tests locally**:
   ```bash
   pytest tests/ -v
   ruff check src/ tests/
   black --check src/ tests/
   ```
5. **Open a PR** — describe what you changed and why. Link any related issue.

## 🧪 Testing guidelines

- Smoke tests should run on **CPU in under 60 seconds** (use tiny synthetic grids).
- Don't write tests that require GPU or real bathymetry data — those belong in `examples/`.
- Mock `jax` devices if you need to test device-specific behavior.

## 🏗️ Architecture notes

- `src/bathdiff/` is the importable library.
- `scripts/` is for end-user entry points only — no business logic there.
- All paths must be `pathlib.Path` objects, never raw strings.
- All randomness flows through `jax.random.PRNGKey` — never `numpy.random.seed`.
- Keep functions pure where possible (inputs in, outputs out, no globals).

## 📝 Commit message conventions

We follow Conventional Commits:

```
feat:     new feature
fix:      bug fix
docs:     documentation only
refactor: code change that neither fixes a bug nor adds a feature
test:     adding missing tests or correcting existing ones
chore:    build / tooling changes
```

Example: `feat: add canal body-type with elongated padding strategy`

## 🐛 Reporting bugs

Open an issue with:

- BathDiff version (`python -c "import bathdiff; print(bathdiff.__version__)"`)
- JAX / Flax versions
- OS & GPU info
- Minimal reproducible example
- Full traceback

## 💡 Suggesting enhancements

Open an issue with the `enhancement` label. Describe:

- The problem you're trying to solve
- The proposed API / CLI surface
- Any prior art (papers, repos, etc.)

## 📜 Code of conduct

Be kind. Be specific. Assume good intent. Credit others' work.
