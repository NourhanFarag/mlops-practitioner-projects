# mlops-practitioner-projects

Production-oriented MLOps mini projects built throughout the MLOps Practitioner course.

This repository contains the Mini Project implementations only. Course session materials
are kept separately and are used as references while building the projects.

Each Mini Project builds on the previous one through Git branches.

```text
mini-project-1
      ↓
mini-project-2
      ↓
mini-project-3
      ↓
...
```

The current branch contains Mini Project 1.

---

## Project Structure

```text
mlops-practitioner-projects/
├── data/                  # Local datasets (not committed)
├── models/                # Saved model artifacts
├── notebooks/             # Baseline and exploration notebooks
├── reports/               # Mini Project reports
├── src/
│   └── prodml/            # Production Python package
├── tests/                 # Project tests
├── pyproject.toml         # Package metadata and dependencies
├── .pre-commit-config.yaml
├── .gitignore
└── README.md
```

The production code lives under:

```text
src/prodml/
```

while the original exploratory baseline remains in:

```text
notebooks/
```

---

# Mini Project 1 — Production ML Foundation

Mini Project 1 starts from a notebook baseline and progressively turns it into a
production-style Python package.

The current model predicts NYC green taxi trip duration.

## Baseline

The baseline notebook is located at:

```text
notebooks/00-baseline.ipynb
```

The model uses:

```text
Features:
- PU_DO
- trip_distance

Target:
- duration in minutes

Model:
- DictVectorizer
- LinearRegression
```

The notebook baseline achieved:

```text
Validation MAE:  4.22 minutes
Validation RMSE: 6.51 minutes
```

The original notebook artifact is stored as:

```text
models/baseline.pkl
```

The refactored Python package produces:

```text
models/model.pkl
```

The refactored training workflow reproduces the notebook baseline within the
required tolerance.

---

# Development Workflow

## 1. Create and activate the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 2. Install the project

Standard Python installation:

```bash
pip install -e ".[dev]"
```

Alternative using `uv`:

```bash
uv pip install -e ".[dev]"
```

The package is installed in editable mode, which means changes made inside
`src/prodml/` are immediately available without reinstalling the project.

Verify the installation:

```bash
python -c "import prodml; print(prodml.__file__)"
```

The returned path should point to:

```text
src/prodml/__init__.py
```

inside this repository.

---

## 3. Lint and format

Run Ruff and Black:

```bash
ruff check src tests && black --check src tests
```

Ruff checks for Python code-quality problems such as unused imports and common
mistakes.

Black verifies that Python source files use consistent formatting.

---

## 4. Test

Run the project test suite with coverage:

```bash
pytest -v --cov=src/prodml --cov-report=term-missing
```

The full automated test suite is introduced later in Mini Project 1.

---

## 5. Train

Train the model with:

```bash
python -m prodml.train
```

The installed CLI entry point can also be used:

```bash
prodml-train
```

The training workflow is:

```text
load Parquet data
        ↓
prepare and clean data
        ↓
create features
        ↓
train / validation split
        ↓
fit DictVectorizer
        ↓
train LinearRegression
        ↓
evaluate MAE and RMSE
        ↓
save models/model.pkl
```

Expected validation metrics:

```text
Validation MAE:  4.22 minutes
Validation RMSE: 6.51 minutes
```

---

## 6. Serve

The FastAPI service will be started with:

```bash
uvicorn prodml.api.main:app --reload --port 8000
```

The API implementation is added later in Mini Project 1.

---

# Python Package Structure

The original notebook responsibilities are separated into production modules:

```text
src/prodml/
├── __init__.py
├── config.py
├── data.py
├── features.py
├── train.py
├── predict.py
└── api/
    └── __init__.py
```

Each module has a focused responsibility.

### `config.py`

Contains project configuration such as:

```text
dataset path
model path
validation size
random seed
cleaning thresholds
API port
```

Configuration is managed using `pydantic-settings`.

Environment variables use the `PRODML_` prefix.

Example:

```bash
PRODML_API_PORT=9000 python -c "from prodml.config import settings; print(settings.api_port)"
```

Another example:

```bash
PRODML_RANDOM_STATE=123 python -c "from prodml.config import settings; print(settings.random_state)"
```

---

### `data.py`

Responsible for:

```text
loading Parquet data
train / validation splitting
```

---

### `features.py`

Responsible for:

```text
calculating trip duration
cleaning duration
cleaning trip distance
creating PU_DO
preparing model features
```

---

### `train.py`

Responsible for:

```text
loading prepared data
vectorization
model training
evaluation
model persistence
```

Running:

```bash
python -m prodml.train
```

creates:

```text
models/model.pkl
```

---

### `predict.py`

Contains the prediction interface:

```text
DurationPredictor.load()
DurationPredictor.predict_one()
DurationPredictor.predict_batch()
```

It also contains the custom:

```text
@timed
```

decorator used to measure prediction execution time.

This prediction interface provides a stable boundary that later components such
as the API and optimized model runtime can use.

---

# Model Artifacts

The original notebook artifact is:

```text
models/baseline.pkl
```

The production package artifact is:

```text
models/model.pkl
```

The artifact currently contains:

```text
DictVectorizer
LinearRegression
```

Keeping both objects together ensures that prediction uses the same fitted
feature vocabulary used during training.

> Pickle files must only be loaded from trusted sources because deserializing
> untrusted Pickle files can execute arbitrary Python code.

---

# Data

The project currently uses the NYC TLC green taxi dataset.

Local data is stored under:

```text
data/
```

Dataset files are not committed to Git.

The current training file is:

```text
data/green_tripdata_2026-01.parquet
```

Training-data cleaning currently keeps:

```text
1 <= duration <= 60 minutes

0 < trip_distance <= 50 miles
```

---

# Pre-commit Hooks

The repository uses `pre-commit` to run automated quality checks before commits.

The configuration is stored in:

```text
.pre-commit-config.yaml
```

Install the Git hook once after cloning the repository:

```bash
pre-commit install
```

After installation, the configured checks run automatically when:

```bash
git commit
```

If a hook changes a file, the commit is stopped so the modification can be
reviewed and staged again.

---

## Run pre-commit manually

Run all configured hooks across the repository:

```bash
pre-commit run --all-files
```

Because this repository contains only the Mini Project code, running across all
files is safe and does not affect unrelated course-session folders.

---

# Pre-commit Checks

## Ruff

Ruff is the Python linter.

It checks for issues such as:

```text
unused imports
incorrect import ordering
common Python mistakes
code-quality problems
```

The hook is configured with automatic safe fixes.

---

## Black

Black is the Python formatter.

It keeps Python source code consistently formatted across the project.

---

## Repository Hygiene

The pre-commit configuration also performs checks for:

```text
trailing whitespace
missing newline at end of file
mixed line endings
invalid YAML
invalid TOML
invalid JSON
merge-conflict markers
large files
invalid Python syntax
debug statements
executable files without shebangs
private keys
```

---

# Useful Commands

Install:

```bash
pip install -e ".[dev]"
```

or:

```bash
uv pip install -e ".[dev]"
```

Lint:

```bash
ruff check src tests && black --check src tests
```

Test:

```bash
pytest -v --cov=src/prodml --cov-report=term-missing
```

Train:

```bash
python -m prodml.train
```

Train through the installed CLI:

```bash
prodml-train
```

Serve:

```bash
uvicorn prodml.api.main:app --reload --port 8000
```

Run pre-commit manually:

```bash
pre-commit run --all-files
```

---

# Current Mini Project 1 Status

Completed:

```text
Step 1 — Establish the baseline
Step 2 — Turn it into a real Python package
```

Current work:

```text
Step 3 — Structured logging
```

Upcoming work progressively adds:

```text
structured logging
serialization
FastAPI serving
automated testing
containerization
documentation and release
```

---

# Before Committing

Run:

```bash
ruff check src tests
black --check src tests
python -m prodml.train
```

The training workflow should continue to produce approximately:

```text
Validation MAE:  4.22 minutes
Validation RMSE: 6.51 minutes
```

Then review the changes:

```bash
git status
git diff
```

and commit when everything is correct.
