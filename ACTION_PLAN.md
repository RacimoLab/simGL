# simGL — Action Plan

Track progress here: mark items `[x]` when done, add notes under each section.

---

## Step 1 — Code review: `simGL.py` and docstrings

- [ ] Verify every public function is exported in `__init__.py`
- [ ] Read through each function's docstring: parameter types/shapes, return values, notes, examples
- [ ] Check private/helper functions have minimal docstrings (so AutoAPI doesn't produce blank entries)
- [ ] Confirm `e=0` restriction is consistently documented everywhere `e` appears
- [ ] Confirm `gm` biallelic requirement is documented everywhere `gm` appears

---

## Step 2 — Sphinx docs review and improvement

- [ ] Browse rendered HTML page by page and note gaps
- [ ] Fix Theory page math rendering
- [ ] Decide whether to hide private helpers from AutoAPI output
- [ ] Check Quickstart example runs correctly end-to-end
- [ ] Fill any missing context in User Guide (e.g. `GL_to_Mm` polymorphic-only requirement)
- [ ] Review Installation page (Python version, optional dependencies)

---

## Step 3 — GitHub: push changes and handle forks

- [ ] Add `.gitignore` (exclude `docs/_build/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.egg-info/`)
- [ ] Stage and commit all changes with a descriptive message
- [ ] Push to `main` on RacimoLab/simGL
- [ ] Inspect the two forked repos to see if they contain unique commits
- [ ] If forks contain nothing new: delete them
- [ ] If forks contain useful changes: open PR → merge → delete fork

---

## Step 4 — Publish documentation (Read the Docs)

- [ ] Create account at readthedocs.org (sign in with GitHub)
- [ ] Import `RacimoLab/simGL` project (RTD detects `.readthedocs.yaml` automatically)
- [ ] Trigger first build and verify it succeeds
- [ ] Set `simgl.readthedocs.io` as canonical URL in RTD dashboard
- [ ] Add RTD badge to README

_Note: `.readthedocs.yaml` and `docs/` are already configured — no code changes needed._

---

## Step 5a — Publish to PyPI

- [ ] Create PyPI account at pypi.org
- [ ] Install build tools: `pip install build twine`
- [ ] Build distribution: `python -m build`
- [ ] Upload: `twine upload dist/*`
- [ ] Verify: `pip install simGL` works from a clean environment
- [ ] (Optional) Set up GitHub Actions workflow for automated PyPI publish on version tag

---

## Step 5b — Publish to conda-forge

_Prerequisite: package must be live on PyPI first._

- [ ] Fork `conda-forge/staged-recipes` on GitHub
- [ ] Add `recipes/simGL/meta.yaml` (can be generated from PyPI metadata)
- [ ] Open PR to `staged-recipes`; wait for conda-forge bot review
- [ ] After merge, `conda install -c conda-forge simGL` will work
- [ ] (Optional) Set up conda-forge autotick bot for future releases

---

## Step 6 — Make the package citable

- [ ] Write `CITATION.cff` in repo root (GitHub renders a "Cite this repository" button)
- [ ] Connect repo to Zenodo (zenodo.org → GitHub → flip switch for simGL)
- [ ] Push a tagged release (`git tag v0.2.0`) → Zenodo auto-mints a DOI
- [ ] Add Zenodo DOI badge to README
- [ ] (Optional) Submit a JOSS paper for a peer-reviewed citation

---

## Notes / decisions log

- `e=0` is valid for `sim_allelereadcounts` (simulation only) but invalid for `allelereadcounts_to_GL` (log formula). Separate validators `check_e` and `check_e_GL` enforce this.
- ANGSD Q30 default is `e = 0.001`.
- `gm` must be biallelic (values 0/1 only). msprime can produce multiallelic sites — filter with `gm_full.max(axis=1) == 1` before passing to simGL.
- conda environment for development: `reindeerGL`. Base environment must not be touched.
- Tests run from `/tmp` to avoid namespace package shadowing: `cd /tmp && python -m pytest /path/to/tests/`.
