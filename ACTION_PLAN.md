# simGL — Action Plan

Track progress here: mark items `[x]` when done, add notes under each section.

---

## Step 1 — Code review: `simGL.py` and docstrings ✅

- [x] Verify every public function is exported in `__init__.py`
- [x] Read through each function's docstring: parameter types/shapes, return values, notes, examples
- [x] Check private/helper functions have minimal docstrings (so AutoAPI doesn't produce blank entries)
- [x] Confirm `e=0` restriction is consistently documented everywhere `e` appears
- [x] Confirm `gm` biallelic requirement is documented everywhere `gm` appears
- [x] Add `Examples` sections to all public functions
- [x] Rename `check_*` validators to `_check_*` (private convention)
- [x] Fix `incorporate_monomorphic` dtype bug (was returning floats)
- [x] Add `ref_alt_to_index` and `GL_to_vcf` public functions
- [x] Remove all `*CLAUDE NOTE*` markers

---

## Step 2 — Sphinx docs review and improvement ✅

- [x] Browse rendered HTML page by page and note gaps
- [x] Fix Theory page math rendering (auto-handled by RTD theme via MathJax)
- [x] Hide private helpers from AutoAPI output (excluded via `_` prefix convention)
- [x] Check Quickstart example runs correctly end-to-end
- [x] Fill missing context in User Guide (coverage model, linked mode, monomorphic sites)
- [x] Review Installation page
- [x] Rewrite README.md; delete minutes.md
- [x] Add citation page to docs

---

## Step 3 — GitHub: push changes and handle forks ✅

- [x] Add `docs/_build/` to `.gitignore`
- [x] Stage and commit all changes with a descriptive message
- [x] Push to `main` on RacimoLab/simGL
- [x] Inspect `main` and `errordist` branches — both superseded by `simreads` rewrite
- [x] Kept `simreads` and `errordist` branches intact (ongoing discussion there)

---

## Step 4 — Publish documentation (Read the Docs) ← We are here

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
- [ ] Update `docs/citation.rst` with the Zenodo DOI once available
- [ ] (Optional) Submit a JOSS paper for a peer-reviewed citation

---

## Notes / decisions log

- `e=0` is valid for `sim_allelereadcounts` (simulation only) but invalid for `allelereadcounts_to_GL` (log formula). Separate validators `_check_e` and `_check_e_GL` enforce this.
- ANGSD Q30 default is `e = 0.001`.
- `gm` must be biallelic (values 0/1 only). msprime can produce multiallelic sites — filter with `gm_full.max(axis=1) == 1` before passing to simGL.
- conda environment for development: `reindeerGL`. Base environment must not be touched.
- Tests run from `/tmp` to avoid namespace package shadowing: `cd /tmp && python -m pytest /path/to/tests/`.
- `simreads` and `errordist` branches kept on origin (ongoing discussion); do not delete.
- AutoAPI shows `simGL.simGL.*` naming (package + module both named simGL) — accepted as-is.
