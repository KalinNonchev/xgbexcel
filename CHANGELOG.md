# Changelog

## 1.0.0

Support for XGBoost 2.x and 3.x. On those versions the previous release did not fail,
it returned formulas that computed the wrong number, so upgrading changes results.

### Correctness

- Splits now compare with strict `<`, matching XGBoost. They previously used `<=`,
  which sent every row sitting exactly on a threshold down the wrong branch.
- Thresholds are emitted as the midpoint between the float32 threshold and the float32
  below it, so a float64 comparison in Excel reproduces XGBoost's float32 comparison.
  Reading the text dump instead had rows taking the wrong branch, because XGBoost picks
  thresholds that are training values and the dump prints them at reduced precision.
- `base_score` is read from the serialised model rather than the estimator attribute,
  which is `None` on XGBoost 2.0 and later. The old fallback to 0.5 shifted every
  regression prediction.
- An explicit `base_score=0.0` is no longer replaced with 0.5.
- Multiclass models use the per-class `base_score` vector that XGBoost 3.x stores,
  instead of adding a single 0.5 to every class.
- Binary classification applies a sigmoid to one tree per boosting round. It was
  previously treated as two interleaved groups of trees fed through a softmax, so the
  output was not a probability.
- `base_score` is converted from output space to margin space per objective: the logit
  for logistic objectives, the log for Poisson, gamma and Tweedie.
- Early stopping is respected. `predict` stops at `best_iteration` while the model
  still holds every tree, so the extra rounds used to be included.
- Vector leaves from `multi_strategy="multi_output_tree"` are supported.
- Multi-output regression, `num_parallel_tree` forest mode, and dart with dropped trees
  are supported. Output grouping reads the model's own `tree_info`.
- `rename_features` matches whole feature names and substitutes all of them at once.
  It previously used `str.replace`, so renaming `x1` also rewrote `x10`, and a new name
  could be renamed again by a later entry in the mapping.
- The `sep` argument reaches nested nodes. It was accepted and then ignored.

### Behaviour

- Unsupported objectives, `gblinear`, and categorical splits raise
  `UnsupportedModelError` instead of producing a formula that is quietly wrong.
- A warning is emitted when a formula exceeds Excel's 8192 character limit.
- The constructor no longer calls `sys.setrecursionlimit(10000)`. Trees are walked
  iteratively, so depth cannot exhaust the stack.
- Formulas are assembled in linear time and no longer nest one bracket level per tree.
- `feature_names` can be passed to the constructor, and names on the booster are used
  when present.
- `rename_features` returns `self` so calls chain.
- `save_expr` writes one line per output.
- New: `expressions`, `n_outputs`, `n_features`, `n_trees_used`, `base_scores`,
  `objective`, `link`, and a useful `__repr__`.

### Packaging

- `pyproject.toml` replaces `setup.py` and `setup.cfg`.
- `requirements.txt` is gone. It pinned `xgboost==1.7.1` and listed `xgbexcel` itself,
  so CI tested the published package rather than the working tree.
- Ships `py.typed`; the public API is annotated.
- Requires Python 3.9 or later.
- Publishing uses PyPI Trusted Publishing rather than a stored API token.
- CI runs on Python 3.9 to 3.13 across Linux, macOS and Windows, against XGBoost 1.7
  through the current pre-release, with coverage reporting and a weekly scheduled run.

## 0.1.0

Initial release.
