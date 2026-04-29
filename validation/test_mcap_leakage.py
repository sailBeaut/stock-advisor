import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trainer, ranker_trainer

# 1. Both FEATURE_COLS lists must NOT include mcap_tier
assert 'mcap_tier' not in trainer.FEATURE_COLS, \
    f"mcap_tier still in trainer.FEATURE_COLS: {trainer.FEATURE_COLS}"
assert 'mcap_tier' not in ranker_trainer.FEATURE_COLS, \
    f"mcap_tier still in ranker_trainer.FEATURE_COLS: {ranker_trainer.FEATURE_COLS}"

# 2. feature_engine.py must NOT compute mcap_tier from stocks.market_cap
with open('feature_engine.py', 'r') as f:
    src = f.read()
# The leaky pattern is `pd.qcut(... market_cap ...)` building mcap_tier
leaky_pattern = re.compile(
    r'mc_df\s*\[\s*["\']mcap_tier["\']\s*\]\s*=\s*pd\.qcut',
    re.MULTILINE,
)
assert not leaky_pattern.search(src), \
    "feature_engine.py still contains pd.qcut on market_cap building mcap_tier"
assert 'mcap_tier removed Apr 2026' in src, \
    "feature_engine.py is missing the removal-rationale comment"

# 6. No mcap_tier references in any model-facing Python file.
#    Allowed only in: feature_engine.py (schema/SQL/removal comment),
#    validation/*.py (these checks), diagnose_prompts_1_to_6.py.
_BLACKLIST = [
    'trainer.py',
    'ranker_trainer.py',
    'predict.py',
    'bias_audit.py',
    'compare_models.py',
    'daily_pipeline.py',
]
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fails = []
for _fname in _BLACKLIST:
    _fpath = os.path.join(_root, _fname)
    if not os.path.exists(_fpath):
        continue
    with open(_fpath, 'r', encoding='utf-8') as _f:
        for _lineno, _line in enumerate(_f, 1):
            if 'mcap_tier' in _line:
                _fails.append(f"{_fname}:{_lineno}: {_line.rstrip()}")
assert not _fails, (
    "mcap_tier found in blacklisted file(s):\n" + "\n".join(_fails)
)

print('mcap leakage check: PASS')
