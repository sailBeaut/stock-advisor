import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import database, backtest
from trainer import UniversalStockModel, FEATURE_COLS, LABEL_BUY
from ranker_trainer import UniversalRanker

# ---- load both models ----
clf = UniversalStockModel.load()
rkr = UniversalRanker.load()

# ---- load holdout slice from DB ----
# split_dates was saved by the classifier; reuse it for fairness
split_dates = clf.split_dates
holdout_start = split_dates.get('test_end')
print(f'Using holdout start: {holdout_start}')

feat_select = ', '.join(f'f.{c}' for c in FEATURE_COLS)
sql = f'''
    SELECT f.ticker, f.date, {feat_select},
           l.label, l.forward_return
    FROM features f
    JOIN labels l ON l.ticker = f.ticker AND l.date = f.date
    WHERE f.date > ?
    ORDER BY f.date, f.ticker
'''
with database.connection() as conn:
    rows = conn.execute(sql, (holdout_start,)).fetchall()
df = pd.DataFrame([dict(r) for r in rows])
print(f'Holdout rows: {len(df)}')
if df.empty:
    print('No holdout data — abort.'); sys.exit(1)

fwd = df['forward_return'].values.astype(float)
mask_valid = ~np.isnan(fwd)

# ---- CLASSIFIER on holdout ----
proba_clf = clf.predict_proba(df)
preds_clf = UniversalStockModel._apply_buy_percentile(proba_clf, clf.buy_top_fraction)
buy_mask_clf = (preds_clf == LABEL_BUY) & mask_valid
clf_buy_ret = fwd[buy_mask_clf].mean() if buy_mask_clf.any() else float('nan')
clf_bah_ret = fwd[mask_valid].mean()

print()
print('=' * 70); print('  CLASSIFIER — HOLDOUT'); print('=' * 70)
print(f'  BUY positions      : {buy_mask_clf.sum()}')
print(f'  BUY avg 30d return : {clf_buy_ret:+.4f}')
print(f'  Buy-and-hold       : {clf_bah_ret:+.4f}')
print(f'  Edge               : {clf_buy_ret - clf_bah_ret:+.4f}')
backtest.run({
    'buy_edge_gross': clf_buy_ret,
    'bah_return':     clf_bah_ret,
    'buy_returns':    fwd[buy_mask_clf].tolist(),
    'all_returns':    fwd[mask_valid].tolist(),
})

# ---- RANKER on holdout ----
# top_n same as ranker config (e.g. top 20 per date)
TOP_N = 20
df_h = df.copy()
df_h['score'] = rkr.model.predict(df_h[FEATURE_COLS].values.astype(float))
df_h['rank'] = df_h.groupby('date')['score'].rank(method='first', ascending=False)
buy_mask_rkr = (df_h['rank'] <= TOP_N).values & mask_valid
rkr_buy_ret = fwd[buy_mask_rkr].mean() if buy_mask_rkr.any() else float('nan')

print()
print('=' * 70); print('  RANKER — HOLDOUT'); print('=' * 70)
print(f'  BUY positions      : {buy_mask_rkr.sum()}')
print(f'  Top-{TOP_N} avg 30d ret : {rkr_buy_ret:+.4f}')
print(f'  Buy-and-hold       : {clf_bah_ret:+.4f}')
print(f'  Edge               : {rkr_buy_ret - clf_bah_ret:+.4f}')
backtest.run({
    'buy_edge_gross': rkr_buy_ret,
    'bah_return':     clf_bah_ret,
    'buy_returns':    fwd[buy_mask_rkr].tolist(),
    'all_returns':    fwd[mask_valid].tolist(),
})

print()
print('=' * 70); print('  DECISION'); print('=' * 70)
print(f'  Classifier edge: {clf_buy_ret - clf_bah_ret:+.4f}')
print(f'  Ranker edge    : {rkr_buy_ret - clf_bah_ret:+.4f}')
winner = 'RANKER' if rkr_buy_ret > clf_buy_ret else 'CLASSIFIER'
print(f'  Winner         : {winner}')
