"""
validation/diagnose_prompts_1_to_6.py

Diagnostic-only audit: verifies that Prompts 1-6 actually landed in the
codebase before trusting the Prompt 7 XGBRanker holdout result.

READ-ONLY against trading.db and all model files.
Does NOT modify any production code, model files, or the database.

Run from the project root:
    python validation/diagnose_prompts_1_to_6.py
"""

import inspect
import math
import os
import sqlite3
import sys
from pathlib import Path

# Force UTF-8 output so Unicode in log messages does not crash on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Resolve project root regardless of where this script is invoked from.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

_results: list[tuple[str, str, str]] = []  # (label, status, notes)


def _record(label: str, passed: bool, notes: str) -> None:
    status = "PASS" if passed else "FAIL"
    marker = "  PASS" if passed else "  FAIL"
    print(f"{marker}  {label} - {notes}")
    _results.append((label, status, notes))


def _warn(label: str, notes: str) -> None:
    print(f"  WARN  {label} - {notes}")
    _results.append((label, "WARN", notes))


# ---------------------------------------------------------------------------
# CHECK 1 -- Prompt 1 landed (.env + dotenv)
# ---------------------------------------------------------------------------

print("\n--- CHECK 1: .env + dotenv (Prompt 1) ---")
try:
    env_path  = _ROOT / ".env"
    _env_path = _ROOT / "_env"

    if not env_path.exists():
        _record("1. .env + dotenv (Prompt 1)", False, ".env file does not exist")
    elif _env_path.exists():
        _record(
            "1. .env + dotenv (Prompt 1)", False,
            "_env still exists - .env rename step did not run",
        )
    else:
        try:
            import dotenv
        except ImportError:
            _record("1. .env + dotenv (Prompt 1)", False, "python-dotenv not installed")
        else:
            dotenv.load_dotenv(str(env_path))
            fred_key = os.environ.get("FRED_API_KEY", "")
            if not fred_key:
                _record(
                    "1. .env + dotenv (Prompt 1)", False,
                    "FRED_API_KEY is empty or missing after load_dotenv()",
                )
            elif fred_key == "PLACEHOLDER":
                _record(
                    "1. .env + dotenv (Prompt 1)", False,
                    "FRED_API_KEY is still the literal string 'PLACEHOLDER'",
                )
            else:
                _record(
                    "1. .env + dotenv (Prompt 1)", True,
                    f"keys loaded; FRED_API_KEY non-empty ({len(fred_key)} chars)",
                )
except Exception as exc:
    _record("1. .env + dotenv (Prompt 1)", False, f"unexpected error: {exc}")


# ---------------------------------------------------------------------------
# CHECK 2 -- Prompt 2 landed (cost threshold + block bootstrap)
# ---------------------------------------------------------------------------

print("\n--- CHECK 2: Cost threshold + block bootstrap (Prompt 2) ---")
try:
    import backtest

    ok2     = True
    notes2: list[str] = []

    # ROUND_TRIP_COST
    if not hasattr(backtest, "ROUND_TRIP_COST"):
        ok2 = False
        notes2.append("ROUND_TRIP_COST attribute missing")
    elif backtest.ROUND_TRIP_COST < 0.010:
        ok2 = False
        notes2.append(f"ROUND_TRIP_COST={backtest.ROUND_TRIP_COST:.4f} < 0.010")
    else:
        notes2.append(f"ROUND_TRIP_COST={backtest.ROUND_TRIP_COST:.4f}")

    # NET_EDGE_PASS_THRESHOLD
    if not hasattr(backtest, "NET_EDGE_PASS_THRESHOLD"):
        ok2 = False
        notes2.append("NET_EDGE_PASS_THRESHOLD attribute missing")
    elif backtest.NET_EDGE_PASS_THRESHOLD < 0.015:
        ok2 = False
        notes2.append(f"NET_EDGE_PASS_THRESHOLD={backtest.NET_EDGE_PASS_THRESHOLD:.4f} < 0.015")
    else:
        notes2.append(f"NET_EDGE_PASS_THRESHOLD={backtest.NET_EDGE_PASS_THRESHOLD:.4f}")

    # _bootstrap_edge_ci_block function presence
    if not hasattr(backtest, "_bootstrap_edge_ci_block"):
        ok2 = False
        notes2.append("_bootstrap_edge_ci_block function not found in backtest")
    else:
        notes2.append("_bootstrap_edge_ci_block present")

    # backtest.run() wires buy_returns_by_date
    try:
        run_src = inspect.getsource(backtest.run)
        if "buy_returns_by_date" in run_src:
            notes2.append("backtest.run() reads buy_returns_by_date")
        else:
            ok2 = False
            notes2.append("backtest.run() does NOT reference buy_returns_by_date")
    except Exception as e:
        ok2 = False
        notes2.append(f"could not inspect backtest.run: {e}")

    # trainer._compare_returns passes buy_returns_by_date to backtest.run()
    try:
        import trainer as _trainer_mod
        method = getattr(_trainer_mod.UniversalStockModel, "_compare_returns", None)
        if method is None:
            ok2 = False
            notes2.append("trainer.UniversalStockModel._compare_returns not found")
        else:
            cmp_src = inspect.getsource(method)
            if "buy_returns_by_date" in cmp_src:
                notes2.append("trainer._compare_returns passes buy_returns_by_date")
            else:
                ok2 = False
                notes2.append("trainer._compare_returns does NOT pass buy_returns_by_date")
    except Exception as e:
        ok2 = False
        notes2.append(f"could not inspect trainer._compare_returns: {e}")

    _record("2. Cost + block bootstrap (Prompt 2)", ok2, "; ".join(notes2))

except Exception as exc:
    _record("2. Cost + block bootstrap (Prompt 2)", False, f"import/check failed: {exc}")


# ---------------------------------------------------------------------------
# CHECK 3 -- Prompt 3 landed (mcap_tier leak removed)
# ---------------------------------------------------------------------------

print("\n--- CHECK 3: mcap_tier leak removed (Prompt 3) ---")
try:
    import trainer        as _tr3
    import ranker_trainer as _rk3

    ok3     = True
    notes3: list[str] = []

    if "mcap_tier" in _tr3.FEATURE_COLS:
        ok3 = False
        notes3.append("mcap_tier still present in trainer.FEATURE_COLS")
    else:
        notes3.append("mcap_tier not in trainer.FEATURE_COLS")

    if "mcap_tier" in _rk3.FEATURE_COLS:
        ok3 = False
        notes3.append("mcap_tier still present in ranker_trainer.FEATURE_COLS")
    else:
        notes3.append("mcap_tier not in ranker_trainer.FEATURE_COLS")

    fe_path = _ROOT / "feature_engine.py"
    try:
        fe_text = fe_path.read_text(encoding="utf-8")
        if "mcap_tier removed" in fe_text:
            notes3.append("feature_engine.py contains 'mcap_tier removed' comment")
        else:
            ok3 = False
            notes3.append("feature_engine.py missing 'mcap_tier removed' comment")
    except Exception as e:
        ok3 = False
        notes3.append(f"could not read feature_engine.py: {e}")

    _record("3. mcap_tier leak removed (Prompt 3)", ok3, "; ".join(notes3))

except Exception as exc:
    _record("3. mcap_tier leak removed (Prompt 3)", False, f"import/check failed: {exc}")


# ---------------------------------------------------------------------------
# CHECK 4 -- Prompt 4 landed (vol-normalized labels)
# ---------------------------------------------------------------------------

print("\n--- CHECK 4: Vol-normalized labels (Prompt 4) ---")
try:
    import labeler

    ok4     = True
    notes4: list[str] = []

    flag = getattr(labeler, "USE_VOL_NORMALIZED_LABELS", None)
    if flag is not True:
        ok4 = False
        notes4.append(f"USE_VOL_NORMALIZED_LABELS={flag!r} (expected True)")
    else:
        notes4.append("USE_VOL_NORMALIZED_LABELS=True")

    if not hasattr(labeler, "_label_ticker_vol_normalized"):
        ok4 = False
        notes4.append("_label_ticker_vol_normalized function not found in labeler")
    else:
        notes4.append("_label_ticker_vol_normalized present")

    db_path = _ROOT / "trading.db"
    try:
        conn = sqlite3.connect(str(db_path))
        label_cols = {row[1] for row in conn.execute("PRAGMA table_info(labels)")}

        if "forward_zscore" not in label_cols:
            ok4 = False
            notes4.append("forward_zscore column missing from labels table")
        else:
            notes4.append("forward_zscore column exists in labels table")
            row = conn.execute(
                "SELECT COUNT(*) FROM labels WHERE forward_zscore IS NOT NULL"
            ).fetchone()
            count = row[0] if row else 0
            if count <= 1000:
                ok4 = False
                notes4.append(
                    f"only {count:,} non-null forward_zscore rows (need >1000; "
                    "labels may not have been regenerated after schema change)"
                )
            else:
                notes4.append(f"{count:,} rows with non-null forward_zscore")
        conn.close()
    except Exception as e:
        ok4 = False
        notes4.append(f"DB check failed: {e}")

    _record("4. Vol-normalized labels (Prompt 4)", ok4, "; ".join(notes4))

except Exception as exc:
    _record("4. Vol-normalized labels (Prompt 4)", False, f"import/check failed: {exc}")


# ---------------------------------------------------------------------------
# CHECK 5 -- Prompt 5 landed (sector-relative features)
# ---------------------------------------------------------------------------

print("\n--- CHECK 5: Sector-relative features (Prompt 5) ---")

_SECTOR_REL = [
    "rsi_14_vs_sector",
    "return_5d_vs_sector",
    "return_20d_vs_sector",
    "macd_hist_vs_sector",
    "vol_20d_vs_sector",
    "dist_sma50_vs_sector",
]

try:
    import trainer        as _tr5
    import ranker_trainer as _rk5

    ok5     = True
    notes5: list[str] = []

    db_path = _ROOT / "trading.db"
    conn5 = sqlite3.connect(str(db_path))

    # DB schema check
    db_feat_cols = {row[1] for row in conn5.execute("PRAGMA table_info(features)")}
    missing_db = [c for c in _SECTOR_REL if c not in db_feat_cols]
    if missing_db:
        ok5 = False
        notes5.append(f"missing from DB features table: {missing_db}")
    else:
        notes5.append("all 6 sector-relative cols in DB schema")

    # trainer.FEATURE_COLS check
    missing_tr = [c for c in _SECTOR_REL if c not in _tr5.FEATURE_COLS]
    if missing_tr:
        ok5 = False
        notes5.append(f"missing from trainer.FEATURE_COLS: {missing_tr}")
    else:
        notes5.append("all 6 in trainer.FEATURE_COLS")

    # ranker_trainer.FEATURE_COLS check
    missing_rk = [c for c in _SECTOR_REL if c not in _rk5.FEATURE_COLS]
    if missing_rk:
        ok5 = False
        notes5.append(f"missing from ranker_trainer.FEATURE_COLS: {missing_rk}")
    else:
        notes5.append("all 6 in ranker_trainer.FEATURE_COLS")

    # Fill-rate check: sector-relative columns must have been computed
    try:
        total_row = conn5.execute("SELECT COUNT(*) FROM features").fetchone()
        total = total_row[0] if total_row else 0
        if total == 0:
            ok5 = False
            notes5.append("features table is empty")
        else:
            nn_row = conn5.execute(
                "SELECT COUNT(*) FROM features WHERE rsi_14_vs_sector IS NOT NULL"
            ).fetchone()
            nn = nn_row[0] if nn_row else 0
            fill_pct = nn / total * 100
            if nn < total * 0.50:
                ok5 = False
                notes5.append(
                    f"rsi_14_vs_sector only {fill_pct:.1f}% non-null "
                    f"({nn:,}/{total:,}) - cross-section step likely never ran"
                )
            else:
                notes5.append(
                    f"rsi_14_vs_sector {fill_pct:.1f}% non-null ({nn:,}/{total:,})"
                )
    except Exception as e:
        ok5 = False
        notes5.append(f"fill-rate check failed: {e}")

    conn5.close()
    _record("5. Sector-relative features (Prompt 5)", ok5, "; ".join(notes5))

except Exception as exc:
    _record("5. Sector-relative features (Prompt 5)", False, f"check failed: {exc}")


# ---------------------------------------------------------------------------
# CHECK 6 -- Prompt 6 landed (persisted LabelEncoder)
# ---------------------------------------------------------------------------

print("\n--- CHECK 6: LabelEncoder persisted (Prompt 6) ---")
try:
    ok6     = True
    notes6: list[str] = []

    enc_path = _ROOT / "models" / "sector_encoder.joblib"
    if not enc_path.exists():
        ok6 = False
        notes6.append("models/sector_encoder.joblib not found")
    else:
        notes6.append("sector_encoder.joblib exists")
        try:
            import encoders
            enc = encoders.get_sector_encoder()
            if "Unknown" in enc.classes_:
                notes6.append("'Unknown' in encoder.classes_")
            else:
                ok6 = False
                notes6.append(f"'Unknown' NOT in encoder.classes_ ({list(enc.classes_[:3])}...)")
        except Exception as e:
            ok6 = False
            notes6.append(f"encoders.get_sector_encoder() failed: {e}")

    # feature_engine.py must NOT have a hard-coded SECTOR_ENCODING dict
    fe_path = _ROOT / "feature_engine.py"
    try:
        fe_text = fe_path.read_text(encoding="utf-8")
        if "SECTOR_ENCODING = {" in fe_text:
            ok6 = False
            notes6.append("feature_engine.py still contains SECTOR_ENCODING = { literal")
        else:
            notes6.append("no SECTOR_ENCODING dict literal in feature_engine.py")
    except Exception as e:
        ok6 = False
        notes6.append(f"could not read feature_engine.py: {e}")

    # Saved ranker payload must bundle sector_encoder_path
    rkr_path = _ROOT / "models" / "universal_ranker.joblib"
    if not rkr_path.exists():
        ok6 = False
        notes6.append("models/universal_ranker.joblib not found - cannot inspect payload")
    else:
        try:
            import joblib as _jl6
            payload = _jl6.load(str(rkr_path))
            if isinstance(payload, dict):
                if "sector_encoder_path" in payload:
                    notes6.append("ranker payload dict has 'sector_encoder_path' key")
                else:
                    ok6 = False
                    notes6.append(
                        f"ranker payload dict missing 'sector_encoder_path' "
                        f"(found keys: {list(payload.keys())})"
                    )
            elif hasattr(payload, "sector_encoder_path"):
                notes6.append("ranker object has sector_encoder_path attribute")
            else:
                ok6 = False
                notes6.append(
                    "ranker payload has no sector_encoder_path "
                    f"(type={type(payload).__name__})"
                )
        except Exception as e:
            ok6 = False
            notes6.append(f"could not load universal_ranker.joblib: {e}")

    _record("6. LabelEncoder persisted (Prompt 6)", ok6, "; ".join(notes6))

except Exception as exc:
    _record("6. LabelEncoder persisted (Prompt 6)", False, f"check failed: {exc}")


# ---------------------------------------------------------------------------
# CHECK 7 -- Holdout sanity (the actual leak hunt)
# ---------------------------------------------------------------------------

print("\n--- CHECK 7: Holdout sanity ---")

_check7_done = False
try:
    import numpy  as np
    import pandas as pd

    db_path  = _ROOT / "trading.db"
    rkr_path = _ROOT / "models" / "universal_ranker.joblib"
    clf_path = _ROOT / "models" / "universal_model.joblib"

    warn7  = False
    notes7: list[str] = []

    if not clf_path.exists():
        _record("7. Holdout sanity", False, "universal_model.joblib not found (need split_dates)")
        _check7_done = True
    elif not rkr_path.exists():
        _record("7. Holdout sanity", False, "universal_ranker.joblib not found")
        _check7_done = True
    else:
        from trainer        import UniversalStockModel, FEATURE_COLS as _FCOLS
        from ranker_trainer import UniversalRanker

        clf = UniversalStockModel.load()
        rkr = UniversalRanker.load()

        holdout_start = clf.split_dates.get("test_end")
        print(f"  Holdout boundary (test_end from classifier): {holdout_start}")

        # Load holdout exactly as compare_models.py does
        feat_select = ", ".join(f"f.{c}" for c in _FCOLS)
        sql = f"""
            SELECT f.ticker, f.date, {feat_select},
                   l.label, l.forward_return
            FROM   features f
            JOIN   labels   l ON l.ticker = f.ticker AND l.date = f.date
            WHERE  f.date > ?
            ORDER  BY f.date, f.ticker
        """
        conn7 = sqlite3.connect(str(db_path))
        conn7.row_factory = sqlite3.Row
        rows7 = conn7.execute(sql, (holdout_start,)).fetchall()
        conn7.close()

        df7 = pd.DataFrame([dict(r) for r in rows7])

        if df7.empty:
            _record("7. Holdout sanity", False, "holdout slice is empty after split boundary")
            _check7_done = True
        else:
            n_dates   = df7["date"].nunique()
            n_rows    = len(df7)
            n_tickers = df7["ticker"].nunique()
            date_min  = df7["date"].min()
            date_max  = df7["date"].max()

            print(f"  Holdout date range : {date_min} to {date_max}")
            print(f"  Unique dates       : {n_dates}")
            print(f"  Total rows         : {n_rows:,}")
            print(f"  Unique tickers     : {n_tickers}")
            notes7.append(
                f"{n_dates} dates, {n_rows:,} rows, {n_tickers} tickers "
                f"({date_min} to {date_max})"
            )

            if n_dates < 100:
                warn7 = True
                print(
                    f"\n  !!! WARNING: holdout spans only {n_dates} trading days "
                    "(need >= 100).\n"
                    "      A sub-100-day window in a trending market cannot distinguish\n"
                    "      skill from beta exposure -- results are unreliable."
                )
                notes7.append(f"WARN: only {n_dates} trading days (need >=100)")

            # Score holdout with ranker
            fwd7       = df7["forward_return"].values.astype(float)
            mask_valid = ~np.isnan(fwd7)
            df_v       = df7[mask_valid].copy()

            X7 = df_v[_FCOLS].values.astype(float)
            df_v = df_v.copy()
            df_v["score"] = rkr.model.predict(X7)
            df_v["rank"]  = (
                df_v.groupby("date")["score"]
                .rank(method="first", ascending=False)
            )

            # Per-date top-20 vs BAH -- build block-bootstrap dicts simultaneously
            buy_returns_by_date: dict = {}
            all_returns_by_date: dict = {}
            per_date_top20: list[float] = []
            per_date_bah:   list[float] = []

            for date, grp in df_v.groupby("date"):
                ret = grp["forward_return"].values.astype(float)
                top = grp.loc[grp["rank"] <= 20, "forward_return"].values.astype(float)
                all_returns_by_date[date] = ret.tolist()
                buy_returns_by_date[date] = top.tolist()
                if len(top) > 0:
                    per_date_top20.append(float(np.mean(top)))
                per_date_bah.append(float(np.mean(ret)))

            mean_top20 = float(np.mean(per_date_top20)) if per_date_top20 else float("nan")
            mean_bah   = float(np.mean(per_date_bah))   if per_date_bah   else float("nan")
            raw_edge   = mean_top20 - mean_bah

            per_date_edges = [t - b for t, b in zip(per_date_top20, per_date_bah)]
            edge_std       = float(np.std(per_date_edges)) if per_date_edges else float("nan")
            n_beat         = sum(e > 0 for e in per_date_edges)
            n_edge_obs     = len(per_date_edges)

            print(f"\n  mean(top-20 return) per date  : {mean_top20:+.4f}  ({mean_top20*100:+.2f}%)")
            print(f"  mean(BAH return) per date     : {mean_bah:+.4f}  ({mean_bah*100:+.2f}%)")
            print(f"  raw edge                      : {raw_edge:+.4f}  ({raw_edge*100:+.2f}%)")
            print(f"  std of per-date edge          : {edge_std:.4f}  ({edge_std*100:.2f}%)")
            print(f"  dates top-20 beat BAH         : {n_beat} / {n_edge_obs}")

            notes7.append(f"raw edge={raw_edge*100:+.2f}%")

            if not math.isnan(raw_edge) and raw_edge > 0.04:
                warn7 = True
                print(
                    f"\n  !!! WARNING: raw edge {raw_edge*100:+.2f}% exceeds 4% per 30-day cycle.\n"
                    "      This is implausibly large for retail XGBoost on free data.\n"
                    "      A figure this high is far more likely to reflect a data leak\n"
                    "      (e.g. mcap_tier computed on full-sample market caps, or sector-\n"
                    "      relative features computed before train/test split) than genuine\n"
                    "      predictive skill."
                )
                notes7.append("WARN: raw edge >4% -- implausible, likely data leak")

            # Block-bootstrap CI via backtest.run()
            import backtest as _bt7
            bt_result = _bt7.run({
                "buy_edge_gross":      mean_top20,
                "bah_return":          mean_bah,
                "buy_returns_by_date": buy_returns_by_date,
                "all_returns_by_date": all_returns_by_date,
            })
            ci_lower = bt_result.get("ci_lower", float("nan"))
            ci_upper = bt_result.get("ci_upper", float("nan"))
            print(f"\n  Block-bootstrap 95% CI        : [{ci_lower*100:+.2f}%, {ci_upper*100:+.2f}%]")
            notes7.append(f"CI=[{ci_lower*100:+.2f}%,{ci_upper*100:+.2f}%]")

            if not math.isnan(ci_lower) and not math.isnan(ci_upper):
                ci_width = ci_upper - ci_lower
                if ci_width < 0.03:
                    warn7 = True
                    print(
                        f"\n  !!! WARNING: CI width {ci_width*100:.2f}% is suspiciously tight.\n"
                        "      For a 6-month holdout the block-bootstrap CI should typically\n"
                        "      be wider than 3%. If the original Prompt 7 evaluation used the\n"
                        "      naive IID bootstrap (flat buy_returns / all_returns arrays only,\n"
                        "      as compare_models.py currently does), that would produce an\n"
                        "      artificially narrow CI by ignoring cross-sectional correlation."
                    )
                    notes7.append(f"WARN: CI width {ci_width*100:.2f}% -- suspiciously tight")

            # Note: compare_models.py does NOT pass block-bootstrap dicts
            print(
                "\n  NOTE: compare_models.py (the script that produced the reported\n"
                "  +5.85% / CI [+4.84%, +6.86%]) calls backtest.run() with flat\n"
                "  'buy_returns'/'all_returns' arrays only -- NOT the dict-style\n"
                "  'buy_returns_by_date'/'all_returns_by_date' inputs.\n"
                "  The reported CI therefore used the naive IID bootstrap, which\n"
                "  understates variance by ignoring same-date cross-sectional correlation."
            )

            _check7_done = True
            if warn7:
                _warn("7. Holdout sanity", "; ".join(notes7))
            else:
                _record("7. Holdout sanity", True, "; ".join(notes7))

except Exception as exc:
    if not _check7_done:
        _record("7. Holdout sanity", False, f"check failed: {exc}")


# ---------------------------------------------------------------------------
# Summary matrix
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print(f"  {'CHECK':<38}  {'STATUS':<8}  NOTES")
print("=" * 70)
for label, status, notes in _results:
    trunc = (notes[:42] + "...") if len(notes) > 43 else notes
    print(f"  {label:<38}  {status:<8}  {trunc}")
print("=" * 70)


# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

failed = [r for r in _results if r[1] == "FAIL"]

print()
if not failed:
    print(
        "ALL PROMPTS 1-6 LANDED -- Prompt 7 result may be trusted, "
        "proceed to Prompt 8."
    )
    warned = [r for r in _results if r[1] == "WARN"]
    if warned:
        print(
            f"  ({len(warned)} warning(s) raised -- review holdout sanity output "
            "above for details.)"
        )
else:
    print(
        "PROMPTS 1-6 INCOMPLETE -- Prompt 7 was run on a leaky/stale codebase.\n"
        "The +5.85% holdout edge is NOT TRUSTWORTHY. Re-run the failed prompts\n"
        "in dependency order, regenerate labels and features, retrain the\n"
        "ranker, then re-evaluate."
    )
    print(f"\n  Failed checks: {', '.join(r[0] for r in failed)}")
