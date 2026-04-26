"""
Validation test for the persisted sector LabelEncoder.

Requires trading.db to exist (stocks table must be populated).
Run after data_collector.py has been executed at least once.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders import (
    SECTOR_ENCODER_PATH,
    encode_sector,
    fit_sector_encoder,
    get_sector_encoder,
)

# ---------------------------------------------------------------------------
# 1. Encoder file exists OR fit_sector_encoder() successfully creates it
# ---------------------------------------------------------------------------
if not SECTOR_ENCODER_PATH.exists():
    enc = fit_sector_encoder()
    if not SECTOR_ENCODER_PATH.exists():
        print(f"FAIL: fit_sector_encoder() did not create {SECTOR_ENCODER_PATH}")
        sys.exit(1)
else:
    enc = get_sector_encoder()

# ---------------------------------------------------------------------------
# 2. Known sector encodes to a valid int
# ---------------------------------------------------------------------------
it_code = encode_sector("Information Technology", encoder=enc)
if not isinstance(it_code, int):
    print(f"FAIL: encode_sector('Information Technology') returned {type(it_code)}, expected int")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Unknown sector falls back to the 'Unknown' class integer
# ---------------------------------------------------------------------------
expected_unknown = int(enc.transform(["Unknown"])[0])
got_unknown      = encode_sector("Made Up Sector", encoder=enc)
if got_unknown != expected_unknown:
    print(
        f"FAIL: encode_sector('Made Up Sector') returned {got_unknown}, "
        f"expected Unknown={expected_unknown}"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Idempotent — two separate get_sector_encoder() calls give same result
# ---------------------------------------------------------------------------
enc2  = get_sector_encoder()
code1 = encode_sector("Information Technology", encoder=enc)
code2 = encode_sector("Information Technology", encoder=enc2)
if code1 != code2:
    print(
        f"FAIL: encode_sector not idempotent across two encoder loads "
        f"(got {code1} vs {code2})"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
print("PASS")
