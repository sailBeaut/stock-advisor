"""
PWA validation: checks all required files exist and contain expected content.
Run from the project root:  python validation/test_pwa.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
WEB  = ROOT / 'app' / 'web'

PASS = True

def check(label, ok, detail=''):
    global PASS
    status = 'OK  ' if ok else 'FAIL'
    if not ok:
        PASS = False
    suffix = f'  ({detail})' if detail and not ok else ''
    print(f'  [{status}] {label}{suffix}')
    return ok


print('\n=== PWA Validation ===\n')

# ── 1. All 7 files exist ──────────────────────────────────────
print('Files:')
required_files = [
    'index.html',
    'styles.css',
    'app.js',
    'manifest.webmanifest',
    'sw.js',
    'icon-192.png',
    'icon-512.png',
]
for fname in required_files:
    p = WEB / fname
    check(fname, p.exists(), f'not found at {p}')

# ── 2. index.html content ─────────────────────────────────────
print('\nindex.html:')
html = (WEB / 'index.html').read_text(encoding='utf-8')
check('has <link rel="manifest">', 'rel="manifest"' in html)
check('references app.js',        'app.js' in html)
check('has viewport-fit=cover',   'viewport-fit=cover' in html)
check('has apple-mobile-web-app-capable', 'apple-mobile-web-app-capable' in html)
check('has 4 tab buttons (data-tab)',     html.count('data-tab="') >= 4)

# ── 3. manifest.webmanifest is valid JSON ─────────────────────
print('\nmanifest.webmanifest:')
try:
    manifest = json.loads((WEB / 'manifest.webmanifest').read_text(encoding='utf-8'))
    check('valid JSON', True)
    for field in ('name', 'short_name', 'start_url', 'display', 'icons'):
        check(f'has field: {field}', field in manifest)
    check('icons is a list', isinstance(manifest.get('icons'), list))
except json.JSONDecodeError as e:
    check('valid JSON', False, str(e))

# ── 4. app.js contains API_BASE ───────────────────────────────
print('\napp.js:')
app_js = (WEB / 'app.js').read_text(encoding='utf-8')
check('contains API_BASE',                'API_BASE' in app_js)
check('contains localStorage (api key)',  'localStorage' in app_js)
check('contains fmtUSD',                 'fmtUSD' in app_js)
check('contains fmtPct',                 'fmtPct' in app_js)
check('contains fmtCompact',             'fmtCompact' in app_js)
check('contains serviceWorker.register', 'serviceWorker.register' in app_js)

# ── 5. sw.js has install handler ─────────────────────────────
print('\nsw.js:')
sw = (WEB / 'sw.js').read_text(encoding='utf-8')
check('has install event listener',  "addEventListener('install'" in sw)
check('has fetch event listener',    "addEventListener('fetch'"   in sw)
check('defines CACHE constant',      'CACHE' in sw)

# ── 6. Icons are valid PNGs of correct size ───────────────────
print('\nIcons (Pillow):')
try:
    from PIL import Image
    for size in [192, 512]:
        p = WEB / f'icon-{size}.png'
        if p.exists():
            with Image.open(p) as img:
                check(f'icon-{size}.png is {size}x{size}', img.size == (size, size),
                      f'actual size: {img.size}')
                check(f'icon-{size}.png mode is RGB(A)', img.mode in ('RGB', 'RGBA'))
        else:
            check(f'icon-{size}.png exists', False)
except ImportError:
    print('  [SKIP] Pillow not installed — cannot validate icons')

# ── Result ────────────────────────────────────────────────────
print()
if PASS:
    print('PASS')
else:
    print('FAIL')
    sys.exit(1)
