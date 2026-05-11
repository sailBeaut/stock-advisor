// ============================================================
// Configuration
// ============================================================
const API_BASE = (window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost')
  ? 'http://127.0.0.1:8000'
  : 'https://stock-advisor-api.onrender.com';
console.log('[app] API_BASE =', API_BASE);

function getApiKey() { return localStorage.getItem('app_api_key') || ''; }

function apiFetch(path, opts = {}) {
  const key = getApiKey();
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60_000);
  return fetch(`${API_BASE}${path}`, {
    ...opts,
    signal: controller.signal,
    headers: {
      'X-API-Key': key,
      'Content-Type': 'application/json',
      ...(opts.headers || {}),
    },
  }).finally(() => clearTimeout(timeoutId));
}

// ============================================================
// Number Formatters
// ============================================================
function fmtUSD(n) {
  if (n == null || isNaN(n)) return '$—';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(n);
}

function fmtPct(n) {
  if (n == null || isNaN(n)) return '—';
  const sign = n >= 0 ? '+' : '';
  return `${sign}${n.toFixed(2)}%`;
}

function fmtCompact(n) {
  if (n == null || isNaN(n)) return '—';
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000)    return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

// ============================================================
// Cache helpers
// ============================================================
function cacheSet(key, data) {
  try { localStorage.setItem(`cache_${key}`, JSON.stringify({ ts: Date.now(), data })); } catch {}
}

function cacheGet(key) {
  try {
    const raw = localStorage.getItem(`cache_${key}`);
    return raw ? JSON.parse(raw).data : null;
  } catch { return null; }
}

// ============================================================
// Toast
// ============================================================
let toastTimer = null;
function showToast(msg, duration = 3000) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), duration);
}

// ============================================================
// Cold-start overlay
// ============================================================
let coldStartShown = false;
function showColdStart() {
  if (coldStartShown) return;
  coldStartShown = true;
  document.getElementById('coldstart-overlay').hidden = false;
}
function hideColdStart() {
  document.getElementById('coldstart-overlay').hidden = true;
}

// ============================================================
// API Key gate
// ============================================================
function initApiKeyGate() {
  if (getApiKey()) return;
  const overlay = document.getElementById('api-key-overlay');
  overlay.hidden = false;
  document.getElementById('api-key-save').addEventListener('click', () => {
    const val = document.getElementById('api-key-input').value.trim();
    if (!val) return;
    localStorage.setItem('app_api_key', val);
    overlay.hidden = true;
    loadTab('dashboard');
  });
}

// ============================================================
// Sheet management
// ============================================================
const backdrop = document.getElementById('sheet-backdrop');

function openSheet(id) {
  const sheet = document.getElementById(id);
  sheet.hidden = false;
  requestAnimationFrame(() => {
    backdrop.hidden = false;
    requestAnimationFrame(() => {
      sheet.classList.add('open');
      backdrop.classList.add('visible');
    });
  });
}

function closeSheet(id) {
  const sheet = document.getElementById(id);
  sheet.classList.remove('open');
  backdrop.classList.remove('visible');
  setTimeout(() => {
    sheet.hidden = true;
    backdrop.hidden = true;
  }, 350);
}

function closeAllSheets() {
  ['stock-sheet', 'add-holding-sheet', 'edit-cash-sheet'].forEach(closeSheet);
}

function initSheetSwipeClose(sheetId) {
  const sheet = document.getElementById(sheetId);
  let startY = 0;
  sheet.addEventListener('touchstart', e => { startY = e.touches[0].clientY; }, { passive: true });
  sheet.addEventListener('touchend', e => {
    const dy = e.changedTouches[0].clientY - startY;
    if (dy > 80) closeSheet(sheetId);
  }, { passive: true });
}

// ============================================================
// Badge helper
// ============================================================
function badgeClass(signal) {
  if (!signal) return 'badge';
  const s = signal.toUpperCase();
  if (s === 'BUY')  return 'badge badge-buy';
  if (s === 'SELL') return 'badge badge-sell';
  return 'badge badge-hold';
}

// ============================================================
// Sparkline SVG
// ============================================================
function buildSparkline(prices, isUp) {
  if (!prices || prices.length < 2) return '';
  const W = 60, H = 24, pad = 1;
  const min = Math.min(...prices), max = Math.max(...prices);
  const range = max - min || 1;
  const pts = prices.map((p, i) => {
    const x = pad + (i / (prices.length - 1)) * (W - pad * 2);
    const y = H - pad - ((p - min) / range) * (H - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const color = isUp ? 'var(--green)' : 'var(--red)';
  return `<svg class="sparkline" viewBox="0 0 ${W} ${H}" aria-hidden="true">
    <path d="M${pts.join('L')}" stroke="${color}" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;
}

// ============================================================
// Inline price chart SVG (stock detail)
// ============================================================
function buildPriceChart(prices) {
  if (!prices || prices.length < 2) return '<p class="text-sec">No price history</p>';
  const W = 320, H = 140, padX = 4, padY = 8;
  const min = Math.min(...prices), max = Math.max(...prices);
  const range = max - min || 1;
  const pts = prices.map((p, i) => {
    const x = padX + (i / (prices.length - 1)) * (W - padX * 2);
    const y = H - padY - ((p - min) / range) * (H - padY * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const isUp = prices[prices.length - 1] >= prices[0];
  const color = isUp ? 'var(--green)' : 'var(--red)';
  const fillPts = `${pts[0].split(',')[0]},${H} ${pts.join(' ')} ${pts[pts.length - 1].split(',')[0]},${H}`;
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px;display:block" aria-label="Price chart">
    <defs>
      <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="${color}" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
      </linearGradient>
    </defs>
    <polygon points="${fillPts}" fill="url(#cg)"/>
    <polyline points="${pts.join(' ')}" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;
}

// ============================================================
// Contribution bars (stock detail "Why")
// ============================================================
function buildContribBars(contribs) {
  if (!contribs || !contribs.length) return '<p class="text-sec text-sm">No feature data available.</p>';
  const maxAbs = Math.max(...contribs.map(c => Math.abs(c.value)));
  return `<div class="contrib-bar-wrap">${contribs.map(c => {
    const pct = maxAbs > 0 ? (Math.abs(c.value) / maxAbs * 100).toFixed(1) : 0;
    const col = c.value >= 0 ? 'var(--green)' : 'var(--red)';
    return `<div class="contrib-row">
      <span class="contrib-label">${esc(c.feature || c.name || '')}</span>
      <div class="contrib-track">
        <div class="contrib-fill" style="width:${pct}%;background:${col}"></div>
      </div>
      <span class="contrib-val ${c.value >= 0 ? 'price-up' : 'price-down'}">${c.value >= 0 ? '+' : ''}${c.value.toFixed(2)}</span>
    </div>`;
  }).join('')}</div>`;
}

// ============================================================
// Escape HTML
// ============================================================
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ============================================================
// Pull-to-refresh
// ============================================================
function initPullToRefresh(sectionId, reloadFn) {
  const section = document.getElementById(sectionId);
  let startY = 0, pulling = false, triggered = false;
  const THRESHOLD = 60;

  section.addEventListener('touchstart', e => {
    if (section.scrollTop === 0) { startY = e.touches[0].clientY; pulling = true; triggered = false; }
  }, { passive: true });

  section.addEventListener('touchmove', e => {
    if (!pulling) return;
    const dy = e.touches[0].clientY - startY;
    if (dy > THRESHOLD && !triggered) {
      triggered = true;
      showToast('Refreshing…', 1500);
      reloadFn();
    }
  }, { passive: true });

  section.addEventListener('touchend', () => { pulling = false; }, { passive: true });
}

// ============================================================
// Stock Detail Sheet
// ============================================================
async function openStockSheet(ticker) {
  openSheet('stock-sheet');
  document.getElementById('sheet-ticker').textContent = ticker;
  document.getElementById('sheet-badge').className = 'badge';
  document.getElementById('sheet-badge').textContent = '';
  document.getElementById('stock-sheet-body').innerHTML = `
    <div class="skeleton-block" style="height:48px;margin-bottom:16px"></div>
    <div class="skeleton-block" style="height:140px;margin-bottom:16px"></div>
    <div class="skeleton-block" style="height:80px"></div>`;

  try {
    showColdStart();
    const res = await apiFetch(`/stock/${encodeURIComponent(ticker)}`);
    hideColdStart();
    if (!res.ok) throw new Error(res.statusText);
    const d = await res.json();
    cacheSet(`stock_${ticker}`, d);
    renderStockSheet(d);
  } catch (err) {
    hideColdStart();
    const cached = cacheGet(`stock_${ticker}`);
    if (cached) { renderStockSheet(cached); showToast('Showing cached data'); }
    else {
      document.getElementById('stock-sheet-body').innerHTML =
        '<p class="text-sec" style="padding:24px 0;text-align:center">Failed to load stock data.</p>';
      showToast("Couldn't reach the server. Try again.");
    }
  }
}

function renderStockSheet(d) {
  const signal = (d.verdict || d.signal || '').toUpperCase();
  const badge = document.getElementById('sheet-badge');
  badge.className = badgeClass(signal);
  badge.textContent = signal;

  const prices = d.price_history || [];
  const isUp = prices.length >= 2 ? prices[prices.length - 1] >= prices[0] : true;
  const priceChange = prices.length >= 2 ? prices[prices.length - 1] - prices[0] : 0;
  const pricePct = prices.length >= 2 && prices[0] ? (priceChange / prices[0]) * 100 : 0;
  const currentPrice = prices.length ? prices[prices.length - 1] : d.price;

  const contribs = d.top_contributions || d.contributions || [];

  document.getElementById('stock-sheet-body').innerHTML = `
    <div>
      <div class="hero-value">${fmtUSD(currentPrice)}</div>
      <div class="hero-sub ${isUp ? 'price-up' : 'price-down'}">${fmtPct(pricePct)} today</div>
    </div>
    <div>${buildPriceChart(prices)}</div>
    <div>
      <div class="section-heading">Why this signal?</div>
      ${buildContribBars(contribs)}
    </div>
    ${d.confidence != null ? `<div class="flex-between"><span class="text-sec">Model confidence</span><span class="bold num">${(d.confidence * 100).toFixed(1)}%</span></div>` : ''}
    ${d.sector ? `<div class="flex-between"><span class="text-sec">Sector</span><span class="bold">${esc(d.sector)}</span></div>` : ''}
  `;
}

// ============================================================
// Dashboard
// ============================================================
async function loadDashboard() {
  const el = document.getElementById('dashboard-content');
  el.innerHTML = `
    <div class="skeleton-block" style="height:120px;margin-bottom:16px"></div>
    <div class="skeleton-block" style="height:72px;margin-bottom:12px"></div>
    <div class="skeleton-block" style="height:72px;margin-bottom:12px"></div>
    <div class="skeleton-block" style="height:72px"></div>`;

  try {
    showColdStart();
    const res = await apiFetch('/portfolio/overview');
    hideColdStart();
    if (!res.ok) throw new Error(res.statusText);
    const d = await res.json();
    cacheSet('dashboard', d);
    renderDashboard(d);
  } catch {
    hideColdStart();
    const cached = cacheGet('dashboard');
    if (cached) { renderDashboard(cached); showToast('Showing cached data'); }
    else {
      el.innerHTML = '<div class="empty-state"><p>Could not load overview.</p></div>';
      showToast("Couldn't reach the server. Try again.");
    }
  }
}

function renderDashboard(d) {
  const el = document.getElementById('dashboard-content');
  const totalValue = d.total_value ?? d.portfolio_value ?? 0;
  const totalCost  = d.total_cost ?? d.cost_basis ?? 0;
  const gainLoss   = totalValue - totalCost;
  const gainPct    = totalCost > 0 ? (gainLoss / totalCost) * 100 : 0;
  const cash       = d.cash ?? d.user_cash ?? 0;
  const dayChange  = d.day_change ?? 0;
  const dayChangePct = d.day_change_pct ?? 0;
  const topHoldings = d.top_holdings || d.holdings || [];

  el.innerHTML = `
    <div class="card" style="margin-bottom:16px">
      <div class="text-xs text-sec" style="margin-bottom:4px">Total Portfolio Value</div>
      <div class="hero-value">${fmtUSD(totalValue)}</div>
      <div class="hero-sub ${gainLoss >= 0 ? 'price-up' : 'price-down'}">${fmtPct(gainPct)} all-time &nbsp;·&nbsp; ${fmtUSD(gainLoss)}</div>
    </div>

    <div class="stat-grid">
      <div class="stat-chip">
        <div class="stat-chip-label">Cash</div>
        <div class="stat-chip-value">${fmtCompact(cash)}</div>
      </div>
      <div class="stat-chip">
        <div class="stat-chip-label">Today</div>
        <div class="stat-chip-value ${dayChange >= 0 ? 'price-up' : 'price-down'}">${fmtPct(dayChangePct)}</div>
      </div>
      <div class="stat-chip">
        <div class="stat-chip-label">Cost Basis</div>
        <div class="stat-chip-value">${fmtCompact(totalCost)}</div>
      </div>
      <div class="stat-chip">
        <div class="stat-chip-label">Positions</div>
        <div class="stat-chip-value">${topHoldings.length}</div>
      </div>
    </div>

    ${topHoldings.length ? `
    <div class="section-heading">Top Holdings</div>
    ${topHoldings.map(h => {
      const chg = h.change_pct ?? h.pct_change ?? 0;
      const val = h.market_value ?? h.value ?? 0;
      return `<div class="card card-tappable" role="button" tabindex="0" data-ticker="${esc(h.ticker)}" style="margin-bottom:10px">
        <div class="stock-row">
          <div class="stock-row-left">
            <div class="ticker-label">${esc(h.ticker)}</div>
            <div class="company-name">${esc(h.name || h.company || '')}</div>
          </div>
          <div class="stock-row-right">
            <div class="bold num">${fmtUSD(val)}</div>
            <div class="text-sm ${chg >= 0 ? 'price-up' : 'price-down'} num">${fmtPct(chg)}</div>
          </div>
        </div>
      </div>`;
    }).join('')}` : ''}
  `;

  el.querySelectorAll('[data-ticker]').forEach(el => {
    el.addEventListener('click', () => openStockSheet(el.dataset.ticker));
  });
}

// ============================================================
// Picks
// ============================================================
let currentSignalFilter = 'BUY';

async function loadPicks() {
  const el = document.getElementById('picks-content');
  el.innerHTML = `
    <div class="skeleton-block" style="height:80px;margin-bottom:12px"></div>
    <div class="skeleton-block" style="height:80px;margin-bottom:12px"></div>
    <div class="skeleton-block" style="height:80px;margin-bottom:12px"></div>
    <div class="skeleton-block" style="height:80px"></div>`;

  try {
    showColdStart();
    const res = await apiFetch('/signals/latest');
    hideColdStart();
    if (!res.ok) throw new Error(res.statusText);
    const d = await res.json();
    cacheSet('picks', d);
    renderPicks(d);
  } catch {
    hideColdStart();
    const cached = cacheGet('picks');
    if (cached) { renderPicks(cached); showToast('Showing cached data'); }
    else {
      el.innerHTML = '<div class="empty-state"><p>Could not load signals.</p></div>';
      showToast("Couldn't reach the server. Try again.");
    }
  }
}

function renderPicks(data) {
  const signals = Array.isArray(data) ? data : (data.signals || data.picks || []);
  const el = document.getElementById('picks-content');

  const filtered = currentSignalFilter === 'ALL'
    ? signals
    : signals.filter(s => (s.signal || s.verdict || '').toUpperCase() === currentSignalFilter);

  if (!filtered.length) {
    el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📭</div><p>No ${currentSignalFilter === 'ALL' ? '' : currentSignalFilter + ' '}signals right now.</p></div>`;
    return;
  }

  el.innerHTML = filtered.map(s => {
    const signal = (s.signal || s.verdict || 'HOLD').toUpperCase();
    const chg = s.change_pct ?? s.pct_change ?? 0;
    const prices = s.price_history || [];
    const isUp = chg >= 0;
    return `<div class="card card-tappable" role="button" tabindex="0" data-ticker="${esc(s.ticker)}" style="margin-bottom:10px">
      <div class="stock-row">
        <div class="stock-row-left">
          <div class="flex-row" style="margin-bottom:4px">
            <span class="ticker-label">${esc(s.ticker)}</span>
            <span class="${badgeClass(signal)}">${signal}</span>
          </div>
          <div class="company-name">${esc(s.name || s.company || '')}</div>
        </div>
        <div class="stock-row-right" style="display:flex;flex-direction:column;align-items:flex-end;gap:4px">
          ${prices.length >= 2 ? buildSparkline(prices, isUp) : ''}
          <div class="text-sm ${isUp ? 'price-up' : 'price-down'} num">${fmtPct(chg)}</div>
        </div>
      </div>
    </div>`;
  }).join('');

  el.querySelectorAll('[data-ticker]').forEach(card => {
    card.addEventListener('click', () => openStockSheet(card.dataset.ticker));
  });
}

// ============================================================
// Portfolio
// ============================================================
async function loadPortfolio() {
  const el = document.getElementById('portfolio-content');
  el.innerHTML = `
    <div class="skeleton-block" style="height:100px;margin-bottom:16px"></div>
    <div class="skeleton-block" style="height:72px;margin-bottom:12px"></div>
    <div class="skeleton-block" style="height:72px;margin-bottom:12px"></div>
    <div class="skeleton-block" style="height:72px"></div>`;

  try {
    showColdStart();
    const res = await apiFetch('/holdings');
    hideColdStart();
    if (!res.ok) throw new Error(res.statusText);
    const holdings = await res.json();
    cacheSet('holdings', holdings);
    renderPortfolio(holdings);
  } catch {
    hideColdStart();
    const cached = cacheGet('holdings');
    if (cached) { renderPortfolio(cached); showToast('Showing cached data'); }
    else {
      el.innerHTML = '<div class="empty-state"><p>Could not load holdings.</p></div>';
      showToast("Couldn't reach the server. Try again.");
    }
  }
}

function renderPortfolio(data) {
  const holdings = Array.isArray(data) ? data : (data.holdings || []);
  const cash = data.cash ?? data.user_cash ?? (Array.isArray(data) ? 0 : 0);
  const el = document.getElementById('portfolio-content');

  const totalValue = holdings.reduce((s, h) => s + (h.market_value ?? h.value ?? 0), 0) + cash;

  el.innerHTML = `
    <div class="card" style="margin-bottom:16px">
      <div class="text-xs text-sec" style="margin-bottom:4px">Portfolio Value</div>
      <div class="hero-value">${fmtUSD(totalValue)}</div>
    </div>

    <!-- Cash row -->
    <div class="card card-tappable" id="cash-row" style="margin-bottom:10px">
      <div class="stock-row">
        <div class="stock-row-left">
          <div class="ticker-label">Cash</div>
        </div>
        <div class="stock-row-right flex-row" style="gap:12px">
          <span class="bold num">${fmtUSD(cash)}</span>
          <button class="icon-btn" id="edit-cash-btn" aria-label="Edit cash" style="min-width:36px;min-height:36px">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
              <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
            </svg>
          </button>
        </div>
      </div>
    </div>

    ${holdings.length ? `
    <div class="section-heading">Holdings</div>
    <div id="holdings-list">
      ${holdings.map(h => buildHoldingCard(h)).join('')}
    </div>` : '<div class="empty-state" style="padding-top:24px"><p>No holdings yet. Tap + to add one.</p></div>'}
  `;

  document.getElementById('edit-cash-btn').addEventListener('click', e => {
    e.stopPropagation();
    document.getElementById('cash-amount').value = cash;
    openSheet('edit-cash-sheet');
  });

  el.querySelectorAll('[data-ticker]').forEach(card => {
    card.addEventListener('click', () => openStockSheet(card.dataset.ticker));
  });

  setupSwipeDelete();
}

function buildHoldingCard(h) {
  const val  = h.market_value ?? h.value ?? 0;
  const cost = h.cost_basis  ?? ((h.avg_cost ?? 0) * (h.shares ?? 0));
  const gain = val - cost;
  const gainPct = cost > 0 ? (gain / cost * 100) : 0;
  const signal = (h.verdict || h.signal || '').toUpperCase();
  return `<div class="swipeable" style="margin-bottom:10px">
    <div class="card card-tappable" data-ticker="${esc(h.ticker)}" role="button" tabindex="0">
      <div class="stock-row">
        <div class="stock-row-left">
          <div class="flex-row" style="margin-bottom:4px">
            <span class="ticker-label">${esc(h.ticker)}</span>
            ${signal ? `<span class="${badgeClass(signal)}">${signal}</span>` : ''}
          </div>
          <div class="company-name">${h.shares ?? '?'} shares · avg ${fmtUSD(h.avg_cost ?? 0)}</div>
        </div>
        <div class="stock-row-right">
          <div class="bold num">${fmtUSD(val)}</div>
          <div class="text-sm ${gain >= 0 ? 'price-up' : 'price-down'} num">${fmtPct(gainPct)}</div>
        </div>
      </div>
    </div>
    <div class="swipe-action" data-delete-ticker="${esc(h.ticker)}">Remove</div>
  </div>`;
}

function setupSwipeDelete() {
  document.querySelectorAll('.swipeable').forEach(row => {
    let startX = 0;
    const card = row.querySelector('.card');
    const action = row.querySelector('.swipe-action');
    if (!card || !action) return;

    card.addEventListener('touchstart', e => { startX = e.touches[0].clientX; }, { passive: true });
    card.addEventListener('touchmove', e => {
      const dx = startX - e.touches[0].clientX;
      if (dx > 0) card.style.transform = `translateX(-${Math.min(dx, 90)}px)`;
    }, { passive: true });
    card.addEventListener('touchend', e => {
      const dx = startX - e.changedTouches[0].clientX;
      if (dx > 60) {
        action.style.transform = 'translateX(0)';
        card.style.transform = 'translateX(-90px)';
      } else {
        card.style.transform = '';
        action.style.transform = '';
      }
    }, { passive: true });

    action.addEventListener('click', async () => {
      const ticker = action.dataset.deleteTicker;
      try {
        const res = await apiFetch(`/holdings/${encodeURIComponent(ticker)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error();
        row.remove();
        showToast(`Removed ${ticker}`);
      } catch {
        showToast("Couldn't remove holding. Try again.");
        card.style.transform = '';
        action.style.transform = '';
      }
    });
  });
}

// ============================================================
// Rebalance
// ============================================================
async function loadRebalance() {
  // No auto-load; user taps "Generate Trades"
}

async function generateTrades() {
  const btn = document.getElementById('generate-trades-btn');
  const el  = document.getElementById('rebalance-content');
  btn.disabled = true;
  btn.textContent = 'Generating…';
  el.innerHTML = `<div class="skeleton-block" style="height:72px;margin-bottom:12px"></div>
    <div class="skeleton-block" style="height:72px;margin-bottom:12px"></div>
    <div class="skeleton-block" style="height:72px"></div>`;

  try {
    showColdStart();
    const res = await apiFetch('/recommend', { method: 'POST', body: JSON.stringify({}) });
    hideColdStart();
    if (!res.ok) throw new Error(res.statusText);
    const d = await res.json();
    cacheSet('rebalance', d);
    renderRebalance(d);
  } catch {
    hideColdStart();
    const cached = cacheGet('rebalance');
    if (cached) { renderRebalance(cached); showToast('Showing cached recommendations'); }
    else {
      el.innerHTML = '<div class="empty-state"><p>Could not generate trades.</p></div>';
      showToast("Couldn't reach the server. Try again.");
    }
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate Trades';
  }
}

function renderRebalance(d) {
  const trades = Array.isArray(d) ? d : (d.trades || d.recommendations || []);
  const el = document.getElementById('rebalance-content');

  if (!trades.length) {
    el.innerHTML = '<div class="empty-state"><p>Portfolio is already balanced.</p></div>';
    return;
  }

  el.innerHTML = `<div class="section-heading">Recommended Trades</div>
    ${trades.map(t => {
      const action = (t.action || t.trade || '').toUpperCase();
      const isBuy  = action === 'BUY';
      return `<div class="card" style="margin-bottom:10px">
        <div class="trade-row">
          <span class="ticker-label">${esc(t.ticker)}</span>
          <span class="${isBuy ? 'trade-action-buy' : 'trade-action-sell'}">${action}</span>
          <span class="num">${t.shares != null ? `${t.shares} shares` : ''}</span>
          <span class="num bold">${fmtUSD(t.value ?? t.amount)}</span>
        </div>
        ${t.reason ? `<div class="text-sm text-sec" style="margin-top:6px">${esc(t.reason)}</div>` : ''}
      </div>`;
    }).join('')}`;
}

// ============================================================
// Tab routing
// ============================================================
const TAB_META = {
  dashboard: { label: 'Dashboard', load: loadDashboard },
  picks:     { label: 'Picks',     load: loadPicks     },
  portfolio: { label: 'Portfolio', load: loadPortfolio },
  rebalance: { label: 'Rebalance', load: loadRebalance },
};

let activeTab = null;

function loadTab(name) {
  if (!TAB_META[name]) return;
  activeTab = name;

  // Update tab bar
  document.querySelectorAll('.tab-btn').forEach(btn => {
    const isActive = btn.dataset.tab === name;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', isActive);
  });

  // Show/hide sections
  document.querySelectorAll('.tab-section').forEach(sec => {
    const isActive = sec.id === `section-${name}`;
    sec.hidden = !isActive;
    if (isActive) sec.removeAttribute('hidden');
    else sec.setAttribute('hidden', '');
  });

  // Update app bar title
  document.getElementById('app-bar-title').textContent = TAB_META[name].label;

  // Load data
  TAB_META[name].load();
}

// ============================================================
// Add Holding Form
// ============================================================
async function submitAddHolding(e) {
  e.preventDefault();
  const ticker = document.getElementById('holding-ticker').value.trim().toUpperCase();
  const shares = parseFloat(document.getElementById('holding-shares').value);
  const cost   = parseFloat(document.getElementById('holding-cost').value);

  if (!ticker || isNaN(shares) || isNaN(cost)) { showToast('Please fill all fields.'); return; }

  try {
    const res = await apiFetch(`/holdings/${encodeURIComponent(ticker)}`, {
      method: 'POST',
      body: JSON.stringify({ shares, avg_cost: cost }),
    });
    if (!res.ok) throw new Error();
    closeSheet('add-holding-sheet');
    showToast(`Added ${ticker}`);
    loadPortfolio();
    document.getElementById('add-holding-form').reset();
  } catch {
    showToast("Couldn't add holding. Try again.");
  }
}

// ============================================================
// Edit Cash Form
// ============================================================
async function submitEditCash(e) {
  e.preventDefault();
  const amount = parseFloat(document.getElementById('cash-amount').value);
  if (isNaN(amount) || amount < 0) { showToast('Enter a valid cash amount.'); return; }

  try {
    const res = await apiFetch('/user/cash', {
      method: 'POST',
      body: JSON.stringify({ cash: amount }),
    });
    if (!res.ok) throw new Error();
    closeSheet('edit-cash-sheet');
    showToast('Cash updated');
    loadPortfolio();
  } catch {
    showToast("Couldn't update cash. Try again.");
  }
}

// ============================================================
// Service Worker
// ============================================================
async function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  try {
    await navigator.serviceWorker.register('./sw.js');
  } catch {}
}

// ============================================================
// Init
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  initApiKeyGate();

  // Tab bar clicks
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => loadTab(btn.dataset.tab));
  });

  // Refresh button
  document.getElementById('refresh-btn').addEventListener('click', () => {
    if (activeTab) TAB_META[activeTab].load();
  });

  // Sheet close buttons
  document.querySelectorAll('.sheet-close').forEach(btn => {
    btn.addEventListener('click', () => closeSheet(btn.dataset.sheet));
  });

  // Backdrop tap to close
  backdrop.addEventListener('click', closeAllSheets);

  // Swipe-down to close sheets
  ['stock-sheet', 'add-holding-sheet', 'edit-cash-sheet'].forEach(initSheetSwipeClose);

  // Add holding
  document.getElementById('add-holding-btn').addEventListener('click', () => openSheet('add-holding-sheet'));
  document.getElementById('add-holding-form').addEventListener('submit', submitAddHolding);

  // Edit cash
  document.getElementById('edit-cash-form').addEventListener('submit', submitEditCash);

  // Generate trades
  document.getElementById('generate-trades-btn').addEventListener('click', generateTrades);

  // Filter buttons (Picks)
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentSignalFilter = btn.dataset.signal;
      const cached = cacheGet('picks');
      if (cached) renderPicks(cached);
    });
  });

  // Pull-to-refresh per section
  ['section-dashboard', 'section-picks', 'section-portfolio', 'section-rebalance'].forEach((secId, i) => {
    const tabs = ['dashboard', 'picks', 'portfolio', 'rebalance'];
    initPullToRefresh(secId, () => TAB_META[tabs[i]].load());
  });

  // Service worker
  registerServiceWorker();

  // Initial load
  if (getApiKey()) loadTab('dashboard');
});
