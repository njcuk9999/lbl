/*
 * The LBL window: the page. It holds the window's state (saved on the server,
 * ~/.lbl_pca2d_gui.json), draws what the server reads from LBL and from the
 * data folder, and hands the server what to build and run. No framework:
 * one file, plain DOM and SVG.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const SERIES = ['--series-1', '--series-2', '--series-3', '--series-4',
                '--series-5', '--series-6', '--series-7', '--series-8'];
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SCATTER_COLOURED = 3;   // colours that stay apart from every other (scatter)
const HIST_COLOURED = 8;      // colours that stay apart from their neighbours (stacks)

const DEFAULTS = {
  lang: 'en', theme: 'light', tab: 'targets', root: '', folder: '', data_dir: '', instrument: '', source: '',
  python: '', ticked: [], hidden_instruments: [], sort: { key: 'name', dir: 1 },
  comparison: {}, teff: {}, lbl: {}, pca2d_on: true, pca2d: {},
  extra: {}, pca2d_extra: {},
  skip_pca2d: null, min_rjd: null, max_rjd: null, resproj: false,
  name: '', name_auto: true, sub: 'local', archive_instruments: null,
};
const S = JSON.parse(JSON.stringify(DEFAULTS));
let T = {};
let INSTRUMENTS = [];
let OPTIONS = [];
let SWITCHES = [];
let TARGETS = [];
let BASES = [];
let TEMPLATES = [];
let VETTED = [];
let POINTS = {};
let wrapEdited = false;
let selectedJob = null;
let stream = null;
let logLast = null;
let scanTimer = null;
let SCIENCE = { state: 'ok', folder: '' };
let LAYOUT = null;       // the root: its instrument folders and its Downloads
let DISPATCH = null;     // the dispatch of Downloads: its plan, then what it did
let dispatchTimer = null;
const TREE_OPEN = new Set(['Downloads/']);   // the tree's folders shown open

// ── plumbing ────────────────────────────────────────────────────────────
const api = {
  async get(path, params) {
    const url = new URL(path, location.origin);
    Object.entries(params || {}).forEach(([k, v]) => url.searchParams.set(k, v ?? ''));
    const response = await fetch(url);
    return response.json();
  },
  async post(path, body) {
    const response = await fetch(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    const data = await response.json().catch(() => ({}));
    data._status = response.status;
    return data;
  },
};

function t(key, ...args) {
  let text = T[key] ?? key;
  args.forEach((arg) => { text = text.replace('%s', arg); });
  return text;
}

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  Object.entries(attrs || {}).forEach(([k, v]) => {
    if (v === null || v === undefined || v === false) return;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : v);
  });
  // flattened all the way, so that a list of [link, ' '] pairs (the Runs
  //   tab's reports) is appended as nodes, not written out as text
  children.flat(Infinity).forEach((child) => {
    if (child === null || child === undefined || child === false) return;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  });
  return node;
}

let saveTimer = null;
function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => api.post('/api/state', S), 400);
}

const fmt = (x, digits = 1) => (x === null || x === undefined || Number.isNaN(x) ? '' : Number(x).toFixed(digits));
function human(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let value = bytes;
  while (value >= 1000 && i < units.length - 1) { value /= 1000; i += 1; }
  return `${value.toFixed(i ? 1 : 0)} ${units[i]}`;
}
function mjdDate(mjd) {
  if (mjd === null || mjd === undefined) return '';
  return new Date((mjd - 40587) * 86400000).toISOString().slice(0, 10);
}
function when(seconds) {
  if (!seconds) return '';
  // the machine's own time, not UTC (which toISOString would give)
  const date = new Date(seconds * 1000);
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}
function took(start, end) {
  if (!start) return '';
  return span((end || Date.now() / 1000) - start);
}
/* a duration in seconds, as 1 h 05 min, 3 min 20 s or 12 s */
function span(seconds) {
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h ? `${h} h ${String(m).padStart(2, '0')} min` : m ? `${m} min ${String(s).padStart(2, '0')} s` : `${s} s`;
}
function fileLink(path, text) {
  return el('a', { href: `/file?path=${encodeURIComponent(path)}`, target: '_blank', text: text || t('open') });
}
/* the server's paths are the system's: / on macOS and Linux, \ on Windows
   (and either one there), so a path is cut on both */
const flatPath = (path) => String(path ?? '').replace(/\\/g, '/');
const baseName = (path) => String(path ?? '').split(/[\\/]/).pop();
/* a folder's child, written with the separator the folder is written with */
function childPath(folder, name) {
  const dir = String(folder ?? '').replace(/[\\/]+$/, '');
  return `${dir}${dir.includes('\\') && !dir.includes('/') ? '\\' : '/'}${name}`;
}
function relative(path) {
  const root = flatPath(S.root || S.data_dir || '').replace(/\/$/, '') + '/';
  return path && flatPath(path).startsWith(root) ? path.slice(root.length) : path;
}

// ── tooltip (data-tip) ──────────────────────────────────────────────────
function tip(event, html) {
  const box = $('tooltip');
  if (!html) { box.hidden = true; return; }
  box.innerHTML = html;
  box.hidden = false;
  const pad = 14;
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  const rect = box.getBoundingClientRect();
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
  box.style.left = `${x}px`;
  box.style.top = `${y}px`;
}
document.addEventListener('mouseover', (event) => {
  const target = event.target.closest('[data-tip]');
  if (target) tip(event, target.dataset.tip);
});
document.addEventListener('mousemove', (event) => {
  const target = event.target.closest('[data-tip]');
  if (target) tip(event, target.dataset.tip);
});
document.addEventListener('mouseout', (event) => {
  if (event.target.closest('[data-tip]')) tip(event, null);
});
const esc = (text) => String(text ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
// a help is plain text, a blank line between its paragraphs
const helpHtml = (text) => esc(text).replace(/\n\n/g, '<br><br>');
const info = (key) => el('span', { class: 'info', 'data-tip': helpHtml(t(key)), text: 'i' });
// an (i) inside a <label> explains it and does not tick or focus it
document.addEventListener('click', (event) => { if (event.target.closest('.info')) event.preventDefault(); });

function confirmBox(text) {
  return new Promise((resolve) => {
    $('modal_text').textContent = text;
    $('modal').hidden = false;
    const done = (answer) => {
      $('modal').hidden = true;
      $('modal_yes').onclick = null;
      $('modal_no').onclick = null;
      resolve(answer);
    };
    $('modal_yes').onclick = () => done(true);
    $('modal_no').onclick = () => done(false);
  });
}

// ── strings ─────────────────────────────────────────────────────────────
async function loadStrings() {
  T = await api.get('/api/strings', { lang: S.lang });
  document.documentElement.lang = S.lang;
  document.querySelectorAll('[data-t]').forEach((node) => { node.textContent = t(node.dataset.t); });
  document.querySelectorAll('[data-t-title]').forEach((node) => { node.title = t(node.dataset.tTitle); node.setAttribute('aria-label', node.title); });
  document.querySelectorAll('[data-t-placeholder]').forEach((node) => { node.placeholder = t(node.dataset.tPlaceholder); });
  document.querySelectorAll('[data-help]').forEach((node) => { node.textContent = 'i'; node.dataset.tip = helpHtml(t(`help_${node.dataset.help}`)); });
  $('lang').textContent = T.lang_switch;
  drawSetupStatus();
  applyTheme();
  updateLaunch(null);
  $('skip_pca2d').replaceChildren(
    el('option', { value: '', text: t('skip_auto') }),
    el('option', { value: 'true', text: t('skip_keep') }),
    el('option', { value: 'false', text: t('skip_redo') }));
  $('skip_pca2d').value = S.skip_pca2d === null ? '' : String(S.skip_pca2d);
}

// ── theme ───────────────────────────────────────────────────────────────
const SUN = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const MOON = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/></svg>';
function applyTheme() {
  document.documentElement.dataset.theme = S.theme;
  // the button offers the other theme
  const light = S.theme === 'light';
  $('theme').innerHTML = `${light ? MOON : SUN} <span>${t(light ? 'theme_dark' : 'theme_light')}</span>`;
}

async function switchTheme() {
  S.theme = S.theme === 'light' ? 'dark' : 'light';
  save();
  applyTheme();
  // the plots and swatches read their colours from the theme when drawn
  drawTargets();
  drawPlots();
  redrawExplorer();
  $('backdrop').classList.remove('ready');
  setBackdrop();
}

// ── tabs ────────────────────────────────────────────────────────────────
function showTab(name) {
  S.tab = name;
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.pane').forEach((p) => { p.hidden = p.id !== `pane-${name}`; });
  if (name === 'run') { if (!wrapEdited) build(); loadWrapList(); }
  if (name === 'runs') loadRuns();
  if (name === 'cleanup') measure();
  if (name === 'explore') loadExplorer();
  if (name === 'targets') drawPlots();
  if (name === 'refs') loadRefs();
  save();
}

// ── references ──────────────────────────────────────────────────────────
// the method's paper and those whose velocities LBL measured, each with the
//   sentence of the paper that says so (static/references.json)
let REFS = null;
async function loadRefs() {
  if (!REFS) {
    try {
      REFS = await (await fetch('/static/references.json')).json();
    } catch (error) {
      REFS = { method: null, papers: [], credits: [] };
    }
  }
  drawRefs();
}
/* a paper: its title, its authors and year, its ADS page */
function refItem(ref) {
  return el('div', { class: 'ref' },
    el('div', { class: 'ref-title', text: ref.title }),
    el('div', { class: 'ref-meta small' },
      el('span', { text: `${ref.authors} (${ref.year})` }),
      ref.ads ? el('a', { href: ref.ads, target: '_blank', rel: 'noopener', text: 'ADS' }) : null));
}
function drawRefs() {
  if (!REFS) return;
  $('refs_method').replaceChildren(REFS.method ? refItem(REFS.method) : el('span', { class: 'muted', text: '-' }));
  const papers = REFS.papers || [];
  $('refs_title').textContent = t('refs_used', papers.length);
  $('refs_hint').textContent = t('refs_used_hint', REFS.updated || '-');
  $('refs_list').replaceChildren(...papers.map((ref) => refItem(ref)));
  $('refs_credits').replaceChildren(...(REFS.credits || []).map((c) => el('li', {},
    el('span', { text: `${c.authors} (${c.year}), ` }), el('em', { text: c.title }), ' ',
    el('span', { class: 'muted', text: `[${c.licence}]` }),
    c.ads ? el('a', { href: c.ads, target: '_blank', rel: 'noopener', text: 'ADS' }) : null)));
}

// ── the targets' sub-tabs ───────────────────────────────────────────────
function showSub(name) {
  S.sub = name;
  document.querySelectorAll('#target_subtabs button').forEach((b) => b.classList.toggle('active', b.dataset.sub === name));
  $('sub-local').hidden = name !== 'local';
  $('sub-archive').hidden = name !== 'archive';
  if (name === 'archive') { drawArchiveInstruments(); pollArchive(); }
  save();
}

// ── the public archives (archives.py, through lbl.gui.fetch) ───────────
const ARCHIVE_INSTRUMENTS = [['SPIROU', 'SPIRou'], ['NIRPS', 'NIRPS'], ['HARPS', 'HARPS'], ['HARPSN', 'HARPS-N'], ['SOPHIE', 'SOPHIE'], ['ESPRESSO', 'ESPRESSO']];
const BIG_FETCH_MB = 5000;
let ARCHIVE = null;
let archiveChosen = new Set();
let archiveTimer = null;
const megabytes = (mb) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.round(mb)} MB`);

function drawArchiveInstruments() {
  const chosen = S.archive_instruments || ARCHIVE_INSTRUMENTS.map(([key]) => key);
  $('archive_instruments').replaceChildren(...ARCHIVE_INSTRUMENTS.map(([key, label]) => el('label', { class: 'check' },
    el('input', {
      type: 'checkbox', checked: chosen.includes(key),
      onchange: (event) => {
        const now = new Set(S.archive_instruments || ARCHIVE_INSTRUMENTS.map(([k]) => k));
        if (event.target.checked) now.add(key); else now.delete(key);
        S.archive_instruments = [...now];
        save();
      },
    }), label)));
}

async function archiveSearch() {
  const target = $('archive_target').value.trim();
  if (!target) return;
  archiveChosen = new Set();
  ARCHIVE = await api.post('/api/archive/search', {
    target, data_dir: S.root,
    instruments: S.archive_instruments || ARCHIVE_INSTRUMENTS.map(([key]) => key),
  });
  drawArchive();
  pollArchive();
}

async function pollArchive() {
  ARCHIVE = await api.get('/api/archive');
  drawArchive();
  const busy = ARCHIVE && (ARCHIVE.search.state === 'running' || ARCHIVE.fetch.state === 'running');
  clearTimeout(archiveTimer);
  if (busy) archiveTimer = setTimeout(pollArchive, 1500);
}

function drawArchive() {
  if (!ARCHIVE) return;
  $('archive_unavailable').textContent = ARCHIVE.unavailable ? t('archive_unavailable', ARCHIVE.unavailable) : '';
  const search = ARCHIVE.search || {};
  const fetch = ARCHIVE.fetch || {};
  $('archive_search').disabled = search.state === 'running' || Boolean(ARCHIVE.unavailable);
  // while a search goes, its Stop takes the place of Search
  $('archive_search').hidden = search.state === 'running';
  $('archive_search_stop').hidden = search.state !== 'running';
  // what the search and the fetch are doing, line by line; followed when
  //   it is at its bottom
  const box = $('archive_log');
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
  const lines = ARCHIVE.log || [];
  $('archive_log_box').hidden = !lines.length;
  box.replaceChildren(...lines.map((line) => el('div', { class: `lv-${line.level}` },
    el('span', { class: 'muted', text: `${line.t}  ` }), line.text)));
  if (atBottom) box.scrollTop = box.scrollHeight;
  logEdges('archive_log');
  // each star as SIMBAD knows it: its name, where it is, its Teff
  const targets = search.targets || (search.target ? [search.target] : []);
  const stars = search.stars || {};
  const several = targets.length > 1;
  $('archive_star').replaceChildren(...(search.error && !targets.length ? [el('div', { text: search.error })] : targets.map((name) => {
    const about = stars[name] || {};
    if (about.state === 'resolving') return el('div', { class: 'muted' }, el('span', { class: 'ring' }), ` ${name}: SIMBAD...`);
    if (about.state === 'failed') return el('div', { class: 'warn-text', text: `${name}: ${about.error || t('archive_failed')}` });
    if (about.state === 'stopped') return el('div', { class: 'muted', text: `${name}: ${t('archive_stopped')}` });
    return el('div', {},
      el('strong', { text: about.main || search.main || name }),
      about.place ? el('span', { class: 'mono', text: `  ${about.place}` }) : null,
      `  ·  Teff ${about.teff ? Math.round(about.teff) : '?'} K`,
      several && (about.main || '') !== name ? el('span', { class: 'muted', text: `  (${name})` }) : null);
  })));
  // one chip per instrument, per star when there are several
  const cells = search.cells || {};
  const chip = (key, about, label) => {
    const text = about.state === 'searching' ? t('archive_searching')
      : about.state === 'failed' ? t('archive_failed')
        : about.state === 'stopped' ? t('archive_stopped')
          : about.state === 'skipped' ? t('archive_skipped') : t('archive_found', about.n || 0);
    return el('span', { class: `chip ${about.state}`, 'data-tip': about.error ? esc(about.error) : null },
      about.state === 'searching' ? el('span', { class: 'ring' }) : null, `${label} · ${text}`);
  };
  const instLabel = (key) => (ARCHIVE_INSTRUMENTS.find(([k]) => k === key) || [key, key])[1];
  $('archive_states').replaceChildren(...(Object.keys(cells).length
    ? Object.entries(cells).map(([key, about]) => {
      const [name, inst] = key.split('|');
      return chip(key, about, several ? `${name} · ${instLabel(inst)}` : instLabel(inst));
    })
    : Object.entries(search.instruments || {}).map(([key, about]) => chip(key, about, instLabel(key)))));
  const rows = search.groups || [];
  $('archive_wrap').hidden = !rows.length;
  if (search.state === 'done' && !rows.length && targets.length) {
    $('archive_star').append(el('div', { class: 'muted', text: t('archive_nothing', targets.join(', ')) }));
  }
  const columns = [...(several ? ['col_target'] : []), 'col_instrument', 'col_reader', 'col_archive', 'col_files', 'col_have', 'col_size', 'col_folder'];
  const head = el('tr', {}, el('th', {}, ''), ...columns.map((key) => el('th', { text: t(key) })));
  const body = rows.map((row) => el('tr', { class: archiveChosen.has(row.id) ? 'ticked' : '' },
    el('td', {}, el('input', {
      type: 'checkbox', checked: archiveChosen.has(row.id), disabled: fetch.state === 'running',
      onchange: (event) => { if (event.target.checked) archiveChosen.add(row.id); else archiveChosen.delete(row.id); drawArchive(); },
    })),
    several ? el('td', { text: row.target || '' }) : null,
    el('td', { text: row.label }),
    el('td', { class: 'mono', text: `${row.lbl} · ${row.source}` }),
    el('td', { class: 'small', text: row.archive }),
    el('td', { class: 'num', text: row.n }),
    el('td', { class: 'num', text: row.have || '' }),
    el('td', { class: 'num', text: row.size_mb ? megabytes(row.size_mb) : '0' }),
    el('td', { class: 'mono', text: `${row.home}/science/${row.folder}` })));
  $('archive_table').replaceChildren(el('thead', {}, head), el('tbody', {}, body));
  const total = rows.filter((row) => archiveChosen.has(row.id)).reduce((sum, row) => sum + (row.size_mb || 0), 0);
  $('archive_fetch_label').textContent = archiveChosen.size ? t('archive_fetch', megabytes(total)) : t('archive_fetch_none');
  $('archive_fetch').disabled = !archiveChosen.size || fetch.state === 'running' || !S.data_dir;
  $('archive_fetch').classList.toggle('running', fetch.state === 'running');
  $('archive_stop').hidden = fetch.state !== 'running';
  // one bar per instrument and target fetched
  $('archive_bars').replaceChildren(...(fetch.bars || []).map((bar) => {
    const share = bar.total ? Math.min(100, (100 * bar.done) / bar.total) : 100;
    const whose = bar.target || search.main || search.target || '';
    return el('div', { class: 'bar-row' },
      el('div', { class: 'bar-text small', text: t('archive_progress', `${bar.label} ${whose} (${bar.home}/science/${bar.folder})`, bar.done, bar.total, bar.failed) + barTime(bar, ARCHIVE.now) }),
      el('div', { class: 'bar' }, el('div', { class: `bar-fill${bar.failed ? ' with-failed' : ''}`, style: `width:${share.toFixed(1)}%` })));
  }));
  if (fetch.state === 'running') {
    $('archive_progress').textContent = '';
  } else if (fetch.state === 'done' || fetch.state === 'stopped') {
    $('archive_progress').textContent = `${t('archive_done', (fetch.folders || []).map((f) => `science/${f}`).join(', '))}${fetch.failed ? ` · ${t('archive_progress', '', fetch.done, fetch.total, fetch.failed)}${fetch.why ? ` (${fetch.why})` : ''}` : ''}`;
  } else {
    $('archive_progress').textContent = '';
  }
}

/* a fetch bar's time: how long it took, or how long it has left at the
   pace it has gone so far (said once a few spectra are in, a first
   estimate being off: four files go at once) */
function barTime(bar, now) {
  if (!bar.started) return bar.total && !bar.done ? ` · ${t('archive_waiting')}` : '';
  if (bar.ended) return ` · ${t('archive_took', took(bar.started, bar.ended))}`;
  const spent = (now || Date.now() / 1000) - bar.started;
  if (bar.done < 4 || spent < 5) return ` · ${t('archive_left', '...')}`;
  // an estimate: to 5 s, then to the minute
  const left = (spent * (bar.total - bar.done)) / bar.done;
  const said = left < 60 ? `${Math.max(5, Math.round(left / 5) * 5)} s` : left < 3600 ? `${Math.round(left / 60)} min` : span(left);
  return ` · ${t('archive_left', said)}`;
}

async function archiveFetch() {
  const rows = ((ARCHIVE || {}).search || {}).groups || [];
  const chosen = rows.filter((row) => archiveChosen.has(row.id));
  const total = chosen.reduce((sum, row) => sum + (row.size_mb || 0), 0);
  // more than a few GB: said, and asked
  if (total > BIG_FETCH_MB && !(await confirmBox(t('archive_big', megabytes(total), S.root)))) return;
  ARCHIVE = await api.post('/api/archive/fetch', { data_dir: S.root, ids: chosen.map((row) => row.id) });
  drawArchive();
  pollArchive();
  // the new folders appear in My targets as their headers are read
  const wait = setInterval(async () => {
    if (!ARCHIVE || ARCHIVE.fetch.state !== 'running') { clearInterval(wait); await loadFolders(); await loadTargets(); }
  }, 2000);
}

// ── instrument and source ───────────────────────────────────────────────
function fillInstruments() {
  const select = $('instrument');
  select.replaceChildren(...INSTRUMENTS.map((row) => el('option', { value: row.instrument, text: row.instrument })));
  if (!S.instrument && INSTRUMENTS.length) S.instrument = INSTRUMENTS[0].instrument;
  select.value = S.instrument;
  fillSources();
}
function sourcesOf(instrument) {
  return (INSTRUMENTS.find((row) => row.instrument === instrument) || { sources: [] }).sources;
}
function fillSources() {
  const sources = sourcesOf(S.instrument);
  $('source').replaceChildren(...sources.map((row) => el('option', { value: row.source, text: row.source })));
  if (!sources.find((row) => row.source === S.source)) {
    // the source the ticked spectra's layout says, else the first
    const said = tickedSource();
    S.source = sources.find((row) => row.source === said) ? said : (sources.length ? sources[0].source : '');
  }
  $('source').value = S.source;
}
/* the data source the ticked targets' files say by their layout
   (targets.layout_source), when they all say the same */
function tickedSource() {
  const said = [...new Set(TARGETS.filter((row) => S.ticked.includes(row.name)).map((row) => row.source || null))];
  return said.length === 1 ? said[0] : null;
}
function instrumentTelluclean() {
  const row = sourcesOf(S.instrument).find((r) => r.source === S.source);
  return row ? row.telluclean : false;
}
/* the LBL instrument a header's INSTRUME most likely is */
function guessInstrument(header) {
  const name = String(header || '').toUpperCase();
  const names = INSTRUMENTS.map((row) => row.instrument);
  if (names.includes(name)) return name;
  if (name.startsWith('NIRPS')) return 'NIRPS_HE';
  if (name === 'HARPS-N' || name === 'HARPN') return 'HARPSN';
  return null;
}

// ── targets ─────────────────────────────────────────────────────────────
async function loadTargets() {
  if (!S.data_dir) { TARGETS = []; SCIENCE = { state: 'none', folder: '' }; drawTargets(); return; }
  const data = await api.get('/api/targets', { data_dir: S.data_dir });
  if (data.error) { $('scan_status').textContent = data.error; return; }
  TARGETS = data.targets || [];
  SCIENCE = { state: data.science || 'ok', folder: data.folder || '', empty: data.empty || [] };
  const scan = data.scan || {};
  // the server reads the headers by itself (one file per folder first):
  //   the page follows while it does
  if (scan.running && !scanTimer) scanTimer = setInterval(loadTargets, 1500);
  if (scan.running && scan.data_dir) {
    $('scan_status').textContent = t('scanning', scan.done, scan.total);
  } else {
    const files = TARGETS.reduce((sum, row) => sum + (row.files || 0), 0);
    $('scan_status').textContent = TARGETS.length ? t('scan_done', TARGETS.length, files) : t('data_dir_hint');
  }
  if (!scan.running && scanTimer) { clearInterval(scanTimer); scanTimer = null; }
  // an instrument from the targets when none was chosen yet
  if (!S.instrument_set) {
    const counts = {};
    TARGETS.forEach((row) => { const g = guessInstrument(row.instrument); if (g) counts[g] = (counts[g] || 0) + row.files; });
    const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    if (best && best[0] !== S.instrument) { S.instrument = best[0]; $('instrument').value = S.instrument; fillSources(); await loadInstrumentThings(); }
  }
  S.ticked = S.ticked.filter((name) => TARGETS.find((row) => row.name === name));
  drawTargets();
  drawPerTarget();
  drawSetupStatus();
  await loadPoints();
}

async function scan() {
  S.root = $('data_dir').value.trim();
  await loadFolders();
  save();
  setBackdrop();
  await scanFolder();
}

/* read the targets of the folder in use (S.data_dir) */
async function scanFolder() {
  // a root without a folder of targets yet: said by the targets' table
  if (LAYOUT && !LAYOUT.legacy && !S.folder) { await loadTargets(); return; }
  const data = await api.post('/api/scan', { data_dir: S.data_dir });
  if (data.error) { $('scan_status').textContent = data.error; return; }
  await loadInstrumentThings();
  if (scanTimer) clearInterval(scanTimer);
  scanTimer = setInterval(loadTargets, 1000);
  loadTargets();
}

// ── the root: its instrument folders, its Downloads ─────────────────────
// The data folder of LBL (DATA_DIR) the window works in is one of the
//   root's, ROOT/{INSTRUMENT}_{SOURCE}, or the root itself when it holds a
//   science/ of its own (the old layout)
async function loadFolders() {
  if (!S.root) { LAYOUT = null; S.data_dir = ''; drawFolders(); return; }
  LAYOUT = await api.get('/api/folders', { root: S.root });
  if (LAYOUT.error) { LAYOUT = null; drawFolders(); return; }
  const folders = LAYOUT.folders || [];
  if (S.folder && !folders.some((f) => f.name === S.folder)) S.folder = '';
  if (!S.folder && !LAYOUT.legacy && folders.length) {
    // the folder of the instrument chosen, else the one with the most targets
    const mine = folders.find((f) => f.instrument === S.instrument && f.source === S.source);
    S.folder = (mine || [...folders].sort((a, b) => b.targets - a.targets)[0]).name;
  }
  S.data_dir = S.folder ? childPath(S.root, S.folder) : S.root;
  drawFolders();
}

function drawFolders() {
  const options = [];
  if (LAYOUT && LAYOUT.legacy) options.push(el('option', { value: '', text: t('folder_this', LAYOUT.legacy_targets, LAYOUT.legacy_spectra) }));
  (LAYOUT ? LAYOUT.folders : []).forEach((f) => options.push(el('option', {
    value: f.name, text: f.targets ? t('folder_row', f.name, f.targets, f.spectra) : t('folder_empty', f.name) })));
  if (!options.length) options.push(el('option', { value: '', text: t('folder_none') }));
  $('data_folder').replaceChildren(...options);
  $('data_folder').value = S.folder;
  $('data_folder').disabled = !LAYOUT || (!LAYOUT.legacy && !LAYOUT.folders.length);
  const down = LAYOUT ? LAYOUT.downloads : null;
  $('downloads_label').textContent = t('downloads_n', down ? down.files : 0);
  $('downloads').title = down ? t('downloads_title', down.path) : '';
  $('downloads').disabled = !S.root;
  drawTree();
  drawDispatch();
}

// The root as a tree: what waits in Downloads (by name: its headers are
//   read when it is dispatched), each instrument folder (its targets and
//   their spectra, its empty folders, calib, models), the shared models,
//   the old layout's science/ and the rest
function drawTree() {
  const box = $('root_tree');
  box.hidden = !LAYOUT || !LAYOUT.exists;
  if (box.hidden) return;
  const L = LAYOUT;
  const sum = (key) => L.folders.reduce((n, f) => n + f[key], 0);
  $('root_tree_summary').textContent = t('tree_summary', L.downloads.files, L.folders.length, sum('targets'), sum('spectra'));
  const name = (text) => el('span', { class: 'tree-name', text });
  const note = (text) => el('span', { class: 'tree-note', text });
  const line = (deep, ...kids) => el('div', { class: deep ? 'tree-line deep' : 'tree-line' }, ...kids);
  const count = (n) => el('span', { class: 'tree-count', text: n });
  const node = (key, head, ...kids) => {
    const box = el('details', { class: `tree-node${key === `${S.folder}/` ? ' current' : ''}`, open: TREE_OPEN.has(key) || null }, el('summary', {}, ...head), ...kids);
    box.addEventListener('toggle', () => { if (box.open) TREE_OPEN.add(key); else TREE_OPEN.delete(key); });
    return box;
  };
  const d = L.downloads;
  const nodes = [node('Downloads/', [name('Downloads/'), note(d.files ? t('tree_downloads', d.files, human(d.size)) : t('tree_downloads_none'))],
    ...d.kinds.map((k) => line(false, count(k.n), name(k.kind))),
    d.unused ? line(false, name('unused/'), note(t('tree_unused', d.unused))) : null)];
  L.folders.forEach((f) => {
    const here = f.name === S.folder;
    nodes.push(node(`${f.name}/`, [name(`${f.name}/`), note(f.targets ? t('tree_home', f.targets, f.spectra) : t('tree_home_empty')),
      here ? el('span', { class: 'tree-here good-text', text: `● ${t('tree_current')}` })
        : el('button', { type: 'button', class: 'button tree-open', text: t('tree_open'), onclick: (event) => { event.preventDefault(); S.folder = f.name; openFolder(); } })],
    line(false, name('science/')),
    ...f.list.map((one) => line(true, count(one.files), name(`${one.name}/`))),
    f.empty.length ? line(true, note(t('tree_empty', f.empty.length, f.empty.join(', ')))) : null,
    line(false, name('calib/'), note(t('tree_calib', f.calib))),
    line(false, name('models'), note(t(f.models === 'merge' ? 'tree_models_own' : f.models === 'link' ? 'tree_models_none' : 'tree_models_link')))));
  });
  if (L.models.exists) nodes.push(line(false, name('models/'), note(t('tree_models', L.models.files, human(L.models.size)))));
  // the root as the data folder it was (one for every instrument)
  const old = L.science;
  const lblFolders = L.old_folders || [];
  if (old || lblFolders.length) {
    nodes.push(node('(root)', [name(t('tree_old')), note(t('tree_old_note'))],
      old ? line(false, name('science/'), note(old.targets.length ? t('tree_home', old.targets.length, old.spectra) : t('tree_legacy_empty', old.empty.length))) : null,
      ...(old ? old.targets.map((one) => line(true, count(one.files), name(`${one.name}/`))) : []),
      old && old.targets.length && old.empty.length ? line(true, note(t('tree_empty', old.empty.length, old.empty.join(', ')))) : null,
      lblFolders.length ? line(false, note(t('tree_old_folders', lblFolders.join(' ')))) : null));
  }
  if (L.others.length) nodes.push(line(false, note(t('tree_others', L.others.join(' ')))));
  $('root_tree_body').replaceChildren(...nodes);
}

/* the folder chosen: its instrument and data source are the wrap's, and
   every tab shows that folder's (targets, runs, tables, cleanup) */
async function openFolder() {
  S.data_dir = S.folder ? childPath(S.root, S.folder) : S.root;
  const folder = LAYOUT && LAYOUT.folders.find((f) => f.name === S.folder);
  if (folder) {
    S.instrument = folder.instrument;
    S.instrument_set = true;
    S.setup_manual = false;
    $('instrument').value = S.instrument;
    fillSources();
    S.source = folder.source;
    $('source').value = S.source;
  }
  S.ticked = [];
  save();
  wrapEdited = false;
  await scanFolder();
  showTab(S.tab);
}

// the files of Downloads into the root's folders: a plan read from their
//   headers first (dispatch.py), shown, then the files moved when asked
async function startDispatch() {
  DISPATCH = await api.post('/api/dispatch/plan', { root: S.root });
  followDispatch();
}

function followDispatch() {
  drawDispatch();
  if (dispatchTimer) return;
  dispatchTimer = setInterval(async () => {
    DISPATCH = await api.get('/api/dispatch');
    if (!['planning', 'moving'].includes(DISPATCH.state)) {
      clearInterval(dispatchTimer);
      dispatchTimer = null;
      if (DISPATCH.state === 'done') {
        await loadFolders();
        await openFolder();
      }
    }
    drawDispatch();
  }, 700);
}

function dispatchNote(text) { return el('li', { text }); }

function drawDispatch() {
  const d = DISPATCH || { state: 'idle' };
  const busy = ['planning', 'moving'].includes(d.state);
  const down = LAYOUT ? LAYOUT.downloads : null;
  $('dispatch').disabled = busy || !down || !down.files;
  const step = d.step === 'simbad' ? t('dispatch_simbad', d.done, d.total)
    : d.step === 'folders' ? t('dispatch_looking') : t('dispatch_headers', d.done, d.total);
  $('dispatch_status').textContent = d.state === 'planning' ? step
    : d.state === 'moving' ? t('dispatch_moving', d.done, d.total)
      : d.state === 'error' ? t('dispatch_error', d.error) : '';
  const panel = $('dispatch_panel');
  panel.hidden = !['planned', 'done'].includes(d.state);
  if (panel.hidden) return;
  const notes = [];
  if (d.state === 'planned') {
    const p = d.plan;
    const work = p.science + p.calib + p.n_unused + p.shares.length;
    $('dispatch_summary').textContent = p.files ? t('dispatch_plan', p.files, p.science, p.folders.length, p.calib, p.n_unused, p.n_left) : t('dispatch_nothing');
    let last = null;
    const rows = p.folders.map((r) => {
      const first = r.star !== last;
      last = r.star;
      return el('tr', { class: first ? 'star-first' : null },
        el('td', { class: 'mono', text: `${r.home}/science/${r.folder}` }),
        el('td', { class: 'num', text: r.science }),
        el('td', { class: r.new ? 'good-text small' : 'muted small', text: t(r.new ? 'dispatch_new' : 'dispatch_existing') }),
        el('td', { class: 'small', text: r.names.join(', ') }));
    });
    $('dispatch_table').replaceChildren(
      el('thead', {}, el('tr', {}, ...['col_dispatch_folder', 'col_dispatch_files', null, 'col_dispatch_names'].map((key) => el('th', { class: key === 'col_dispatch_files' ? 'num' : null, text: key ? t(key) : '' })))),
      el('tbody', {}, ...rows));
    $('dispatch_wrap').hidden = !rows.length;
    p.calibs.forEach((c) => notes.push(dispatchNote(t('dispatch_calib', c.n, c.home))));
    // the models of the instrument folders, the root's: one line a kind
    ['link', 'merge'].forEach((state) => {
      const homes = p.shares.filter((one) => one.state === state).map((one) => one.home);
      if (homes.length) notes.push(dispatchNote(t(`dispatch_share_${state}`, homes.join(', '))));
    });
    (p.defaults || []).forEach((home) => notes.push(dispatchNote(t('dispatch_default_blaze', home))));
    p.unused.forEach((u) => notes.push(dispatchNote(t('dispatch_unused', u.n, u.why, u.example))));
    p.left.forEach((u) => notes.push(dispatchNote(t('dispatch_left', u.n, u.why, u.example))));
    p.warnings.forEach((w) => notes.push(el('li', { class: 'warn-text', text: t(`dispatch_${w.key}`, ...w.args) })));
    $('dispatch_go').hidden = !work;
    $('dispatch_cancel').textContent = t('dispatch_cancel');
  } else {
    const r = d.result || {};
    $('dispatch_summary').textContent = t('dispatch_done', r.moved || 0, r.copies || 0, r.unused || 0);
    $('dispatch_wrap').hidden = true;
    if ((r.made || []).length) notes.push(dispatchNote(t('dispatch_made', r.made.join(', '))));
    if ((r.defaults || []).length) notes.push(dispatchNote(t('dispatch_defaults', r.defaults.join(', '))));
    const linked = [];
    Object.entries(r.shared || {}).forEach(([home, done]) => Object.values(done).forEach((how) => {
      if (how && how.kept) notes.push(el('li', { class: 'warn-text', text: t('dispatch_share_kept', home, how.kept.join(', ')) }));
      else if (how === 'linked' || how === 'merged') linked.push(home);
      else notes.push(el('li', { class: 'warn-text', text: `${home}/models: ${how}` }));
    }));
    if (linked.length) notes.push(dispatchNote(t('dispatch_shared', linked.join(', '))));
    if ((r.kept || []).length) notes.push(el('li', { class: 'warn-text', text: t('dispatch_kept', r.kept.map((k) => k.name).join(', ')) }));
    if ((r.failed || []).length) notes.push(el('li', { class: 'warn-text', text: t('dispatch_failed', r.failed.map((f) => `${f.name} (${f.why})`).join(', ')) }));
    $('dispatch_go').hidden = true;
    $('dispatch_cancel').textContent = t('dispatch_close');
  }
  $('dispatch_notes').replaceChildren(...notes);
}

const COLUMNS = [
  { key: 'name', label: 'col_target' },
  { key: 'instrument', label: 'col_instrument' },
  { key: 'files', label: 'col_files', num: true },
  { key: 'nights', label: 'col_nights', num: true },
  { key: 'snr', label: 'col_snr', num: true, digits: 0 },
  { key: 'exptime', label: 'col_exptime', num: true, digits: 0 },
  { key: 'berv_min', label: 'col_berv', num: true },
  { key: 'berv_span', label: 'col_span', num: true },
  { key: 'first', label: 'col_dates' },
  { key: 'teff', label: 'col_teff', num: true, digits: 0 },
];

function colourOf(name) {
  const index = S.ticked.indexOf(name);
  if (index < 0) return null;
  return index < HIST_COLOURED ? css(SERIES[index]) : css('--series-other');
}

function drawTargets() {
  // no folder to read: said instead of an empty table
  const message = !S.data_dir ? t('no_data_dir')
    : SCIENCE.state === 'missing' ? (LAYOUT && !LAYOUT.legacy && !S.folder ? t('no_folder', S.root) : t('no_science', SCIENCE.folder))
      : SCIENCE.state === 'empty' ? t('empty_science', SCIENCE.folder) : '';
  $('targets_message').textContent = message;
  $('targets_message').hidden = !message;
  // folders without spectra are no targets: said under the table
  const empty = SCIENCE.empty || [];
  $('targets_empty').textContent = empty.length ? t('targets_empty_hidden', empty.length, empty.join(', ')) : '';
  $('targets_wrap').hidden = Boolean(message);
  // an instrument folder holds one instrument, read from one source (the
  //   menu above chose it): no tick per instrument, no column of it; the
  //   old layout's science/ may mix them
  const one = Boolean(S.folder);
  const instruments = one ? [] : [...new Set(TARGETS.map((row) => row.instrument))].sort();
  $('instrument_filters').hidden = one;
  $('instrument_filters').replaceChildren(...instruments.map((name) => el('label', { class: 'check' },
    el('input', {
      type: 'checkbox', checked: !S.hidden_instruments.includes(name),
      onchange: (event) => {
        S.hidden_instruments = event.target.checked ? S.hidden_instruments.filter((n) => n !== name) : [...S.hidden_instruments, name];
        save(); drawTargets();
      },
    }), name)));
  const rows = TARGETS.filter((row) => one || !S.hidden_instruments.includes(row.instrument));
  const columns = COLUMNS.filter((col) => !(one && col.key === 'instrument'));
  const { key, dir } = S.sort;
  rows.sort((a, b) => {
    const va = a[key];
    const vb = b[key];
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    return (typeof va === 'string' ? va.localeCompare(vb) : va - vb) * dir;
  });
  const head = el('tr', {}, el('th', {}, ''), ...columns.map((col) => el('th', {
    class: 'sortable', text: t(col.label) + (S.sort.key === col.key ? (S.sort.dir > 0 ? ' ▴' : ' ▾') : ''),
    onclick: () => { S.sort = { key: col.key, dir: S.sort.key === col.key ? -S.sort.dir : 1 }; save(); drawTargets(); },
  })));
  const body = rows.map((row) => {
    const ticked = S.ticked.includes(row.name);
    const colour = colourOf(row.name);
    const reading = (row.read || 0) < row.files;
    // a target whose headers are not all read yet: its numbers are
    //   provisional (greyed), its counts left out until they are
    return el('tr', { class: [ticked ? 'ticked' : '', reading ? 'reading' : ''].join(' ').trim(), ondblclick: () => toggle(row.name) },
      el('td', {}, el('input', { type: 'checkbox', checked: ticked, onchange: () => toggle(row.name) })),
      el('td', {}, colour ? el('span', { class: 'swatch', style: `background:${colour}` }) : null, row.name),
      one ? null : el('td', { text: row.instrument }),
      el('td', { class: 'num', 'data-tip': reading ? t('read_of', row.read || 0, row.files) : null },
        reading ? el('span', { class: 'muted', text: '… ' }) : null, String(row.files)),
      el('td', { class: 'num', text: reading ? '' : (row.nights || '') }),
      el('td', { class: 'num', text: fmt(row.snr, 0) }),
      el('td', { class: 'num', text: fmt(row.exptime, 0) }),
      el('td', { class: 'num', text: row.berv_min === null ? '' : `${fmt(row.berv_min)} → ${fmt(row.berv_max)}` }),
      el('td', { class: 'num', text: reading ? '' : fmt(row.berv_span) }),
      el('td', { class: 'mono', text: row.first ? `${mjdDate(row.first).slice(0, 7)} → ${mjdDate(row.last).slice(0, 7)}` : '', 'data-tip': row.first ? `${mjdDate(row.first)} → ${mjdDate(row.last)}` : null }),
      el('td', { class: 'num', text: fmt(row.teff, 0) }));
  });
  $('targets_table').replaceChildren(el('thead', {}, head), el('tbody', {}, body));
  $('ticked').textContent = t('ticked', S.ticked.length, TARGETS.length);
  drawTiles();
  // TICKED TARGETS OF MORE THAN ONE INSTRUMENT: said loudly, at once (a
  //   wrap runs one instrument)
  const byInstrument = {};
  TARGETS.filter((row) => S.ticked.includes(row.name) && row.instrument && row.instrument !== '?')
    .forEach((row) => { (byInstrument[family(row.instrument)] = byInstrument[family(row.instrument)] || []).push(row.name); });
  const families = Object.keys(byInstrument);
  // one button per instrument to keep its targets and untick the others;
  //   first the instrument of the target ticked last, the likely intent
  const last = [...S.ticked].reverse().map((name) => TARGETS.find((row) => row.name === name))
    .find((row) => row && row.instrument && row.instrument !== '?');
  const proposed = last ? family(last.instrument) : null;
  const choices = [...families].sort((a, b) => (b === proposed) - (a === proposed));
  $('target_errors').replaceChildren(...(families.length > 1 ? [el('div', { class: 'message error' },
    el('strong', { text: t('err_mixed_title') }), ' ',
    t('err_mixed', families.map((f) => `${f} : ${byInstrument[f].join(', ')}`).join(' · ')), ' ',
    ...choices.map((f) => el('button', { type: 'button', class: 'button small', text: t('keep_family', f),
      onclick: () => { S.ticked = S.ticked.filter((name) => byInstrument[f].includes(name)); ticksChanged(); } })))] : []));
  $('target_notes').textContent = S.ticked.length > 1 && families.length <= 1 ? t('joint_note') : '';
}

function drawTiles() {
  const ticked = TARGETS.filter((row) => S.ticked.includes(row.name));
  const sum = (rows, key) => rows.reduce((total, row) => total + (row[key] || 0), 0);
  const count = (x) => Number(x).toLocaleString(S.lang === 'fr' ? 'fr-CA' : 'en-CA');
  const span = (rows) => {
    const firsts = rows.map((row) => row.first).filter((x) => x !== null && x !== undefined);
    const lasts = rows.map((row) => row.last).filter((x) => x !== null && x !== undefined);
    return firsts.length ? `${mjdDate(Math.min(...firsts))} → ${mjdDate(Math.max(...lasts))}` : '';
  };
  // THE SELECTION ONLY: what is ticked, not the whole folder (which is said
  //   in small when nothing is)
  const bervs = ticked.filter((row) => row.berv_min !== null && row.berv_min !== undefined);
  const lo = bervs.length ? Math.min(...bervs.map((row) => row.berv_min)) : null;
  const hi = bervs.length ? Math.max(...bervs.map((row) => row.berv_max)) : null;
  const instruments = [...new Set(ticked.map((row) => row.instrument).filter(Boolean))].sort();
  // the nights and SNR of the ticked spectra themselves (POINTS), the
  //   targets' own numbers until they are read. The nights are the
  //   server's (targets.night_of: cut at local noon at the observatory of
  //   each spectrum's instrument), never cut here
  const nightsOf = ticked.flatMap((row) => ((POINTS[row.name] || {}).night || []).filter((n) => n !== null && n !== undefined));
  const nights = nightsOf.length ? new Set(nightsOf).size : sum(ticked, 'nights');
  const snrs = ticked.flatMap((row) => ((POINTS[row.name] || {}).snr || []).filter((v) => v !== null && v !== undefined));
  const median = (values) => { const v = [...values].sort((a, b) => a - b); return v.length ? v[Math.floor(v.length / 2)] : null; };
  const snr = median(snrs.length ? snrs : ticked.map((row) => row.snr).filter((v) => v !== null && v !== undefined));
  const none = !ticked.length;
  const tile = (label, value, sub) => el('div', { class: 'tile' },
    el('div', { class: 'label', text: label }),
    el('div', { class: 'value', text: value }),
    el('div', { class: 'sub', text: sub, title: sub }));
  $('tiles').replaceChildren(
    tile(t('tile_targets'), count(ticked.length), none ? t('tile_folder', count(TARGETS.length), count(sum(TARGETS, 'files'))) : instruments.join(' · ')),
    tile(t('tile_spectra'), none ? '-' : count(sum(ticked, 'files')), none ? t('tile_none') : span(ticked)),
    tile(t('tile_nights'), none ? '-' : count(nights), none || snr === null ? t('tile_none') : t('tile_snr', fmt(snr, 0))),
    tile(t('tile_berv'), lo === null ? '-' : `${fmt(hi - lo)} km/s`, lo === null ? t('tile_none') : `${fmt(lo)} → ${fmt(hi)} km/s`));
}

function toggle(name) {
  S.ticked = S.ticked.includes(name) ? S.ticked.filter((n) => n !== name) : [...S.ticked, name];
  ticksChanged();
}

// what follows a change of the ticked targets
function ticksChanged() {
  save();
  drawTargets();
  loadPoints();
  drawPerTarget();
  drawOptions();
  wrapEdited = false;
  followTicked();
}

// the instrument follows the ticked targets when they are all of one other
//   family than the one chosen: spectra run as another instrument make a
//   template of the wrong one
const family = (name) => String(name || '').toUpperCase().split('_')[0].replace(/[^A-Z0-9]/g, '');
// ── the instrument and the data source: detected, or chosen by hand ─────
// What the ticked targets' files say: their archive's reader when fetched,
//   else the headers' instrument and the files' layout (targets.py). null
//   where nothing is ticked; instrument null when the ticked disagree
function detectedSetup() {
  const ticked = TARGETS.filter((row) => S.ticked.includes(row.name));
  if (!ticked.length) return null;
  const from = ticked.map((row) => row.name);
  const readers = [...new Set(ticked.map((row) => (row.archive ? `${row.archive.lbl}|${row.archive.source}` : null)))];
  if (readers.length === 1 && readers[0]) {
    const [instrument, source] = readers[0].split('|');
    return { instrument, source, from };
  }
  const found = [...new Set(ticked.map((row) => guessInstrument(row.instrument)))];
  if (found.length !== 1 || !found[0]) return { instrument: null, source: null, from };
  const said = tickedSource();
  const source = said && sourcesOf(found[0]).some((row) => row.source === said) ? said : null;
  return { instrument: found[0], source, from };
}

// The detection applied, unless the instrument and the source were chosen
//   by hand (S.setup_manual) for an instrument of the same family as the
//   files: a NIRPS mode, a source the files do not tell. A choice of
//   another family cannot read the files and gives way, said so
async function followTicked() {
  const d = detectedSetup();
  if (!d || !d.instrument) { drawSetupStatus(); return; }
  if (S.setup_manual && family(S.instrument) === family(d.instrument)) { drawSetupStatus(); return; }
  const before = `${S.instrument}|${S.source}`;
  if (S.setup_manual) $('target_notes').textContent = t('setup_overruled', S.instrument, d.instrument);
  S.setup_manual = false;
  if (family(d.instrument) !== family(S.instrument) || d.instrument !== S.instrument) {
    S.instrument = d.instrument;
    S.instrument_set = true;
    $('instrument').value = S.instrument;
    fillSources();
  }
  if (d.source && d.source !== S.source) { S.source = d.source; $('source').value = d.source; }
  if (`${S.instrument}|${S.source}` !== before) {
    save();
    await loadInstrumentThings();
    $('target_notes').textContent = t('instrument_followed', `${S.instrument} · ${S.source}`);
  }
  drawSetupStatus();
}

// What is in force, said in one line under the fields: detected (and from
//   which targets), or chosen by hand (and what the files say), and the
//   Python: the window's own or another
function drawSetupStatus() {
  const box = $('lbl_setup_status');
  if (!box) return;
  const d = detectedSetup();
  const now = `${S.instrument} · ${S.source}`;
  // an instrument folder sets them: shown, not chosen here
  const folder = LAYOUT && S.folder ? LAYOUT.folders.find((f) => f.name === S.folder) : null;
  const manual = Boolean(S.setup_manual) && !folder;
  $('instrument').disabled = Boolean(folder);
  $('source').disabled = Boolean(folder);
  $('instrument').classList.toggle('set', manual);
  $('source').classList.toggle('set', manual);
  let line;
  if (folder) {
    line = el('span', { class: 'good-text', text: `✓ ${t('setup_folder', now, folder.name)}` });
  } else if (!d) {
    line = el('span', { class: 'muted', text: t('setup_none') });
  } else if (!d.instrument) {
    line = el('span', { class: 'warn-text', text: t('setup_mixed') });
  } else if (manual) {
    line = el('span', { class: 'warn-text' }, t('setup_manual', now, `${d.instrument} · ${d.source || '?'}`), ' ',
      el('button', { type: 'button', class: 'button', text: t('setup_reset'), onclick: async () => { S.setup_manual = false; save(); await followTicked(); } }));
  } else {
    const names = d.from.length > 3 ? `${d.from.slice(0, 3).join(', ')}...` : d.from.join(', ');
    line = el('span', { class: 'good-text' }, `✓ ${t('setup_auto', names, now)}`,
      d.source ? '' : ` ${t('setup_unknown', S.source)}`);
  }
  const python = S.python && S.python !== DEFAULT_PYTHON;
  $('python').classList.toggle('set', python);
  box.replaceChildren(line, el('div', { class: 'muted small' }, python ? t('python_manual') : t('python_auto'),
    python ? el('button', { type: 'button', class: 'reset', title: t('reset'), text: '↺', onclick: () => { S.python = DEFAULT_PYTHON; $('python').value = S.python; save(); drawSetupStatus(); wrapEdited = false; } }) : null));
}

async function loadPoints() {
  if (!S.ticked.length || !S.data_dir) { POINTS = {}; drawPlots(); return; }
  POINTS = await api.get('/api/points', { data_dir: S.data_dir, names: S.ticked.join(',') });
  drawPlots();
  drawTiles();
}

// ── plots (SVG) ─────────────────────────────────────────────────────────
const NS = 'http://www.w3.org/2000/svg';
function svg(tag, attrs) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs || {}).forEach(([k, v]) => { if (v !== null && v !== undefined) node.setAttribute(k, v); });
  return node;
}
function ticks(lo, hi, count) {
  const span = hi - lo || 1;
  const step = 10 ** Math.floor(Math.log10(span / count));
  const nice = [1, 2, 5, 10].map((m) => m * step).find((s) => span / s <= count) || step * 10;
  const out = [];
  for (let v = Math.ceil(lo / nice) * nice; v <= hi + 1e-9; v += nice) out.push(Number(v.toFixed(10)));
  return out;
}
function frame(width, height, margin) {
  const root = svg('svg', { viewBox: `0 0 ${width} ${height}` });
  const inner = { x0: margin.l, x1: width - margin.r, y0: margin.t, y1: height - margin.b };
  return { root, inner };
}
function axes(root, inner, xs, ys, xscale, yscale, xlabel, xfmt) {
  const grid = svg('g', { class: 'grid' });
  ys.forEach((v) => grid.append(svg('line', { x1: inner.x0, x2: inner.x1, y1: yscale(v), y2: yscale(v) })));
  root.append(grid);
  const axis = svg('g', { class: 'axis' });
  axis.append(svg('line', { x1: inner.x0, x2: inner.x1, y1: inner.y1, y2: inner.y1 }));
  xs.forEach((v) => {
    axis.append(svg('line', { x1: xscale(v), x2: xscale(v), y1: inner.y1, y2: inner.y1 + 4 }));
    const label = svg('text', { x: xscale(v), y: inner.y1 + 15, 'text-anchor': 'middle' });
    label.textContent = xfmt ? xfmt(v) : v;
    axis.append(label);
  });
  ys.forEach((v) => {
    const label = svg('text', { x: inner.x0 - 5, y: yscale(v) + 3, 'text-anchor': 'end' });
    label.textContent = v;
    axis.append(label);
  });
  if (xlabel) {
    const label = svg('text', { x: (inner.x0 + inner.x1) / 2, y: inner.y1 + 29, 'text-anchor': 'middle' });
    label.textContent = xlabel;
    axis.append(label);
  }
  root.append(axis);
}
function empty(holder, message) {
  const { root } = frame(360, 80, { l: 0, r: 0, t: 0, b: 0 });
  const text = svg('text', { x: 180, y: 44, 'text-anchor': 'middle', class: 'empty' });
  text.textContent = message || t('no_points');
  root.append(text);
  holder.replaceChildren(root);
}
// a target's colour is its place among the ticked ones, in every plot
// a legend for two targets or more, and for one when the targets are split
//   between the plots against v_tot and those against the BERV (SPLIT)
let SPLIT = false;
function legend(names, coloured) {
  if (names.length < 2 && !SPLIT) return null;
  const place = (n) => S.ticked.indexOf(n);
  const shown = names.filter((n) => place(n) < coloured);
  const rest = names.length - shown.length;
  const items = shown.map((name) => el('span', {}, el('span', { class: 'swatch', style: `background:${css(SERIES[place(name)])}` }), name));
  if (rest) items.push(el('span', {}, el('span', { class: 'swatch', style: `background:${css('--series-other')}` }), `${t('other_targets')} (${rest})`));
  return el('div', { class: 'legend' }, items);
}
// the bounds are in RJD (BJD - 2400000), as LBL reads them; the headers give MJD, half a day less
function inBounds(mjd) {
  const rjd = mjd + 0.5;
  return (S.min_rjd === null || S.min_rjd === '' || rjd >= S.min_rjd) && (S.max_rjd === null || S.max_rjd === '' || rjd <= S.max_rjd);
}

// a target is drawn against v_tot, the star's velocity in the telluric
//   frame, when its systemic velocity is known (SIMBAD's, else the
//   headers': targets.py's _vsys), against the BERV otherwise: when the
//   ticked targets are of both kinds, each plot comes twice, one per kind
// the speed of light [km/s], astropy's, sent by the server (/api/state)
let C_KMS = null;
// |v_tot| under which the star's lines sit on the tellurics (km/s)
const VTOT_CLOSE = 4;
function vsysOf(name, i) {
  const rv = (SIMBAD[name] || {}).rv;
  if (typeof rv === 'number') return rv;
  const header = (POINTS[name].vsys || [])[i];
  return header === null || header === undefined ? null : header;
}
const hasVsys = (name) => POINTS[name].berv.some((_, i) => vsysOf(name, i) !== null);
// v_sys less the BERV, composed relativistically
const vtotOf = (vsys, berv) => (vsys - berv) / (1 - (vsys * berv) / (C_KMS * C_KMS));
const showCard = (id, on) => { $(id).closest('.card').hidden = !on; };

function drawPlots() {
  const names = S.ticked.filter((name) => POINTS[name] && POINTS[name].berv.length);
  // every target against the BERV; those whose systemic velocity is known
  //   against v_tot as well, SNR on the y axis in both
  const withV = names.filter(hasVsys);
  // a legend even for one target when not every target is on every plot
  SPLIT = withV.length > 0 && withV.length < names.length;
  showCard('plot_vtot', withV.length > 0);
  if (!names.length) { empty($('plot_berv')); empty($('plot_snr')); empty($('plot_time')); return; }
  histogram($('plot_berv'), names, (n) => POINTS[n].berv.map((v, i) => [v, POINTS[n].mjd[i]]), 'BERV (km/s)', 'BERV');
  drawScatter($('plot_snr'), names, (n, i) => POINTS[n].berv[i], 'BERV (km/s)', 'BERV', false);
  if (withV.length) {
    const vtot = (n, i) => { const v = vsysOf(n, i); return v === null ? null : vtotOf(v, POINTS[n].berv[i]); };
    drawScatter($('plot_vtot'), withV, vtot, 'v_tot = v_sys - BERV (km/s)', 'v_tot', true);
  }
  drawTimeline(names);
}

// the band where the star's lines sit on the tellurics, |v_tot| < 4 km/s,
//   and the zero line, behind the marks
function tellurics(root, inner, xscale) {
  const x0 = Math.max(inner.x0, xscale(-VTOT_CLOSE));
  const x1 = Math.min(inner.x1, xscale(VTOT_CLOSE));
  if (x1 > x0) root.append(svg('rect', { x: x0, width: x1 - x0, y: inner.y0, height: inner.y1 - inner.y0, fill: css('--tellu-band') }));
  const z = xscale(0);
  if (z > inner.x0 && z < inner.x1) root.append(svg('line', { x1: z, x2: z, y1: inner.y0, y2: inner.y1, stroke: css('--muted'), 'stroke-dasharray': '3 3' }));
}

// a histogram stacked by target, per 3 km/s; each target keeps its colour
//   (its place among the ticked ones), past the coloured slots one grey
function histogram(holder, names, valuesOf, xlabel, what, band = false) {
  const width = 380;
  const height = 190;
  const { root, inner } = frame(width, height, { l: 30, r: 8, t: 6, b: 36 });
  const values = Object.fromEntries(names.map((n) => [n, valuesOf(n)]));
  const all = names.flatMap((n) => values[n].map((pair) => pair[0]));
  const bin = 3;
  const lo = Math.floor(Math.min(...all) / bin) * bin;
  const hi = Math.ceil(Math.max(...all) / bin) * bin + (Math.max(...all) % bin === 0 ? bin : 0);
  const bins = Math.max(1, Math.round((hi - lo) / bin));
  const slot = (n) => Math.min(S.ticked.indexOf(n), HIST_COLOURED);
  const counts = {};
  names.forEach((n) => {
    const s = slot(n);
    counts[s] = counts[s] || new Array(bins).fill(0);
    values[n].forEach(([v, mjd]) => { if (inBounds(mjd)) counts[s][Math.min(bins - 1, Math.floor((v - lo) / bin))] += 1; });
  });
  const slots = Object.keys(counts).map(Number).sort((a, b) => a - b);
  const colourOfSlot = (s) => (s < HIST_COLOURED ? css(SERIES[s]) : css('--series-other'));
  const totals = new Array(bins).fill(0);
  slots.forEach((s) => counts[s].forEach((v, i) => { totals[i] += v; }));
  const top = Math.max(1, ...totals);
  const xscale = (v) => inner.x0 + ((v - lo) / (hi - lo)) * (inner.x1 - inner.x0);
  const yscale = (v) => inner.y1 - (v / top) * (inner.y1 - inner.y0);
  axes(root, inner, ticks(lo, hi, 6), ticks(0, top, 4), xscale, yscale, xlabel);
  if (band) tellurics(root, inner, xscale);
  const gap = 1;
  for (let i = 0; i < bins; i += 1) {
    let base = 0;
    const x0 = xscale(lo + i * bin) + gap;
    const x1 = xscale(lo + (i + 1) * bin) - gap;
    slots.forEach((s) => {
      const c = counts[s][i];
      if (!c) return;
      root.append(svg('rect', { x: x0, width: Math.max(1, x1 - x0), y: yscale(base + c), height: Math.max(0, yscale(base) - yscale(base + c) - (base ? 1 : 0)), fill: colourOfSlot(s), rx: 1.5 }));
      base += c;
    });
    const lines = names.map((n) => {
      const count = values[n].filter(([v, mjd]) => inBounds(mjd) && Math.min(bins - 1, Math.floor((v - lo) / bin)) === i).length;
      return count ? `${esc(n)}: ${count}` : null;
    }).filter(Boolean);
    const hit = svg('rect', { class: 'hit', x: xscale(lo + i * bin), width: xscale(lo + bin) - xscale(lo), y: inner.y0, height: inner.y1 - inner.y0 });
    hit.dataset.tip = `<span class="mono">${what} ${lo + i * bin} → ${lo + (i + 1) * bin} km/s</span><br>${t('n_spectra', totals[i])}${lines.length > 1 ? `<br>${lines.join('<br>')}` : ''}`;
    root.append(hit);
  }
  holder.replaceChildren(...[legend(names, HIST_COLOURED), root].filter(Boolean));
}

function drawScatter(holder, names, xOf, xlabel, what, band) {
  const width = 380;
  const height = 200;
  const { root, inner } = frame(width, height, { l: 36, r: 8, t: 6, b: 36 });
  const xs = [];
  const ys = [];
  names.forEach((n) => POINTS[n].berv.forEach((_, i) => { const x = xOf(n, i); if (POINTS[n].snr[i] !== null && x !== null) { xs.push(x); ys.push(POINTS[n].snr[i]); } }));
  if (!xs.length) { empty(holder); return; }
  const xlo = Math.min(...xs) - 1;
  const xhi = Math.max(...xs) + 1;
  const yhi = Math.max(...ys) * 1.05;
  const xscale = (v) => inner.x0 + ((v - xlo) / (xhi - xlo)) * (inner.x1 - inner.x0);
  const yscale = (v) => inner.y1 - (v / yhi) * (inner.y1 - inner.y0);
  axes(root, inner, ticks(xlo, xhi, 6), ticks(0, yhi, 4), xscale, yscale, xlabel);
  if (band) tellurics(root, inner, xscale);
  // each target keeps its colour, its place among the ticked ones (the
  //   same in every plot); grey first, so that the coloured three sit on top
  const place = (n) => S.ticked.indexOf(n);
  const order = [...names].sort((a, b) => (place(b) >= SCATTER_COLOURED) - (place(a) >= SCATTER_COLOURED));
  order.forEach((n) => {
    const s = place(n);
    const colour = s < SCATTER_COLOURED ? css(SERIES[s]) : css('--series-other');
    POINTS[n].berv.forEach((_, i) => {
      const snr = POINTS[n].snr[i];
      const x = xOf(n, i);
      if (snr === null || x === null) return;
      const dot = svg('circle', { cx: xscale(x), cy: yscale(snr), r: 3.6, fill: colour, stroke: css('--surface'), 'stroke-width': 1, opacity: inBounds(POINTS[n].mjd[i]) ? 1 : 0.25 });
      dot.dataset.tip = `${esc(n)}<br><span class="mono">${what} ${fmt(x, 2)} km/s · SNR ${fmt(snr, 0)} · ${mjdDate(POINTS[n].mjd[i])}</span>`;
      root.append(dot);
    });
  });
  holder.replaceChildren(...[legend(names, SCATTER_COLOURED), root].filter(Boolean));
}

function drawTimeline(names) {
  const width = 380;
  const rowHeight = 14;
  const height = 40 + rowHeight * names.length;
  const { root, inner } = frame(width, height, { l: 8, r: 8, t: 4, b: 34 });
  const all = names.flatMap((n) => POINTS[n].mjd.filter((m) => m !== null));
  if (!all.length) { empty($('plot_time')); return; }
  const lo = Math.min(...all) - 5;
  const hi = Math.max(...all) + 5;
  const xscale = (v) => inner.x0 + ((v - lo) / (hi - lo)) * (inner.x1 - inner.x0);
  const years = [];
  for (let y = Number(mjdDate(lo).slice(0, 4)); y <= Number(mjdDate(hi).slice(0, 4)) + 1; y += 1) {
    const mjd = (Date.UTC(y, 0, 1) / 86400000) + 40587;
    if (mjd >= lo && mjd <= hi) years.push(mjd);
  }
  // every year when they fit, else every 2nd, 5th or 10th: a label is
  //   about 32 px wide (two decades of labels would run into one another)
  const room = inner.x1 - inner.x0;
  const every = [1, 2, 5, 10, 20].find((k) => (years.length / k) * 32 <= room) || 50;
  const shown = years.filter((m) => Number(mjdDate(m).slice(0, 4)) % every === 0);
  const marks = years.length >= 2 ? (shown.length >= 2 ? shown : years) : ticks(lo, hi, 4);
  axes(root, inner, marks, [], xscale, () => 0, years.length >= 2 ? '' : 'RJD', (v) => (years.length >= 2 ? mjdDate(v).slice(0, 4) : `${Math.round(v + 0.5)}`));
  if (S.min_rjd !== null && S.min_rjd !== '' && S.min_rjd > lo) root.append(svg('rect', { x: inner.x0, width: xscale(S.min_rjd) - inner.x0, y: inner.y0, height: inner.y1 - inner.y0, fill: 'rgba(255,255,255,0.05)' }));
  if (S.max_rjd !== null && S.max_rjd !== '' && S.max_rjd < hi) root.append(svg('rect', { x: xscale(S.max_rjd), width: inner.x1 - xscale(S.max_rjd), y: inner.y0, height: inner.y1 - inner.y0, fill: 'rgba(255,255,255,0.05)' }));
  names.forEach((n, s) => {
    const colour = s < HIST_COLOURED ? css(SERIES[s]) : css('--series-other');
    const y = inner.y0 + 4 + s * rowHeight;
    POINTS[n].mjd.forEach((m) => {
      if (m === null) return;
      const mark = svg('rect', { x: xscale(m) - 1, width: 2, y, height: rowHeight - 4, fill: colour, opacity: inBounds(m) ? 1 : 0.3 });
      mark.dataset.tip = `${esc(n)}<br><span class="mono">${mjdDate(m)} · RJD ${fmt(m + 0.5, 3)}</span>`;
      root.append(mark);
    });
  });
  $('plot_time').replaceChildren(root);
}

// ── settings ────────────────────────────────────────────────────────────
async function loadInstrumentThings() {
  const data = await api.get('/api/options', { instrument: S.instrument, lang: S.lang });
  OPTIONS = data.options || [];
  SWITCHES = data.switches || [];
  VETTED = await api.get('/api/vetted', { instrument: S.instrument });
  if (!Array.isArray(VETTED)) VETTED = [];
  if (S.data_dir) {
    BASES = await api.get('/api/bases', { data_dir: S.data_dir, instrument: S.instrument, source: S.source });
    TEMPLATES = await api.get('/api/templates', { data_dir: S.data_dir, instrument: S.instrument });
    // a frozen basis of another instrument or source cannot stay chosen
    const basis = S.pca2d.input_name;
    if (basis && !BASES.some((b) => b.compatible && b.name === basis)) {
      delete S.pca2d.input_name;
      save();
      $('target_notes').textContent = t('basis_dropped', basis, S.instrument, S.source);
    }
  }
  drawOptions();
  drawSwitches();
  drawPerTarget();
  drawExtra();
}

const CHOICES = { weight: ['flux', 'velocity'], mask: ['common', 'exposure', 'none'] };

function optionValue(row) {
  return Object.prototype.hasOwnProperty.call(S.pca2d, row.key) ? S.pca2d[row.key] : row.default;
}

function setOption(row, value) {
  const same = value === row.default || (value === '' && (row.default === null || row.default === undefined));
  if (same) delete S.pca2d[row.key];
  else S.pca2d[row.key] = value;
  wrapEdited = false;
  save();
  drawOptions();
}

function control(row) {
  const value = optionValue(row);
  const set = Object.prototype.hasOwnProperty.call(S.pca2d, row.key);
  if (row.key === 'input_name') {
    // only the bases of this instrument and data source: LBL refuses the
    //   others. The kept ones (any subfolder of models/, one group per
    //   folder) first, which LBL reads first for a name in both places
    const usable = BASES.filter((b) => b.compatible);
    const option = (b) => el('option', { value: b.name, text: t('basis_label', b.name, b.n_earth ?? '?', b.objects.length) });
    const group = (label, rows) => (rows.length ? el('optgroup', { label }, ...rows.map(option)) : null);
    const keptFolders = [...new Set(usable.filter((b) => b.kept).map((b) => b.folder))];
    const select = el('select', { class: set ? 'set' : '', onchange: (event) => setOption(row, event.target.value || null) },
      el('option', { value: '', text: t('basis_none') }),
      ...keptFolders.map((folder) => group(t('basis_group_kept', folder), usable.filter((b) => b.kept && b.folder === folder))),
      group(t('basis_group_fits', 'pca2d/components'), usable.filter((b) => !b.kept && !usable.some((k) => k.kept && k.name === b.name))));
    select.value = value || '';
    const chosen = usable.find((b) => b.name === value);
    if (!chosen || chosen.kept || usable.some((b) => b.kept && b.name === value)) return select;
    const keep = el('button', {
      type: 'button', class: 'button small', text: t('basis_keep'), title: t('basis_keep_tip', S.instrument, S.source),
      onclick: async () => {
        const out = await api.post('/api/keep_basis', { data_dir: S.data_dir, name: chosen.file, instrument: S.instrument });
        if (out && out.error) { $('target_notes').textContent = out.error; return; }
        BASES = await api.get('/api/bases', { data_dir: S.data_dir, instrument: S.instrument, source: S.source });
        drawOptions();
      },
    });
    return el('span', { class: 'basis-choice' }, select, keep);
  }
  if (CHOICES[row.key]) {
    const select = el('select', { class: set ? 'set' : '', onchange: (event) => setOption(row, event.target.value) },
      ...CHOICES[row.key].map((c) => el('option', { value: c, text: c })));
    select.value = value ?? '';
    return select;
  }
  if (row.kind === 'bool') {
    return el('input', { type: 'checkbox', checked: Boolean(value), onchange: (event) => setOption(row, event.target.checked) });
  }
  if (row.kind === 'int' || row.kind === 'float') {
    return el('input', {
      type: 'number', step: row.kind === 'int' ? 1 : 'any', class: set ? 'set' : '',
      value: set ? value : '', placeholder: row.default ?? t('unset'),
      onchange: (event) => {
        const raw = event.target.value;
        if (raw === '') { setOption(row, row.default); return; }
        setOption(row, row.kind === 'int' ? parseInt(raw, 10) : parseFloat(raw));
      },
    });
  }
  return el('input', {
    type: 'text', class: set ? 'set' : '', value: set ? (value ?? '') : '', placeholder: row.default ?? defaultName(row.key) ?? t('unset'),
    spellcheck: 'false',
    onchange: (event) => setOption(row, event.target.value.trim() || row.default),
  });
}

// what an unset name will be, said in its grey: the corrected object's
//   (<object>_pca2d) for the variant, the objects' for the basis saved
function defaultName(key) {
  const objects = S.ticked.length ? S.ticked : ['<object>'];
  const more = objects.length > 1 ? ', …' : '';
  if (key === 'name') return `${objects[0]}_pca2d${more}`;
  if (key === 'output_name') return S.pca2d.joint ? objects.join('+') : `${objects[0]}${more}`;
  return null;
}

function drawOptions() {
  const holder = $('options');
  const groups = {};
  // THE JOINT FIT means nothing for one target, and is what several want:
  //   not offered for one, on by default for several (wrapgen.build does the
  //   same)
  const several = S.ticked.length > 1;
  OPTIONS.filter((row) => row.key !== 'joint' || several)
    .map((row) => (row.key === 'joint' ? { ...row, default: true } : row))
    .forEach((row) => { (groups[row.group] = groups[row.group] || []).push(row); });
  const parts = [];
  ['model', 'bases', 'correction', 'runs', 'more'].forEach((group) => {
    if (!groups[group]) return;
    parts.push(el('div', { class: 'group-title', text: t(`group_${group}`) }));
    parts.push(el('div', { class: 'options-grid' }, groups[group].map((row) => {
      const set = Object.prototype.hasOwnProperty.call(S.pca2d, row.key);
      // the window's own help; LBL's description (English) for an option it has none for yet
      const help = row.help ? helpHtml(row.help) : (row.lbl ? `${esc(t('lbl_says'))}<br>${helpHtml(row.lbl)}` : '');
      return el('div', { class: `option${set ? ' changed' : ''}` },
        el('div', { class: 'name' }, row.label, help ? el('span', { class: 'info', 'data-tip': help, text: 'i' }) : null,
          el('div', { class: 'key', text: `${row.key} · ${t('default')} ${row.default === null || row.default === undefined ? t('unset') : row.default}` })),
        el('div', { class: 'control' }, control(row)),
        el('button', { type: 'button', class: 'reset', title: t('reset'), text: '↺', onclick: () => { delete S.pca2d[row.key]; save(); drawOptions(); } }));
    })));
  });
  holder.replaceChildren(...parts);
  $('pca2d_on').checked = S.pca2d_on;
  holder.style.opacity = S.pca2d_on ? 1 : 0.4;
}

// NIRPS's and SPIRou's spectra come telluric-corrected (APERO): LBL's own
//   telluric cleaning is off for them, and cannot be ticked
const NO_TELLUCLEAN = ['NIRPS', 'SPIROU'];
// REWRITE OR KEEP what is on disk, for every LBL step at once: the SKIP_LBL_*
//   switches together (template, mask and velocities kept by default, the
//   rdb tables always compiled again), each still set on its own below
const KEEP_KEYS = ['SKIP_LBL_TEMPLATE', 'SKIP_LBL_MASK', 'SKIP_LBL_COMPUTE'];
function switchValue(key) {
  const row = SWITCHES.find((r) => r.key === key) || {};
  return Object.prototype.hasOwnProperty.call(S.lbl, key) ? S.lbl[key] : row.default;
}
function drawOverwrite() {
  const kept = KEEP_KEYS.map((key) => Boolean(switchValue(key)));
  const box = $('overwrite_all');
  box.checked = kept.every((v) => !v) && !switchValue('SKIP_LBL_COMPILE');
  box.indeterminate = !box.checked && !kept.every((v) => v);
}
function setOverwrite(rewrite) {
  KEEP_KEYS.forEach((key) => { S.lbl[key] = !rewrite; });
  S.lbl.SKIP_LBL_COMPILE = false;
  wrapEdited = false;
  save();
  drawSwitches();
}

function drawSwitches() {
  $('switches').replaceChildren(...SWITCHES.map((row) => {
    let value = Object.prototype.hasOwnProperty.call(S.lbl, row.key) ? S.lbl[row.key] : row.default;
    if (value === null) value = instrumentTelluclean();
    const locked = row.key === 'RUN_LBL_TELLUCLEAN' && NO_TELLUCLEAN.includes(family(S.instrument));
    return el('label', { class: `check${locked ? ' locked' : ''}`, title: locked ? t('tellu_off') : null }, el('input', {
      type: 'checkbox', checked: locked ? false : Boolean(value), disabled: locked,
      onchange: (event) => { S.lbl[row.key] = event.target.checked; wrapEdited = false; save(); drawOverwrite(); },
    }), el('span', { class: 'sw-text' }, t(`sw_${row.key}`), el('span', { class: 'key', text: row.key })), info(`help_sw_${row.key}`));
  }));
  drawOverwrite();
  $('resproj').checked = Boolean(S.resproj);
}

function drawPerTarget() {
  $('vetted_bar').hidden = !VETTED.length;
  const rows = S.ticked.map((name) => {
    const target = TARGETS.find((row) => row.name === name) || {};
    // every template offered says its temperature: the header's for those
    //   in templates/, the scan's for a ticked target's own
    const temps = {};
    TEMPLATES.forEach((tp) => { if (tp.teff) temps[tp.name] = tp.teff; });
    TARGETS.forEach((row) => { const known = row.teff || simbadTeff(row.name); if (known && !temps[row.name]) temps[row.name] = known; });
    const labelled = (n) => `${n} (${temps[n] ? Math.round(temps[n]) : '?'} K)`;
    const others = TEMPLATES.map((tp) => tp.name).filter((n) => !/_pca2d(_|$)/.test(n) && !/_tc$/.test(n) && n !== name);
    const choices = [...new Set([...others, ...S.ticked.filter((n) => n !== name)])]
      .sort((a, b) => (temps[a] || 1e9) - (temps[b] || 1e9));
    const own = S.teff[name] || target.teff || simbadTeff(name) || temps[name];
    const select = el('select', { onchange: (event) => { if (event.target.value) S.comparison[name] = event.target.value; else delete S.comparison[name]; wrapEdited = false; save(); } },
      el('option', { value: '', text: own ? `${t('comparison_self')} (${Math.round(own)} K)` : t('comparison_self') }),
      ...choices.map((c) => el('option', { value: c, text: labelled(c) })),
      // LBL's vetted templates of this instrument, with their temperature
      VETTED.length ? el('optgroup', { label: t('vetted_group') },
        VETTED.map((v) => el('option', { value: v.name, text: t('vetted_label', v.object, v.teff, v.nfiles) }))) : null);
    select.value = S.comparison[name] || '';
    // the headers' Teff in grey, else SIMBAD's; SIMBAD's said under it
    const teff = el('input', {
      type: 'number', step: 10, value: S.teff[name] ?? '', placeholder: target.teff ? Math.round(target.teff) : (simbadTeff(name) ?? ''),
      onchange: (event) => { if (event.target.value === '') delete S.teff[name]; else S.teff[name] = Number(event.target.value); wrapEdited = false; save(); },
    });
    const best = VETTED.length ? el('button', { type: 'button', class: 'button small', text: t('best_match'), onclick: () => bestMatch([name]) }) : null;
    return el('tr', {}, el('td', { text: name }), el('td', {}, el('span', { class: 'row gap' }, select, best)), el('td', {}, teff, simbadNote(name, target)));
  });
  $('per_target').replaceChildren(
    el('thead', {}, el('tr', {}, el('th', { text: t('col_target') }), el('th', {}, t('comparison'), info('help_comparison')), el('th', {}, t('teff'), info('help_teff')))),
    el('tbody', {}, rows.length ? rows : el('tr', {}, el('td', { colspan: 3, class: 'muted', text: t('no_points') }))));
  $('simbad_teff').disabled = !S.ticked.length;
  // a target's Teff asked of SIMBAD once (simbad.py keeps the answer)
  askSimbad(S.ticked.filter((n) => !simbadAsked.has(n)));
}

// ── SIMBAD's Teff (simbad.py): the median of its measurements, else its
//   spectral type's; used when the headers have none and nothing is typed
let SIMBAD = {};
const simbadAsked = new Set();
function simbadTeff(name) {
  const answer = SIMBAD[name];
  return answer && answer.teff ? Math.round(answer.teff) : null;
}
async function askSimbad(names, refresh = false) {
  // a target the scan has not read yet waits: its headers' OBJECT is one
  //   of the names SIMBAD is asked (a folder's name can be one SIMBAD
  //   does not know)
  names = names.filter((n) => TARGETS.some((row) => row.name === n));
  if (!names.length || !S.data_dir) return;
  names.forEach((n) => { simbadAsked.add(n); SIMBAD[n] = { state: 'asking' }; });
  drawPerTarget();
  const targets = names.map((name) => ({ name, object: (TARGETS.find((row) => row.name === name) || {}).object || null }));
  let data = {};
  try {
    data = await api.post('/api/teff', { data_dir: S.data_dir, targets, refresh }) || {};
  } catch (error) { /* said as failed below */ }
  names.forEach((n) => { SIMBAD[n] = data[n] || { state: 'failed' }; });
  wrapEdited = false;
  drawPerTarget();
  // its systemic velocity puts a target against v_tot
  drawPlots();
}
function simbadNote(name, target) {
  const answer = SIMBAD[name];
  if (!answer) return null;
  let text = '';
  if (answer.state === 'asking') text = t('simbad_asking');
  else if (answer.state === 'failed') text = t('simbad_failed');
  else if (answer.state === 'unknown') text = t('simbad_unknown');
  else if (!answer.teff) text = t('simbad_none', answer.main);
  else if (answer.source === 'sptype') text = t('simbad_teff_spt', Math.round(answer.teff), answer.sptype);
  else if (answer.n < answer.n_all) text = t('simbad_teff_some', Math.round(answer.teff), answer.n, answer.n_all);
  else text = t('simbad_teff', Math.round(answer.teff), answer.n);
  // the headers say another: SIMBAD's taken with a click
  const other = answer.teff && target.teff && S.teff[name] === undefined && Math.abs(answer.teff - target.teff) >= 10;
  return el('div', { class: 'teff-note small muted', title: answer.main ? t('simbad_why', answer.main, answer.sptype || '?') : null }, text,
    other ? el('button', { type: 'button', class: 'linkish', text: t('simbad_use'),
      onclick: () => { S.teff[name] = Math.round(answer.teff); wrapEdited = false; save(); drawPerTarget(); } }) : null);
}

// the vetted template of the nearest Teff, for targets of 2900-4300 K
//   (lbl.core.vetted.best_match); the others are said and left as they are
const VETTED_RANGE = [2900, 4300];
function bestMatch(names) {
  const notes = [];
  names.forEach((name) => {
    const target = TARGETS.find((row) => row.name === name) || {};
    const own = TEMPLATES.find((tp) => tp.name === name) || {};
    const teff = Number(S.teff[name] || target.teff || simbadTeff(name) || own.teff);
    if (!teff || teff < VETTED_RANGE[0] || teff > VETTED_RANGE[1] || !VETTED.length) {
      notes.push(t('best_match_none', name, teff ? Math.round(teff) : '?'));
      return;
    }
    const best = VETTED.reduce((a, b) => (Math.abs(b.teff - teff) < Math.abs(a.teff - teff) ? b : a));
    S.comparison[name] = best.name;
  });
  $('vetted_notes').textContent = notes.join(' · ');
  wrapEdited = false;
  save();
  drawPerTarget();
}

function drawExtra() {
  const show = (box, text, values) => {
    const keys = Object.keys(values || {});
    $(box).hidden = keys.length === 0;
    $(text).textContent = keys.map((k) => `${k} = ${JSON.stringify(values[k])}`).join('\n');
  };
  show('extra_box', 'extra', S.extra);
  show('pca2d_extra_box', 'pca2d_extra', S.pca2d_extra);
}

// ── loading a wrap ──────────────────────────────────────────────────────
async function loadWrapList() {
  if (!S.data_dir) return;
  const wraps = await api.get('/api/wraps', { data_dir: S.data_dir });
  if (!Array.isArray(wraps)) return;
  $('wrap_list').replaceChildren(el('option', { value: '', text: t('load_pick') }),
    ...wraps.map((w) => el('option', { value: w.path, text: `${w.name}  \u00B7  ${when(w.mtime)}` })));
}

async function loadWrap() {
  const path = $('wrap_path').value.trim() || $('wrap_list').value;
  if (!path) return;
  const data = await api.post('/api/load_wrap', { path });
  const box = $('load_messages');
  if (data.error) { box.replaceChildren(el('div', { class: 'message error', text: data.error })); return; }
  const loaded = data.state;
  const keep = { lang: S.lang, tab: S.tab, python: S.python, sort: S.sort, hidden_instruments: S.hidden_instruments };
  Object.assign(S, JSON.parse(JSON.stringify(DEFAULTS)), keep, loaded,
    { name: loaded.name || '', name_auto: !loaded.name, instrument_set: true, setup_manual: true });
  const notes = [t('loaded', data.path), ...(data.notes || [])];
  if (data.skipped && data.skipped.length) notes.push(t('load_skipped', data.skipped.join(', ')));
  box.replaceChildren(el('div', { class: 'message warning' }, el('ul', {}, notes.map((n) => el('li', { text: n })))));
  $('data_dir').value = S.root || S.data_dir;
  $('run_name').value = S.name;
  await loadFolders();
  $('min_rjd').value = S.min_rjd ?? '';
  $('max_rjd').value = S.max_rjd ?? '';
  $('skip_pca2d').value = S.skip_pca2d === null || S.skip_pca2d === undefined ? '' : String(S.skip_pca2d);
  fillInstruments();
  save();
  await loadInstrumentThings();
  await loadTargets();
  drawExtra();
  wrapEdited = false;
  await build();
}

// ── run ─────────────────────────────────────────────────────────────────
// the big button: green to run, orange and pulsing while a run goes
function updateLaunch(running) {
  const button = $('queue');
  button.classList.toggle('running', Boolean(running));
  $('launch_label').textContent = running
    ? t('launch_running', [running.stage, running.counter].filter(Boolean).join(' ') || t('state_running'))
    : t('launch');
  button.title = running ? t('launch_more') : '';
}

// ── the steps of a run, ticked off as its log goes ──────────────────────
let plannedSteps = [];
const STEP_ICON = { pending: '○', running: '◐', done: '✓', failed: '✗', stopped: '■' };
function drawSteps(steps, title) {
  const holder = $('steps');
  $('steps_note').textContent = title || t('steps_planned');
  const groups = [];
  // group: the name a step is shown under, when not its object's (the
  //   template between the two telluric passes, under the target's)
  (steps || []).forEach((step) => {
    const last = groups[groups.length - 1];
    const under = step.group || step.obj;
    if (last && last.obj === under) last.steps.push(step);
    else groups.push({ obj: under, steps: [step] });
  });
  holder.replaceChildren(...groups.map((group) => el('div', { class: 'step-group' },
    el('span', { class: 'step-obj mono', text: group.obj }),
    ...group.steps.map((step) => el('span', {
      // a step going whose files are counted fills its pill as it goes
      class: `step ${step.state || 'pending'}${step.at === undefined ? '' : ' linked'}${step.state === 'running' && step.progress !== undefined ? ' counted' : ''}`,
      style: step.state === 'running' && step.progress !== undefined ? `--progress: ${(100 * step.progress).toFixed(1)}%` : null,
      title: [step.at === undefined ? null : t('step_goto'),
        step.state === 'running' && step.progress !== undefined ? `${Math.round(100 * step.progress)} %` : null].filter(Boolean).join(' · ') || null,
      onclick: step.at === undefined ? null : () => showStep(step),
    },
      step.state === 'running' ? el('span', { class: 'ring' })
        : el('span', { class: 'step-icon', text: STEP_ICON[step.state || 'pending'] }),
      t(step.label || `step_${step.kind}`))))));
}

// ── the wrap, with Python's colours ─────────────────────────────────────
// one pass over the text: comments, strings (triple-quoted first), numbers,
//   keywords, constants and calls; everything else as it is. The coloured
//   copy sits under a transparent textarea that keeps the editing
const PY_KEYWORDS = new Set(('and as assert break class continue def del elif else except finally for from global if import in '
  + 'is lambda nonlocal not or pass raise return try while with yield').split(' '));
const PY_CONSTANTS = new Set(['True', 'False', 'None']);
const PY_TOKEN = /(#[^\n]*)|("""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?)|(\b\d+(?:\.\d*)?(?:[eE][-+]?\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)/g;
function highlightPython(text) {
  let out = '';
  let last = 0;
  let match;
  PY_TOKEN.lastIndex = 0;
  while ((match = PY_TOKEN.exec(text)) !== null) {
    out += esc(text.slice(last, match.index));
    const [token, comment, string, number, word] = match;
    let kind = null;
    if (comment) kind = 'cm';
    else if (string) kind = 'st';
    else if (number) kind = 'nu';
    else if (word && PY_KEYWORDS.has(word)) kind = 'kw';
    else if (word && PY_CONSTANTS.has(word)) kind = 'co';
    else if (word && text[PY_TOKEN.lastIndex] === '(') kind = 'fn';
    out += kind ? `<span class="py-${kind}">${esc(token)}</span>` : esc(token);
    last = PY_TOKEN.lastIndex;
  }
  // a last newline needs a line under it, or the copy stops one line short
  return `${out}${esc(text.slice(last))}\n`;
}
function highlightWrap() {
  $('wrap_hl').innerHTML = highlightPython($('wrap_text').value);
  syncWrapScroll();
}
function syncWrapScroll() {
  const pre = $('wrap_hl').parentElement;
  pre.scrollTop = $('wrap_text').scrollTop;
  pre.scrollLeft = $('wrap_text').scrollLeft;
}
function setWrap(text) {
  $('wrap_text').value = text || '';
  highlightWrap();
}

function buildState() {
  const teff = {};
  S.ticked.forEach((name) => {
    const target = TARGETS.find((row) => row.name === name) || {};
    teff[name] = S.teff[name] ?? (target.teff ? Math.round(target.teff) : simbadTeff(name));
  });
  return {
    data_dir: S.data_dir, instrument: S.instrument, source: S.source, python: S.python,
    targets: S.ticked, comparison: S.comparison, teff, lbl: S.lbl,
    pca2d_on: S.pca2d_on, pca2d: S.pca2d, skip_pca2d: S.skip_pca2d,
    extra: S.extra, pca2d_extra: S.pca2d_extra,
    min_rjd: S.min_rjd, max_rjd: S.max_rjd, resproj: S.resproj,
    name: S.name_auto ? '' : S.name,
    lang: S.lang,
    // the instrument each target's headers give, checked against the one chosen
    instruments: Object.fromEntries(S.ticked.map((n) => [n, (TARGETS.find((row) => row.name === n) || {}).instrument || ''])),
  };
}

function showMessages(errors, warnings) {
  const parts = [];
  if (errors && errors.length) parts.push(el('div', { class: 'message error' }, el('strong', { text: t('errors') }), el('ul', {}, errors.map((e) => el('li', { text: e })))));
  if (warnings && warnings.length) parts.push(el('div', { class: 'message warning' }, el('strong', { text: t('warnings') }), el('ul', {}, warnings.map((w) => el('li', { text: w })))));
  $('messages').replaceChildren(...parts);
}

async function build() {
  const data = await api.post('/api/build', buildState());
  if (data.error) { showMessages([data.error], []); return null; }
  if (S.name_auto) { S.name = data.name; $('run_name').value = data.name; }
  setWrap(data.wrap);
  wrapEdited = false;
  plannedSteps = data.steps || [];
  if (!selectedJob) drawSteps(plannedSteps);
  showMessages(data.errors, data.warnings);
  $('queue').disabled = Boolean(data.errors && data.errors.length);
  return data;
}

async function queue() {
  const data = await api.post('/api/queue', { state: buildState(), name: S.name, wrap: $('wrap_text').value });
  if (data.errors) { showMessages(data.errors, []); return; }
  if (data.error) { showMessages([data.error], []); return; }
  selectJob(data.id);
  pollQueue();
}

async function saveWrap(overwrite) {
  const data = await api.post('/api/save_wrap', { data_dir: S.data_dir, name: S.name, wrap: $('wrap_text').value, state: buildState(), overwrite });
  if (data.exists && !overwrite) {
    if (await confirmBox(t('overwrite_q', data.path || data.error))) saveWrap(true);
    return;
  }
  $('save_status').textContent = data.path ? t('saved', data.path) : (data.error || '');
}

async function pollQueue() {
  const jobs = await api.get('/api/queue', { data_dir: S.data_dir });
  if (!Array.isArray(jobs)) return;
  const shown = jobs.slice(-12).reverse();
  $('queue_list').replaceChildren(...(shown.length ? shown.map((job) => el('div', {
    class: `job${job.id === selectedJob ? ' selected' : ''}`, onclick: () => selectJob(job.id),
  },
  el('div', { class: 'row spread' }, el('span', { class: 'title', text: job.name }), el('span', { class: `badge ${job.state}`, text: t(`state_${job.state}`) })),
  el('div', { class: 'meta', text: [job.stage, job.counter, job.state === 'running' ? took(job.started) : (job.ended ? took(job.started, job.ended) : '')].filter(Boolean).join(' · ') }),
  job.reports && job.reports.length ? el('div', { class: 'meta' }, job.reports.slice(-3).map((p) => [fileLink(p, baseName(p)), ' '])) : null))
    : [el('p', { class: 'hint small', text: t('queue_empty') })]));
  if (!selectedJob) {
    const running = jobs.find((job) => job.state === 'running') || jobs[jobs.length - 1];
    if (running) selectJob(running.id);
  }
  updateLaunch(jobs.find((j) => j.state === 'running'));
  // the Runs tab follows the queue: read again as a run is queued, starts
  //   or ends, and every 10 s while one goes (its duration)
  const runsNow = [S.data_dir, ...jobs.map((j) => `${j.id}:${j.state}`)].join(' ');
  if (S.tab === 'runs' && (runsNow !== runsSeen || (jobs.some((j) => j.state === 'running') && Date.now() - runsRead > 10000))) loadRuns();
  runsSeen = runsNow;
  const job = jobs.find((j) => j.id === selectedJob);
  if (job && job.steps && job.steps.length) drawSteps(job.steps, t('steps_of', job.name));
  else drawSteps(plannedSteps);
  if (job) {
    $('log_stage').textContent = [job.stage, job.counter].filter(Boolean).join(' · ');
    $('log_stage').className = `badge ${job.state}`;
    $('log_status').textContent = job.status || '';
    $('stop').textContent = job.state === 'queued' ? t('cancel') : t('stop');
    $('stop').disabled = !['queued', 'running'].includes(job.state);
  }
}

function selectJob(id) {
  if (selectedJob === id && stream) return;
  selectedJob = id;
  if (stream) stream.close();
  $('log').replaceChildren();
  logLast = null;
  $('log_title').textContent = id;
  stream = new EventSource(`/api/stream?id=${encodeURIComponent(id)}&from=0`);
  const apply = (payload) => {
    const box = $('log');
    // the log follows the run, unless one has scrolled up to read it: back
    //   at the bottom, it follows again
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    payload.events.forEach((event) => {
      if (event.op === 'replace' && logLast) {
        logLast.textContent = event.text;
        logLast.className = `lv-${event.level}`;
      } else {
        // lv-: a line's level; a bare 'info' is the (i) of the helps
        logLast = el('div', { class: `lv-${event.level}`, text: event.text, 'data-n': event.n });
        box.append(logLast);
      }
    });
    if (atBottom) box.scrollTop = box.scrollHeight;
    logEdges();
  };
  stream.onmessage = (message) => apply(JSON.parse(message.data));
  stream.addEventListener('end', (message) => { apply(JSON.parse(message.data)); stream.close(); stream = null; pollQueue(); });
  stream.onerror = () => { /* the browser reconnects on its own */ };
  pollQueue();
}

/* a step clicked: the log at the line it reached that step on (the last
   line from at or before its event: a bar's frames replace their line) */
function showStep(step) {
  const box = $('log');
  const lines = box.children;
  if (!lines.length) return;
  let lo = 0;
  let hi = lines.length - 1;
  let found = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (Number(lines[mid].dataset.n) <= step.at) { found = mid; lo = mid + 1; } else hi = mid - 1;
  }
  const line = lines[found];
  box.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  box.scrollTop = Math.max(0, line.offsetTop - 8);
  line.classList.remove('flash');
  void line.offsetWidth;
  line.classList.add('flash');
  logEdges();
}

/* the jumps of a log (the run's, the archive search's): off where the log
   already is */
function logEdges(id = 'log') {
  const box = $(id);
  $(`${id}_top`).disabled = box.scrollTop < 4;
  $(`${id}_bottom`).disabled = box.scrollHeight - box.scrollTop - box.clientHeight < 4;
}

async function stopJob() {
  if (!selectedJob) return;
  await api.post('/api/stop', { id: selectedJob });
  pollQueue();
}

// ── runs ────────────────────────────────────────────────────────────────
function table(holder, head, rows) {
  holder.replaceChildren(el('thead', {}, el('tr', {}, head.map((h) => el('th', { text: t(h) })))),
    el('tbody', {}, rows.length ? rows : el('tr', {}, el('td', { colspan: head.length, class: 'muted', text: t('nothing') }))));
}

let runsSeen = '';
let runsRead = 0;
async function loadRuns() {
  if (!S.data_dir) return;
  runsRead = Date.now();
  const data = await api.get('/api/runs', { data_dir: S.data_dir });
  table($('runs_window'), ['col_name', 'col_state', 'col_started', 'col_duration', 'col_reports', 'log'],
    (data.window || []).map((run) => el('tr', {},
      el('td', { class: 'mono', text: run.name }),
      el('td', {}, el('span', { class: `badge ${run.state}`, text: t(`state_${run.state}`) })),
      el('td', { class: 'mono', text: when(run.started || run.created) }),
      el('td', { class: 'mono', text: run.started ? took(run.started, run.ended) : '' }),
      el('td', {}, (run.reports || []).map((p) => [fileLink(p, baseName(p)), ' '])),
      el('td', {}, fileLink(run.log, run.log ? baseName(run.log) : 'log'), ' ', fileLink(run.wrap, 'wrap.py')))));
  table($('runs_reports'), ['col_name', 'col_date', 'col_bytes'],
    (data.reports || []).map((report) => el('tr', {},
      el('td', {}, fileLink(report.path, report.name)),
      el('td', { class: 'mono', text: when(report.mtime) }),
      el('td', { class: 'num', text: human(report.size) }))));
}

/* a table's flavour, the data folder it is in before it when that is not
   the one the window works in (another instrument's, the root's own) */
function flavourOf(file) {
  return file.own === false ? `${file.home || t('explore_root_home')}: ${file.flavour}` : file.flavour;
}

// ── cleanup ─────────────────────────────────────────────────────────────
let CLEANUP = [];
async function measure() {
  if (!S.data_dir) return;
  $('cleanup_status').textContent = t('measuring');
  const data = await api.get('/api/cleanup', { data_dir: S.data_dir });
  CLEANUP = data.items || [];
  const busy = data.busy || [];
  $('cleanup_status').textContent = '';
  $('delete').disabled = busy.length > 0;
  const kinds = ['scratch', 'rebuildable', 'results', 'data'];
  const parts = [];
  kinds.forEach((kind) => {
    const items = CLEANUP.filter((item) => item.kind === kind);
    if (!items.length) return;
    const total = items.reduce((sum, item) => sum + item.bytes, 0);
    parts.push(el('div', { class: 'kind-title' }, t(`kind_${kind}`), el('span', { class: 'size', text: `${t('total')} ${human(total)}` })));
    const deletable = kind === 'scratch' || kind === 'rebuildable';
    // ONE CLICK FOR A WHOLE TYPE: every per-exposure velocity, every mask...;
    //   a second click unticks them
    if (deletable) {
      const codes = [...new Set(items.map((item) => item.code))];
      const button = (label, which, title) => el('button', {
        type: 'button', class: 'button small', text: label, title,
        'data-which': which, onclick: () => toggleCleanup(which),
      });
      parts.push(el('div', { class: 'row gap wrap cleanup-select' },
        button(t('select_kind'), `[data-kind="${kind}"]`, t('select_kind_hint')),
        ...codes.map((code) => {
          const mine = items.filter((item) => item.code === code);
          const size = mine.reduce((sum, item) => sum + item.bytes, 0);
          return button(`${T[`type_${code}`] || code} (${mine.length}, ${human(size)})`,
            `[data-kind="${kind}"][data-code="${code}"]`, t('select_type_hint'));
        })));
    }
    parts.push(el('div', { class: 'table-wrap' }, el('table', { class: 'data' },
      el('tbody', {}, items.map((item) => el('tr', {},
        el('td', {}, deletable ? el('input', { type: 'checkbox', 'data-path': item.path, 'data-kind': kind, 'data-code': item.code }) : ''),
        el('td', { class: 'mono', text: relative(item.path) + (item.link ? ` → ${item.link}` : '') }),
        el('td', { class: 'muted small', text: T[`what_${item.code}`] || item.what }),
        el('td', { class: 'num', text: human(item.bytes) }),
        el('td', { class: 'num', text: item.files })))))));
  });
  if (busy.length) {
    // unknown: the server has no way to tell whether a run is going
    parts.unshift(el('div', { class: 'message warning' }, busy.some((p) => p.unknown) ? t('busy_unknown')
      : t('busy', busy.map((p) => `pid ${p.pid}: ${p.command}`).join(' \u00B7 '))));
  }
  $('cleanup').replaceChildren(...parts);
  cleanupSelection();
}

/* every box that matches `which`: all ticked if one was not, else none */
function toggleCleanup(which) {
  const boxes = [...document.querySelectorAll(`#cleanup input[type=checkbox]${which}`)];
  const tick = boxes.some((box) => !box.checked);
  boxes.forEach((box) => { box.checked = tick; });
  cleanupSelection();
}

/* what is ticked, said beside Delete */
function cleanupSelection() {
  const paths = [...document.querySelectorAll('#cleanup input[type=checkbox]:checked')].map((box) => box.dataset.path);
  const size = CLEANUP.filter((item) => paths.includes(item.path)).reduce((sum, item) => sum + item.bytes, 0);
  $('cleanup_selected').textContent = paths.length ? t('selected_summary', paths.length, human(size)) : '';
  // a type all ticked: its button lit
  document.querySelectorAll('.cleanup-select button').forEach((button) => {
    const boxes = [...document.querySelectorAll(`#cleanup input[type=checkbox]${button.dataset.which}`)];
    button.classList.toggle('on', boxes.length > 0 && boxes.every((box) => box.checked));
  });
}

async function deleteSelected() {
  const paths = [...document.querySelectorAll('#cleanup input[type=checkbox]:checked')].map((box) => box.dataset.path);
  if (!paths.length) return;
  const total = CLEANUP.filter((item) => paths.includes(item.path)).reduce((sum, item) => sum + item.bytes, 0);
  const list = paths.map((p) => `  ${relative(p)}`).join('\n');
  if (!(await confirmBox(`${t('confirm_delete', paths.length, human(total))}\n\n${list}`))) return;
  const results = await api.post('/api/delete', { data_dir: S.data_dir, paths });
  const done = (Array.isArray(results) ? results : []).filter((r) => r.done).length;
  $('cleanup_status').textContent = t('deleted', done);
  measure();
}

// ── data explorer ───────────────────────────────────────────────────────
// the rdb tables of DATA_DIR/lblrdb, any column against any other, the FIP
//   of what is plotted, and the template each table was measured against
//   with the TAPAS transmission of each molecule under it (explore.py).
//   Several tables overplot: one colour each, the y of each (and each
//   template) about its median. Plotly draws them, fetched the first time
//   the tab opens: the rest of the window does without it.
const PLOTLY_URL = 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js';
let plotlyLoading = null;
function loadPlotly() {
  if (window.Plotly) return Promise.resolve(true);
  if (!plotlyLoading) {
    plotlyLoading = new Promise((resolve) => {
      const script = document.createElement('script');
      script.src = PLOTLY_URL;
      script.onload = () => resolve(true);
      script.onerror = () => { plotlyLoading = null; script.remove(); resolve(false); };
      document.head.append(script);
    });
  }
  return plotlyLoading;
}

// what is shown, kept while the window is open; a window opens on the
//   newest table. sets: the tables plotted, each { base: the table of its
//   menu, slot: its colour (kept while it is shown, whatever the others
//   do), path: the table shown (its nights' one, lbl2_, when the box is
//   ticked), table, fips }. picked: the columns chosen in the menus, kept
//   from one table to the next (the others are the first table's own: MJD,
//   vrad, BERV); c '': no colour
const XP = { files: [], sets: [], nightly: false, x: '', y: '', c: '', picked: {}, gen: 0, tpls: [], tplRange: null,
  tplY: null, tplDraw: 0, lines: null, vsys: null, rest: true, showLines: true };
// water blue, O2 red, CH4 green (as in the PCA2D figures)
const MOLECULE_COLOURS = { H2O: '--series-1', O2: '--series-8', CH4: '--series-3', CO2: '--series-4',
  O3: '--series-7', N2O: '--series-5', NO2: '--series-other', others: '--series-2' };
const PLOT_CONFIG = { responsive: true, displaylogo: false, scrollZoom: true };

// numbers as they are: no 12.3k, no 1,596.2
function xpAxis(extra) {
  return { gridcolor: css('--grid'), linecolor: css('--axis'), zerolinecolor: css('--axis'),
    color: css('--muted'), automargin: true, exponentformat: 'none', ...extra };
}
// a molecule as it is written (H2O: H<sub>2</sub>O)
const chemical = (name) => name.replace(/(\d)/g, '<sub>$1</sub>');
function xpLayout(extra) {
  return { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', separators: '. ',
    font: { family: 'Space Grotesk, sans-serif', color: css('--text'), size: 12 },
    margin: { l: 64, r: 16, t: 28, b: 48 }, hoverlabel: { font: { family: 'IBM Plex Mono, monospace' } },
    legend: { orientation: 'h', x: 0, y: 1.0, yanchor: 'bottom', font: { size: 11 } }, ...extra };
}
const fipText = (value) => (value < 1e-15 ? '< 1e-15' : value < 1e-3 ? value.toExponential(1) : value.toFixed(3));
// a table as the legends say it: its flavour when all the tables are of
//   one star, else its star and its flavour
function tableName(set) {
  const file = XP.files.find((row) => row.path === set.base) || {};
  if (!file.object) return baseName(set.path).replace(/\.rdb$/, '');
  const stars = new Set(XP.sets.map((other) => (XP.files.find((row) => row.path === other.base) || {}).object));
  return stars.size === 1 && XP.sets.length > 1 ? flavourOf(file) : `${file.object}: ${flavourOf(file)}`;
}
const setColour = (set) => css(SERIES[set.slot % SERIES.length]);
const finite = (values) => values.filter((v) => v !== null && Number.isFinite(v));
function median(values) {
  const good = finite(values).sort((a, b) => a - b);
  if (!good.length) return 0;
  const mid = Math.floor(good.length / 2);
  return good.length % 2 ? good[mid] : 0.5 * (good[mid - 1] + good[mid]);
}
// the statistics of the report (lbl/report/stats.py): the rms with one
//   degree of freedom less, the robust rms the median absolute deviation
//   times that of a gaussian (1 / its third quartile, 0.67449)
const MAD_TO_SIGMA = 1 / 0.6744897501960817;
function spread(values) {
  const good = finite(values);
  if (good.length < 2) return { n: good.length, rms: NaN, robust: NaN };
  const mean = good.reduce((sum, v) => sum + v, 0) / good.length;
  const rms = Math.sqrt(good.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (good.length - 1));
  const mid = median(good);
  return { n: good.length, rms, robust: MAD_TO_SIGMA * median(good.map((v) => Math.abs(v - mid))) };
}
/* a number with as many digits as the scatter asks */
function figure(value, scale) {
  if (!Number.isFinite(value)) return '';
  const digits = Math.max(0, Math.min(6, 2 - Math.floor(Math.log10(Math.abs(scale) || 1))));
  return value.toFixed(digits);
}

function exploreMessage(text) {
  $('explore_message').hidden = !text;
  $('explore_message').textContent = text || '';
}

/* the tables read: each set with its table */
const shownSets = () => XP.sets.filter((set) => set.table);

async function loadExplorer() {
  if (!S.data_dir) return;
  if (!(await loadPlotly())) { exploreMessage(t('explore_no_plotly')); return; }
  const data = await api.get('/api/explore/rdbs', { data_dir: S.data_dir });
  XP.files = data.files || [];
  if (!XP.files.length) {
    exploreMessage(t('explore_no_rdb', data.folder || ''));
    ['explore_plot', 'explore_fip', 'template_plot'].forEach((id) => Plotly.purge($(id)));
    ['explore_note', 'explore_fip_note', 'template_note'].forEach((id) => { $(id).textContent = ''; });
    XP.sets = [];
    $('explore_tables').replaceChildren();
    return;
  }
  exploreMessage('');
  XP.sets = XP.sets.filter((set) => XP.files.some((file) => file.path === set.base));
  if (!XP.sets.length) XP.sets = [{ base: data.latest || XP.files[0].path, slot: 0 }];
  drawTableRows();
  // the tables shown have not changed: the plots as they were (their
  //   zoom), to the width of the tab shown again
  if (XP.sets.every((set) => { const shown = shownOf(set); return set.table && set.path === shown.path && set.mtime === shown.mtime; })) {
    ['explore_plot', 'explore_fip', 'template_plot'].forEach((id) => { if ($(id).data) Plotly.Plots.resize($(id)); });
    return;
  }
  await openSets();
}

/* the table a set shows: that of the nights when the box is ticked and
   there is one */
function shownOf(set) {
  const file = XP.files.find((row) => row.path === set.base) || XP.files[0];
  return XP.nightly && file.nightly ? file.nightly : file;
}

/* a row per table: its star, then its flavour (what else its name says:
   the PCA2D variant, the template), its colour beside it when there are
   several, x to take it off */
const fileOf = (set) => XP.files.find((file) => file.path === set.base) || {};
const freeFile = (file, but) => !XP.sets.some((set) => set !== but && set.base === file.path);
function drawTableRows() {
  const several = XP.sets.length > 1;
  const objects = [...new Set(XP.files.map((file) => file.object))];
  const count = (name) => XP.files.filter((file) => file.object === name).length;
  const stamp = (file) => when(Math.max(file.mtime, (file.nightly || file).mtime));
  $('explore_tables').replaceChildren(...XP.sets.map((set, i) => {
    const current = fileOf(set);
    // another star: its newest flavour not shown yet
    const star = el('select', { class: 'explore-object', onchange: (event) => {
      const mine = XP.files.filter((file) => file.object === event.target.value);
      set.base = (mine.find((file) => freeFile(file, set)) || mine[0]).path;
      drawTableRows();
      openSets();
    } }, objects.map((name) => el('option', { value: name, text: `${name} (${count(name)})` })));
    star.value = current.object;
    const flavour = el('select', { class: 'explore-flavour', onchange: (event) => { set.base = event.target.value; drawTableRows(); openSets(); } },
      XP.files.filter((file) => file.object === current.object).map((file) => el('option', { value: file.path, text: `${flavourOf(file)} · ${stamp(file)}` })));
    flavour.value = set.base;
    return el('div', { class: 'explore-table-row' },
      several ? el('span', { class: 'swatch', style: `background: ${setColour(set)}` }) : null,
      star, flavour,
      several ? el('button', { type: 'button', class: 'button small', text: '\u00D7', title: t('explore_remove'),
        onclick: () => { XP.sets.splice(i, 1); drawTableRows(); openSets(); } }) : null);
  }));
  $('explore_add').disabled = XP.sets.length >= SERIES.length || !XP.files.some((file) => freeFile(file));
  const nights = XP.sets.some((set) => fileOf(set).nightly);
  $('explore_nightly').disabled = !nights;
  $('explore_nightly').checked = XP.nightly && nights;
}

/* another flavour of the first table's star, the newest not shown; another
   star once they all are */
function addTable() {
  const used = XP.sets.map((set) => set.slot);
  let slot = 0;
  while (used.includes(slot)) slot += 1;
  const star = fileOf(XP.sets[0]).object;
  const next = XP.files.find((file) => file.object === star && freeFile(file)) || XP.files.find((file) => freeFile(file));
  if (!next) return;
  XP.sets.push({ base: next.path, slot });
  drawTableRows();
  openSets();
}

/* every table read (those not read yet or changed), then the plots: the
   columns picked stay when all the tables have them */
async function openSets() {
  const gen = ++XP.gen;
  const templatesBefore = shownSets().map((set) => (set.table.template || {}).path).join('|');
  await Promise.all(XP.sets.map(async (set) => {
    const shown = shownOf(set);
    if (set.table && set.path === shown.path && set.mtime === shown.mtime) return;
    const table = await api.get('/api/explore/rdb', { data_dir: S.data_dir, path: shown.path });
    if (gen !== XP.gen) return;
    Object.assign(set, { path: shown.path, mtime: shown.mtime, fips: {},
      table: table.error || !table.columns ? null : table });
  }));
  if (gen !== XP.gen) return;
  const sets = shownSets();
  const unread = XP.sets.filter((set) => !set.table).map((set) => baseName(set.base));
  exploreMessage(unread.length ? t('explore_unreadable', unread.join(', ')) : '');
  if (!sets.length) return;
  const first = sets[0].table;
  const columns = first.columns.filter((name) => sets.every((set) => set.table.columns.includes(name)));
  const has = (name) => columns.includes(name);
  const own = (name, fallback) => (has(name) ? name : fallback);
  const keep = (key, fallback) => (XP.picked[key] && (has(XP[key]) || (key === 'c' && ['', 'outlier'].includes(XP[key]))) ? XP[key] : fallback);
  XP.x = keep('x', own(first.x, columns[0]));
  XP.y = keep('y', own(first.y, columns[1] || columns[0]));
  XP.c = keep('c', first.c && has(first.c) ? first.c : '');
  const options = (names) => names.map((name) => el('option', { value: name, text: name }));
  $('explore_x').replaceChildren(...options(columns));
  $('explore_y').replaceChildren(...options(columns));
  $('explore_c').replaceChildren(el('option', { value: '', text: t('explore_colour_none') }),
    el('option', { value: 'outlier', text: t('explore_outlier') }), ...options(columns));
  $('explore_x').value = XP.x;
  $('explore_y').value = XP.y;
  $('explore_c').value = XP.c;
  // several tables: their colours are theirs
  $('explore_c').disabled = sets.length > 1;
  drawTableRows();
  drawExplore();
  loadFips();
  // the same templates (a run's two tables): seen where they were
  const same = templatesBefore === sets.map((set) => (set.table.template || {}).path).join('|');
  if (same && XP.tplRange) loadTemplates(XP.tplRange[0], XP.tplRange[1]); else loadTemplates();
}

function drawExplore() {
  const sets = shownSets();
  if (!sets.length || !window.Plotly) return;
  const several = sets.length > 1;
  const bars = (array) => (array ? { type: 'data', array, visible: true, thickness: 1, width: 0, color: css('--axis') } : { visible: false });
  let colourNote = '';
  let colourBar = false;
  const ys = [];
  const traces = sets.map((set) => {
    const table = set.table;
    const values = table.values;
    const errorOf = (name) => (table.errors[name] ? values[table.errors[name]] : null);
    // several: each about its median, so that they sit on each other
    let y = values[XP.y];
    if (several) {
      const mid = median(y);
      y = y.map((v) => (v === null ? null : v - mid));
    }
    ys.push(y);
    const marker = { size: 7, line: { width: 0.6, color: css('--surface') } };
    let colour = null;
    let title = '';
    if (several) marker.color = setColour(set);
    else {
      if (XP.c === 'outlier') {
        const fip = set.fips[XP.y];
        colour = fip && fip.prob && fip.prob.length ? fip.prob : null;
        title = t('explore_outlier_short');
        if (!colour) colourNote = t('explore_outlier_wait');
      } else if (XP.c) {
        colour = values[XP.c];
        title = XP.c;
      }
      if (colour) {
        colourBar = true;
        Object.assign(marker, { color: colour, showscale: true,
          colorbar: { title: { text: title, side: 'right' }, thickness: 12, outlinewidth: 0, tickfont: { color: css('--muted') }, x: 0.735, xanchor: 'left' } });
        // an outlier's probability: blue reliable, grey undecided, red outlier
        if (XP.c === 'outlier') Object.assign(marker, { cmin: 0, cmax: 1, colorscale: [[0, css('--series-1')], [0.5, '#9aa3b2'], [1, css('--series-8')]] });
        else marker.colorscale = 'Viridis';
      } else marker.color = css('--series-1');
    }
    return { type: 'scatter', mode: 'markers', name: tableName(set), x: values[XP.x], y, text: table.labels, marker,
      error_x: bars(errorOf(XP.x)), error_y: bars(errorOf(XP.y)), showlegend: several,
      hovertemplate: `%{text}<br>${esc(XP.x)} %{x}<br>${esc(XP.y)} %{y}${colour ? `<br>${esc(title)} %{marker.color:.4g}` : ''}<extra>${several ? esc(tableName(set)) : ''}</extra>` };
  });
  // on the right, on the same y axis, the histogram of y: of each table
  //   as a fraction of its points when there are several, the same bins
  //   for all
  const pooled = finite(ys.flat()).sort((a, b) => a - b);
  if (pooled.length > 1) {
    const iqr = pooled[Math.floor(0.75 * (pooled.length - 1))] - pooled[Math.floor(0.25 * (pooled.length - 1))];
    const span = pooled[pooled.length - 1] - pooled[0];
    // Freedman-Diaconis, at least 10 bins and at most 400 (a far outlier
    //   does not make the core one bin)
    let size = 2 * iqr / Math.cbrt(pooled.length) || span / 20 || 1;
    if (span > 0) size = Math.min(Math.max(size, span / 400), span / 10);
    sets.forEach((set, i) => traces.push({ type: 'histogram', y: ys[i], xaxis: 'x2', yaxis: 'y', orientation: 'h',
      name: tableName(set), showlegend: false, histnorm: several ? 'probability' : '',
      ybins: { start: pooled[0] - size / 2, end: pooled[pooled.length - 1] + size, size },
      marker: { color: several ? setColour(set) : css('--series-1'), line: { width: 0 } }, opacity: several ? 0.5 : 0.8,
      hovertemplate: `%{y}<br>%{x}<extra>${several ? esc(tableName(set)) : ''}</extra>` }));
  }
  Plotly.react($('explore_plot'), traces, xpLayout({
    uirevision: `${sets.map((set) => set.path).join('|')}|${XP.x}|${XP.y}`, margin: { l: 70, r: 16, t: several ? 28 : 12, b: 48 },
    barmode: 'overlay', bargap: 0.05,
    xaxis: xpAxis({ title: { text: XP.x }, domain: [0, colourBar ? 0.72 : 0.79] }),
    xaxis2: xpAxis({ title: { text: t(several ? 'explore_hist_fraction' : 'explore_hist_count') }, domain: [0.83, 1], anchor: 'y', rangemode: 'tozero' }),
    yaxis: xpAxis({ title: { text: several ? t('explore_minus_median', XP.y) : XP.y } }),
  }), PLOT_CONFIG);
  drawStats(sets, ys);
  const first = sets[0].table;
  const said = [t('explore_rows', sets.reduce((sum, set) => sum + set.table.rows, 0))];
  said.push(first.errors[XP.y] ? t('explore_bars', first.errors[XP.y]) : t('explore_no_bars', XP.y));
  if (several) said.push(t('explore_colour_tables'));
  if (colourNote) said.push(colourNote);
  $('explore_note').textContent = said.join(' · ');
}

/* under the plot, for each table: its points, the rms and the robust rms
   of y, its median error bar */
function drawStats(sets, ys) {
  const stats = sets.map((set, i) => ({ set, ...spread(ys[i]),
    sigma: set.table.errors[XP.y] ? median(finite(set.table.values[set.table.errors[XP.y]])) : NaN }));
  const scale = Math.min(...stats.map((row) => row.robust).filter((v) => v > 0), Infinity);
  const several = sets.length > 1;
  const head = ['explore_col_table', 'explore_col_n', 'explore_col_rms', 'explore_col_robust', 'explore_col_sigma'];
  $('explore_stats').replaceChildren(
    el('thead', {}, el('tr', {}, head.map((key, i) => el('th', { class: i ? 'num' : '', text: i > 1 ? `${t(key)} (${XP.y})` : t(key) })))),
    el('tbody', {}, stats.map((row) => el('tr', {},
      el('td', { class: 'mono' }, several ? el('span', { class: 'swatch', style: `background: ${setColour(row.set)}` }) : null, tableName(row.set)),
      el('td', { class: 'num', text: row.n }),
      el('td', { class: 'num mono', text: figure(row.rms, scale) }),
      el('td', { class: 'num mono', text: figure(row.robust, scale) }),
      el('td', { class: 'num mono', text: figure(row.sigma, scale) })))));
}

/* the quantities whose FIP is shown: y, and x when x is not a time (one
   table); the y of each table (several) */
function fipQuantities() {
  const sets = shownSets();
  if (sets.length !== 1) return [XP.y];
  return [XP.y].concat(!sets[0].table.times.includes(XP.x) && XP.x !== XP.y ? [XP.x] : []);
}
async function loadFips() {
  const sets = shownSets().filter((set) => set.table.time);
  if (!sets.length) { Plotly.purge($('explore_fip')); $('explore_fip_note').textContent = t('explore_fip_no_time'); return; }
  const gen = XP.gen;
  const quantities = fipQuantities();
  const asks = [];
  sets.forEach((set) => quantities.forEach((name) => { if (!set.fips[name]) asks.push([set, name]); }));
  if (asks.length) $('explore_fip_note').textContent = t('explore_fip_wait', [...new Set(asks.map(([, name]) => name))].join(', '));
  await Promise.all(asks.map(async ([set, name]) => {
    const result = await api.get('/api/explore/fip', { data_dir: S.data_dir, path: set.path, column: name });
    if (gen === XP.gen) set.fips[name] = result;
  }));
  if (gen !== XP.gen) return;
  drawFip();
  if (XP.c === 'outlier') drawExplore();
}

function drawFip() {
  if (!window.Plotly) return;
  const sets = shownSets().filter((set) => set.table.time);
  const several = sets.length > 1;
  const quantities = fipQuantities();
  const traces = [];
  const said = [];
  let top = 0;
  let longest = 0;
  sets.forEach((set) => quantities.forEach((name, i) => {
    const fip = set.fips[name];
    if (!fip) return;
    const who = several ? tableName(set) : name;
    if (!fip.period || !fip.period.length) { said.push(`${who}: ${fip.error || t('explore_fip_none')}`); return; }
    // one table: a colour per quantity; several: that of each table, the
    //   gaussian curve with the outlier-aware one in the legend
    const colour = several ? setColour(set) : css(['--series-1', '--series-2'][i % 2]);
    const hover = (kind) => `${esc(who)}, ${kind}<br>%{x:.4f} d<br>-log10 FIP %{y:.2f}<extra></extra>`;
    traces.push({ x: fip.period, y: fip.gauss, mode: 'lines', name: `${who}, ${t('explore_gauss')}`, legendgroup: who,
      showlegend: !several, line: { color: colour, width: 1, dash: 'dash' }, opacity: 0.75, hovertemplate: hover(t('explore_gauss')) });
    traces.push({ x: fip.period, y: fip.oa, mode: 'lines', name: several ? who : `${who}, ${t('explore_oa')}`, legendgroup: who,
      line: { color: colour, width: 1.8 }, hovertemplate: hover(t('explore_oa')) });
    top = Math.max(top, ...fip.oa, ...fip.gauss);
    longest = Math.max(longest, fip.period[fip.period.length - 1]);
    const best = fip.best.oa;
    said.push(t('explore_fip_best', several ? `${tableName(set)} (${name})` : name, best.period.toFixed(3), fipText(best.fip), fip.outliers, fip.points)
      + (fip.err ? '' : ` (${t('explore_fip_noerr')})`));
  }));
  if (!traces.length) { Plotly.purge($('explore_fip')); $('explore_fip_note').textContent = said.join(' · '); return; }
  // 1 % dotted (koloa's), and the year where the periods reach it
  const shapes = [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 2, y1: 2, line: { color: css('--muted'), width: 1, dash: 'dot' } }];
  const notes = [];
  if (longest > 365.25) {
    shapes.push({ type: 'line', xref: 'x', x0: 365.25, x1: 365.25, yref: 'paper', y0: 0, y1: 1, line: { color: css('--muted'), width: 1, dash: 'dot' } });
    notes.push({ x: Math.log10(365.25), xref: 'x', y: 1, yref: 'paper', yanchor: 'bottom', text: t('explore_year'), showarrow: false, font: { size: 10, color: css('--muted') } });
  }
  Plotly.react($('explore_fip'), traces, xpLayout({
    uirevision: sets.map((set) => set.path).join('|'), shapes, annotations: notes,
    xaxis: xpAxis({ type: 'log', dtick: 'D2', title: { text: t('explore_period_axis') } }),
    yaxis: xpAxis({ title: { text: t('explore_fip_axis') }, range: [0, Math.max(2.5, top * 1.08)] }),
  }), PLOT_CONFIG);
  $('explore_fip_note').textContent = said.join(' · ');
}

// ── template explorer ───────────────────────────────────────────────────
// the template of each table over the wavelengths shown (one for the
//   tables that share it), and under them a quarter as high, on the same
//   wavelengths, the TAPAS transmission of each molecule (the first
//   table's instrument); a zoom asks the server for that range again
//   (every point once narrow enough)
let templateAsked = '';
let templateTimer = null;
async function loadTemplates(lo, hi) {
  const groups = [];
  shownSets().forEach((set) => {
    const path = set.table.template ? set.table.template.path : `none:${set.path}`;
    const group = groups.find((g) => g.path === path);
    if (group) group.sets.push(set); else groups.push({ path, sets: [set] });
  });
  if (!groups.length) return;
  const n = Math.round(Math.max(400, $('template_plot').clientWidth || 1000));
  const key = JSON.stringify([groups.map((g) => g.path), lo, hi, n]);
  templateAsked = key;
  // in the star's rest frame: the velocity typed, else the one found
  const vsys = XP.rest ? (XP.vsys === null ? 'auto' : XP.vsys) : '';
  if (XP.showLines && !XP.lines) XP.lines = await api.get('/api/explore/lines');
  const data = await Promise.all(groups.map((group, i) => api.get('/api/explore/template', {
    data_dir: S.data_dir, path: group.sets[0].path, lo: lo ?? '', hi: hi ?? '', n, tapas: i ? 0 : 1, vsys })));
  if (templateAsked !== key) return;
  XP.tpls = groups.map((group, i) => ({ ...group, data: data[i] }));
  XP.tplRange = lo === undefined ? null : [lo, hi];
  drawTemplate();
}

function drawTemplate() {
  const div = $('template_plot');
  if (!window.Plotly) return;
  const tpls = (XP.tpls || []).filter((tpl) => tpl.data && tpl.data.template);
  if (!tpls.length) {
    if (!(XP.tpls || []).length) return;
    Plotly.purge(div);
    div.hidden = true;
    $('template_note').textContent = t('template_none');
    return;
  }
  div.hidden = false;
  const several = tpls.length > 1;
  const traces = tpls.map((tpl) => {
    // several: each divided by its median over what is shown
    let y = tpl.data.y;
    if (several) {
      const mid = median(y) || 1;
      y = y.map((v) => (v === null ? null : v / mid));
    }
    const name = tpl.sets.map(tableName).join(', ');
    return { x: tpl.data.x, y, mode: 'lines', name, showlegend: several, legendgroup: 'templates',
      line: { color: several ? setColour(tpl.sets[0]) : css('--text'), width: 1 }, xaxis: 'x', yaxis: 'y',
      hovertemplate: `%{x:.4f} nm<br>%{y:.4f}<extra>${several ? esc(tpl.data.template.name) : ''}</extra>` };
  });
  const tapas = tpls[0].data.tapas;
  Object.entries((tapas && tapas.molecules) || {}).forEach(([name, mol]) => {
    const label = name === 'others' ? t('template_others') : chemical(name);
    traces.push({ x: mol.x, y: mol.y, mode: 'lines', name: label, legendgroup: 'tapas',
      line: { color: css(MOLECULE_COLOURS[name] || '--series-other'), width: 1 }, xaxis: 'x', yaxis: 'y2',
      hovertemplate: `${label} %{x:.4f} nm<br>${t('template_transmission_short')} %{y:.3f}<extra></extra>` });
  });
  const range = XP.tplRange || [Math.min(...tpls.map((tpl) => tpl.data.range[0])), Math.max(...tpls.map((tpl) => tpl.data.range[1]))];
  // y: the range cut by the user, else that of what is seen
  let yRange = XP.tplY;
  if (!yRange) {
    const seen = traces.filter((tr) => tr.yaxis === 'y').flatMap((tr) => tr.y.filter((v, i) => v !== null && tr.x[i] >= range[0] && tr.x[i] <= range[1]));
    if (seen.length) {
      const lo = Math.min(...seen);
      const hi = Math.max(...seen);
      const pad = 0.04 * (hi - lo || Math.abs(hi) || 1);
      yRange = [lo - pad, hi + pad];
    }
  }
  // copies: Plotly keeps the arrays it is given, and changes them in place
  //   when the plot is zoomed
  XP.tplYShown = yRange ? [...yRange] : null;
  // the lines of interest in the template's frame: at their vacuum
  //   wavelengths in the star's, moved by v_sys otherwise
  const vsys = tpls[0].data.vsys || {};
  const used = vsys.shift ?? null;
  const moved = !used && vsys.value !== null && vsys.value !== undefined ? vsys.value : 0;
  const factor = moved ? Math.sqrt((1 + moved / C_KMS) / (1 - moved / C_KMS)) : 1;
  const shapes = [];
  const notes = [];
  if (XP.showLines && XP.lines) {
    XP.lines.forEach((line) => {
      const centre = line.wave * factor;
      if (centre < range[0] || centre > range[1]) return;
      const colour = css(line.kind === 'flare' ? '--series-2' : '--series-3');
      shapes.push({ type: 'rect', xref: 'x', yref: 'y domain', x0: centre - line.width * factor / 2, x1: centre + line.width * factor / 2,
        y0: 0, y1: 1, fillcolor: colour, opacity: 0.18, line: { width: 0 }, layer: 'below' });
      shapes.push({ type: 'line', xref: 'x', yref: 'y domain', x0: centre, x1: centre, y0: 0, y1: 1, opacity: 0.55, line: { color: colour, width: 1, dash: 'dot' } });
      notes.push({ xref: 'x', yref: 'y domain', x: centre, y: 0.99, yanchor: 'top', xanchor: 'right', textangle: -90, showarrow: false,
        text: esc(line.label), font: { size: 10, color: css('--muted') },
        hovertext: `${esc(line.label)} · ${esc(line.species)} · ${line.wave.toFixed(4)} nm (vacuum)` });
    });
  }
  XP.tplDraw += 1;
  // the footer a quarter as high as the templates, one x axis for both; the
  //   axes' ranges are the window's own (uirevision a new one each time),
  //   the rest of what the user did (the legend) is kept
  Plotly.react(div, traces, xpLayout({
    uirevision: tpls.map((tpl) => tpl.path).join('|'), margin: { l: 70, r: 16, t: several ? 48 : 28, b: 48 },
    legend: { orientation: 'h', x: 1, xanchor: 'right', y: 1.0, yanchor: 'bottom', font: { size: 11 } },
    shapes, annotations: notes,
    xaxis: xpAxis({ title: { text: t('template_wavelength') }, range: [...range], autorange: false, anchor: 'y2', uirevision: XP.tplDraw }),
    yaxis: xpAxis({ domain: [0.25, 1], title: { text: t(several ? 'template_flux_median' : 'template_flux') },
      range: yRange ? [...yRange] : undefined, autorange: !yRange, uirevision: XP.tplDraw }),
    // transmission, 1 clear to 0 opaque
    yaxis2: xpAxis({ domain: [0, 0.1875], range: [0, 1.04], fixedrange: true, tickvals: [0, 0.5, 1],
      title: { text: t('template_transmission'), font: { size: 11 } } }),
  }), PLOT_CONFIG).then(() => {
    if (div.dataset.zoom) return;
    div.dataset.zoom = '1';
    div.on('plotly_relayout', (event) => {
      const pair = (axis) => (event[`${axis}.range[0]`] !== undefined ? [event[`${axis}.range[0]`], event[`${axis}.range[1]`]]
        : (Array.isArray(event[`${axis}.range`]) ? event[`${axis}.range`] : null));
      const zoom = pair('xaxis');
      const cut = pair('yaxis');
      if (event['xaxis.autorange'] || event['yaxis.autorange']) {
        // a double click: all of it, y to what is seen
        XP.tplY = null;
        clearTimeout(templateTimer);
        templateTimer = setTimeout(() => loadTemplates(), 120);
        return;
      }
      if (!zoom && !cut) return;
      // y cut by a box or a scroll: kept; moved in x alone (a thin box, or
      //   a pan that keeps y's extent when y was not cut): y follows what
      //   is seen
      // the extent of y as it was drawn (the layout holds the new one)
      const before = XP.tplYShown || [0, 0];
      const panned = cut && !XP.tplY && Math.abs((cut[1] - cut[0]) - (before[1] - before[0])) < 1e-9 * Math.abs(before[1] - before[0] || 1);
      XP.tplY = cut && !panned ? cut : (cut ? XP.tplY : null);
      clearTimeout(templateTimer);
      templateTimer = setTimeout(() => (zoom ? loadTemplates(zoom[0], zoom[1]) : drawTemplate()), 120);
    });
  });
  const said = [t(several ? 'template_several' : 'template_of', tpls.map((tpl) => tpl.data.template.name).join(', '))];
  if (used !== null) said.push(t('template_shifted', Number(used).toFixed(2), XP.vsys === null ? (vsys.source || '') : t('template_vsys_typed')));
  else if (XP.rest && (vsys.value === null || vsys.value === undefined)) said.push(t('template_vsys_unknown'));
  else if (moved && XP.showLines) said.push(t('template_lines_moved', Number(moved).toFixed(2)));
  // the velocity found shown in the field, unless one was typed
  if (XP.vsys === null) $('template_vsys').value = vsys.value ?? '';
  if (tapas) {
    said.push(t(tapas.kind === 'molecules' ? 'template_tapas' : 'template_tapas_lbl', tapas.file));
    const bervs = shownSets().map((set) => set.table.berv).filter(Boolean);
    if (bervs.length) said.push(t('template_berv', Math.min(...bervs.map((b) => b[0])).toFixed(1), Math.max(...bervs.map((b) => b[1])).toFixed(1)));
  } else said.push(t('template_no_tapas'));
  const missing = (XP.tpls || []).filter((tpl) => !(tpl.data && tpl.data.template)).flatMap((tpl) => tpl.sets.map(tableName));
  if (missing.length) said.push(t('template_missing', missing.join(', ')));
  $('template_note').textContent = said.join(' · ');
}

/* everything the tab shows, drawn again (a theme, a language) */
function redrawExplorer() {
  if (S.tab !== 'explore' || !window.Plotly || !XP.sets.length) return;
  drawTableRows();
  drawExplore();
  drawFip();
  drawTemplate();
}

// ── background ──────────────────────────────────────────────────────────
// blurred figures of papers whose velocities LBL measured, one mosaic per
//   theme (maintenance/gui/paper_backdrop.py); the References tab
//   credits them
let backdropFor = null;
// the Python this window runs with: the wraps' by default
let DEFAULT_PYTHON = '';
function setBackdrop() {
  const theme = S.theme === 'dark' ? 'dark' : 'light';
  if (backdropFor === theme) return;
  backdropFor = theme;
  const url = `/static/backdrop_${theme}.jpg`;
  const image = new Image();
  image.onload = () => {
    if (backdropFor !== theme) return;
    $('backdrop').style.backgroundImage = `url("${url}")`;
    $('backdrop').classList.add('ready');
  };
  image.src = url;
}

// ── start ───────────────────────────────────────────────────────────────
async function init() {
  const state = await api.get('/api/state');
  Object.assign(S, JSON.parse(JSON.stringify(DEFAULTS)), state.state || {});
  // the window opens on every target, none ticked, no instrument hidden,
  //   and the instrument and source detected again from what gets ticked
  S.ticked = [];
  S.hidden_instruments = [];
  S.setup_manual = false;
  // a window of before the roots: its data folder is its root
  if (!S.root) S.root = S.data_dir;
  document.documentElement.dataset.theme = S.theme;
  DEFAULT_PYTHON = state.python;
  S.python = S.python || state.python;
  $('version').textContent = state.version;
  $('lbl_home').textContent = state.lbl;
  C_KMS = state.c_kms;
  await loadStrings();
  INSTRUMENTS = await api.get('/api/instruments');
  fillInstruments();
  $('data_dir').value = S.root;
  $('python').value = S.python;
  $('min_rjd').value = S.min_rjd ?? '';
  $('max_rjd').value = S.max_rjd ?? '';
  $('run_name').value = S.name || '';

  $('theme').onclick = switchTheme;
  $('lang').onclick = async () => { S.lang = S.lang === 'fr' ? 'en' : 'fr'; save(); await loadStrings(); await loadInstrumentThings(); drawTargets(); drawPlots(); pollQueue(); if (S.tab === 'runs') loadRuns(); redrawExplorer(); };
  document.querySelectorAll('#tabs button').forEach((b) => { b.onclick = () => showTab(b.dataset.tab); });
  $('scan').onclick = scan;
  $('data_folder').onchange = async (event) => { S.folder = event.target.value; await openFolder(); };
  $('downloads').onclick = loadFolders;
  $('dispatch').onclick = startDispatch;
  $('dispatch_go').onclick = async () => { DISPATCH = await api.post('/api/dispatch/go', { root: S.root }); followDispatch(); };
  $('dispatch_cancel').onclick = async () => { DISPATCH = await api.post('/api/dispatch/cancel', {}); drawDispatch(); };
  $('data_dir').onkeydown = (event) => { if (event.key === 'Enter') scan(); };
  $('instrument').onchange = async (event) => { S.instrument = event.target.value; S.instrument_set = true; S.setup_manual = true; fillSources(); save(); await loadInstrumentThings(); wrapEdited = false; drawSetupStatus(); };
  $('source').onchange = async (event) => { S.source = event.target.value; S.setup_manual = true; save(); await loadInstrumentThings(); wrapEdited = false; drawSetupStatus(); };
  $('python').onchange = (event) => { S.python = event.target.value.trim() || DEFAULT_PYTHON; $('python').value = S.python; save(); drawSetupStatus(); wrapEdited = false; };
  $('tick_all').onclick = () => { S.ticked = TARGETS.filter((row) => S.folder || !S.hidden_instruments.includes(row.instrument)).map((row) => row.name); save(); drawTargets(); loadPoints(); drawPerTarget(); followTicked(); };
  $('tick_none').onclick = () => { S.ticked = []; save(); drawTargets(); loadPoints(); drawPerTarget(); drawSetupStatus(); };
  const bound = (key) => (event) => { S[key] = event.target.value === '' ? null : Number(event.target.value); save(); drawPlots(); wrapEdited = false; };
  $('min_rjd').onchange = bound('min_rjd');
  $('max_rjd').onchange = bound('max_rjd');
  $('pca2d_on').onchange = (event) => { S.pca2d_on = event.target.checked; save(); drawOptions(); wrapEdited = false; };
  $('skip_pca2d').onchange = (event) => { S.skip_pca2d = event.target.value === '' ? null : event.target.value === 'true'; save(); wrapEdited = false; };
  $('resproj').onchange = (event) => { S.resproj = event.target.checked; save(); wrapEdited = false; };
  $('run_name').oninput = (event) => { S.name = event.target.value.trim(); S.name_auto = false; save(); };
  $('name_auto').onclick = () => { S.name_auto = true; save(); build(); };
  $('wrap_text').oninput = () => { wrapEdited = true; highlightWrap(); };
  $('wrap_text').onscroll = syncWrapScroll;
  $('check').onclick = build;
  $('load_wrap').onclick = loadWrap;
  $('wrap_list').onchange = (event) => { $('wrap_path').value = event.target.value; };
  $('queue').onclick = queue;
  $('save_wrap').onclick = () => saveWrap(false);
  $('stop').onclick = stopJob;
  $('overwrite_all').onchange = (event) => setOverwrite(event.target.checked);
  // to the top: the log stops following the run; to the bottom: it follows again
  for (const id of ['log', 'archive_log']) {
    $(`${id}_top`).onclick = () => { $(id).scrollTop = 0; };
    $(`${id}_bottom`).onclick = () => { const box = $(id); box.scrollTop = box.scrollHeight; };
    $(id).onscroll = () => logEdges(id);
    logEdges(id);
  }
  $('runs_refresh').onclick = loadRuns;
  $('measure').onclick = measure;
  $('explore_refresh').onclick = loadExplorer;
  $('explore_add').onclick = addTable;
  $('explore_nightly').onchange = (event) => { XP.nightly = event.target.checked; openSets(); };
  $('explore_x').onchange = (event) => { XP.x = event.target.value; XP.picked.x = true; drawExplore(); loadFips(); };
  $('explore_y').onchange = (event) => { XP.y = event.target.value; XP.picked.y = true; drawExplore(); loadFips(); };
  $('explore_c').onchange = (event) => { XP.c = event.target.value; XP.picked.c = true; drawExplore(); };
  const sameWindow = () => (XP.tplRange ? loadTemplates(XP.tplRange[0], XP.tplRange[1]) : loadTemplates());
  $('template_lines').onchange = (event) => { XP.showLines = event.target.checked; sameWindow(); };
  $('template_rest').onchange = (event) => { XP.rest = event.target.checked; sameWindow(); };
  $('template_vsys').onchange = (event) => {
    const value = event.target.value.trim();
    XP.vsys = value === '' || !Number.isFinite(Number(value)) ? null : Number(value);
    if (XP.vsys !== null) { XP.rest = true; $('template_rest').checked = true; }
    sameWindow();
  };
  $('best_all').onclick = () => bestMatch(S.ticked);
  $('simbad_teff').onclick = () => askSimbad(S.ticked, true);
  // read the folder again: new or removed targets and spectra, and the
  //   headers of the files that changed (the others are kept)
  $('refresh_targets').onclick = async () => {
    if (!S.data_dir) return;
    await api.post('/api/scan', { data_dir: S.data_dir, again: true });
    if (scanTimer) clearInterval(scanTimer);
    scanTimer = setInterval(loadTargets, 1500);
    await loadTargets();
  };
  document.querySelectorAll('#target_subtabs button').forEach((b) => { b.onclick = () => showSub(b.dataset.sub); });
  $('archive_search').onclick = archiveSearch;
  $('archive_target').onkeydown = (event) => { if (event.key === 'Enter') archiveSearch(); };
  $('archive_fetch').onclick = archiveFetch;
  $('archive_stop').onclick = async () => { ARCHIVE = await api.post('/api/archive/stop', {}); drawArchive(); };
  $('archive_search_stop').onclick = async () => { ARCHIVE = await api.post('/api/archive/search/stop', {}); drawArchive(); };
  $('delete').onclick = deleteSelected;
  $('cleanup').addEventListener('change', cleanupSelection);

  await loadFolders();
  await loadInstrumentThings();
  await loadTargets();
  await followTicked();
  setBackdrop();
  // a plan or a dispatch going on (the page opened again meanwhile)
  DISPATCH = await api.get('/api/dispatch');
  if (['planning', 'moving'].includes(DISPATCH.state)) followDispatch(); else drawDispatch();
  showTab(S.tab || 'targets');
  showSub(S.sub || 'local');
  pollQueue();
  setInterval(pollQueue, 2000);
}

init();
