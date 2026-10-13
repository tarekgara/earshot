/* Shared UI helpers: data lookups, persistence, waveform drawing, play buttons. */
(function () {
  'use strict';

  // The engine calls entries "instruments" internally; Earshot's data calls them sounds.
  const LIB = Object.assign({}, window.EARSHOT, { instruments: window.EARSHOT.sounds });
  const byId = new Map(LIB.instruments.map(i => [i.id, i]));
  const famById = new Map(LIB.families.map(f => [f.id, f]));

  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function sample(inst, key) { return inst && inst.samples.find(s => s.key === key); }

  /** The one clip that best represents a sound at a glance. */
  function canonical(inst) {
    if (!inst || !inst.samples.length) return null;
    return sample(inst, 'ex-1') || inst.samples[0];
  }

  /** Nearest single-note sample to a MIDI pitch, or the canonical clip. */
  function nearestNote(inst, midi) {
    const notes = inst.samples.filter(s => s.group === 'register' && s.midi != null);
    if (!notes.length || midi == null) return canonical(inst);
    return notes.reduce((a, b) => (Math.abs(b.midi - midi) < Math.abs(a.midi - midi) ? b : a));
  }

  /** Find a sample by "instrument/key". */
  function resolve(ref) {
    const [id, key] = ref.split('/');
    const inst = byId.get(id);
    return inst ? { inst, sample: sample(inst, key) } : null;
  }

  const store = {
    get(k, d) { try { const v = localStorage.getItem('es.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('es.' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
  };

  /* ---- compare list (shared by every view) ---- */
  const compare = {
    ids: store.get('compare', []).filter(id => byId.has(id)),
    add(id) { if (!this.ids.includes(id)) { this.ids.push(id); if (this.ids.length > 6) this.ids.shift(); this.save(); } },
    remove(id) { this.ids = this.ids.filter(x => x !== id); this.save(); },
    set(ids) { this.ids = ids.filter(id => byId.has(id)).slice(0, 6); this.save(); },
    has(id) { return this.ids.includes(id); },
    save() { store.set('compare', this.ids); document.dispatchEvent(new CustomEvent('compare-change')); },
  };

  /* ---- markup ---- */

  const icon = (name, cls = '') => `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;

  function playBtn(inst, s, size = '', label) {
    if (!s) return `<span class="play ${size}" aria-hidden="true" style="visibility:hidden"></span>`;
    const name = label || `Play ${inst.name}, ${s.label}`;
    return `<button class="play ${size}" type="button" data-fam="${inst.family}" data-play="${inst.id}/${s.key}" aria-label="${esc(name)}">${icon('play', 'i-play')}${icon('stop', 'i-stop')}</button>`;
  }

  function famDot(inst) { return `<span class="dot" data-fam="${inst.family}"></span>`; }

  function rangeText(inst) {
    if (!inst.range) return '';
    return window.DSP.noteName(inst.range[0]) + '–' + window.DSP.noteName(inst.range[1]);
  }

  function copyPath(inst) {
    const fam = famById.get(inst.family);
    return [fam && fam.name, inst.sub, inst.name].filter(Boolean).filter((v, i, a) => a.indexOf(v) === i).join(' › ');
  }

  /* ---- canvas ---- */

  function fit(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return { w, h, dpr, g: canvas.getContext('2d') };
  }

  function cssVar(name, el = document.documentElement) {
    return getComputedStyle(el).getPropertyValue(name).trim();
  }

  /** Mirrored bar waveform. progress in 0..1 paints the played part in `hot`. */
  function drawWave(canvas, peaks, opt = {}) {
    if (!canvas || !peaks) return;
    const { w, h, dpr, g } = fit(canvas);
    g.clearRect(0, 0, w, h);
    const bar = Math.max(1, Math.round((opt.bar || 2) * dpr)), gap = Math.max(1, Math.round((opt.gap || 1) * dpr));
    const n = Math.max(1, Math.floor(w / (bar + gap)));
    const cold = opt.cold || cssVar('--text-3');
    const hot = opt.hot || cssVar('--accent');
    const prog = opt.progress == null ? -1 : opt.progress;
    const mid = h / 2;
    for (let i = 0; i < n; i++) {
      const a = Math.floor(i * peaks.length / n), b = Math.max(a + 1, Math.floor((i + 1) * peaks.length / n));
      let p = 0;
      for (let j = a; j < b; j++) p = Math.max(p, peaks[j]);
      const amp = Math.max(dpr, Math.pow(p, 0.8) * (h * 0.46));
      g.fillStyle = i / n <= prog ? hot : cold;
      g.globalAlpha = opt.alpha == null ? 1 : opt.alpha;
      g.fillRect(i * (bar + gap), mid - amp, bar, amp * 2);
    }
    g.globalAlpha = 1;
  }

  /* ---- playing-state sync ---- */

  function syncPlaying() {
    const id = window.Player.state.current && window.Player.state.current.id;
    document.querySelectorAll('[data-play]').forEach(b => {
      if (b.dataset.play === id) b.setAttribute('data-playing', ''); else b.removeAttribute('data-playing');
    });
  }

  let toastTimer = 0;
  function toast(msg) {
    const t = document.querySelector('[data-toast]');
    t.textContent = msg;
    t.setAttribute('data-show', '');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.removeAttribute('data-show'), 2400);
  }

  async function copy(text) {
    try { await navigator.clipboard.writeText(text); toast('Copied'); }
    catch (e) {
      const ta = document.createElement('textarea');
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); toast('Copied'); } catch (err) { toast('Copy failed. Select the text instead.'); }
      ta.remove();
    }
  }

  /** Score how well each sound fits a set of trait answers (0..1). */
  function scoreInstruments(answers) {
    const qs = LIB.questions;
    const total = qs.reduce((s, q) => s + (answers[q.id] ? q.weight : 0), 0);
    return LIB.instruments.map(inst => {
      let s = 0;
      for (const q of qs) {
        const a = answers[q.id];
        if (!a) continue;
        const have = inst.traits[q.id] || [];
        const at = have.indexOf(a);
        if (at < 0) { s -= q.weight * 0.6; continue; }
        // A trait the sound always has counts more than one it merely can have.
        const spec = q.id === 'tone' ? (at === 0 ? 1 : 0.72) : (have.length === 1 ? 1 : have.length === 2 ? 0.82 : 0.68);
        s += q.weight * spec;
      }
      const score = total ? Math.max(0, (s + total * 0.6) / (total * 1.6)) : 0.5;
      return { inst, score };
    }).sort((a, b) => b.score - a.score || (b.inst.samples.length > 0) - (a.inst.samples.length > 0));
  }

  /* ---- vocabulary: every word people use for a sound ---- */

  const KIND_RANK = { name: 0, synonym: 1, written: 2, loose: 3 };
  const KIND_LABEL = { name: 'Main name', synonym: 'Synonym', written: 'Written as', loose: 'Loosely used for' };
  // "groaning", "groans", "groaned" should all find "groan".
  const stem = w => w.replace(/(ing|ed|es|s)$/, '');

  let wordMap = null;
  function wordIndex() {
    if (wordMap) return wordMap;
    wordMap = new Map();
    const add = (raw, inst, kind) => {
      const w = String(raw || '').toLowerCase().trim();
      if (!w) return;
      if (!wordMap.has(w)) wordMap.set(w, []);
      const list = wordMap.get(w);
      const hit = list.find(e => e.inst === inst);
      if (!hit) list.push({ inst, kind });
      else if (KIND_RANK[kind] < KIND_RANK[hit.kind]) hit.kind = kind;
    };
    LIB.instruments.forEach(i => {
      add(i.name, i, 'name');
      (i.aliases || []).forEach(a => add(a, i, 'synonym'));
      const w = i.words || {};
      (w.synonyms || []).forEach(x => add(x, i, 'synonym'));
      (w.written || []).forEach(x => add(x, i, 'written'));
      (w.loose || []).forEach(x => add(x, i, 'loose'));
    });
    wordMap.forEach(list => list.sort((a, b) => KIND_RANK[a.kind] - KIND_RANK[b.kind] || (b.inst.samples.length > 0) - (a.inst.samples.length > 0)));
    return wordMap;
  }

  /** Words matching a query, best first: exact, same word stem, starts with, contains. */
  function findWords(q, limit = 12) {
    q = String(q || '').trim().toLowerCase();
    if (!q) return [];
    const qs = stem(q);
    const out = [];
    for (const [word, sounds] of wordIndex()) {
      let r = -1;
      if (word === q) r = 0;
      else if (qs.length > 2 && stem(word) === qs) r = 0.5;
      else if (word.startsWith(q)) r = 1;
      else if (word.split(/[\s-]/).some(p => p.startsWith(q))) r = 1.5;
      else if (q.length > 2 && word.includes(q)) r = 2;
      if (r >= 0) out.push({ word, sounds, r });
    }
    return out.sort((a, b) => a.r - b.r || b.sounds.length - a.sounds.length || a.word.localeCompare(b.word)).slice(0, limit);
  }

  window.UI = {
    KIND_RANK, KIND_LABEL, wordIndex, findWords,
    LIB, byId, famById, esc, sample, canonical, nearestNote, resolve, store, compare,
    icon, playBtn, famDot, rangeText, copyPath, fit, cssVar, drawWave, syncPlaying, toast, copy, scoreInstruments,
  };
})();
