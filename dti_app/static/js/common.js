/* Shared helpers. Plain JavaScript, no build step. */
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, body, method) {
  const opt = { method: method || (body ? "POST" : "GET") };
  if (body instanceof FormData) opt.body = body;
  else if (body) { opt.headers = { "Content-Type": "application/json" }; opt.body = JSON.stringify(body); }
  let r;
  try { r = await fetch(path, opt); } catch (e) { throw new Error("Could not reach the server."); }
  let data = {};
  try { data = await r.json(); } catch (e) { /* non-JSON body */ }
  if (!r.ok) throw new Error(data.error || `Request failed (${r.status})`);
  return data;
}

function fmtConc(nM) {
  if (nM < 1) return `${(nM * 1000).toPrecision(3)} pM`;
  if (nM < 1000) return `${Number(nM.toPrecision(3))} nM`;
  if (nM < 1e6) return `${Number((nM / 1000).toPrecision(3))} µM`;
  return `${Number((nM / 1e6).toPrecision(3))} mM`;
}

const badge = (label) => `<span class="potency-badge ${esc(label).toLowerCase()}"><span style="width:6px;height:6px;border-radius:50%;background:currentColor;display:inline-block;margin-right:4px;"></span>${esc(label)}</span>`;
const unitName = (ep) => (ep === "Ki" ? "pKi" : "pIC50");

function showError(el, msg) { 
  if (!el) return;
  el.textContent = msg; 
  el.hidden = !msg; 
}

async function pollJob(jobId, onProgress, intervalMs = 800) {
  for (;;) {
    const st = await api(`/api/jobs/${jobId}`);
    if (onProgress) onProgress(st);
    if (st.state === "done") return st.result;
    if (st.state === "error") throw new Error(st.error || "The job failed.");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

function bindProgress(prefix = "") {
  const box = $("#progress"), fill = $("#bar-fill"), msg = $("#progress-msg");
  return {
    show() { 
      if (box) box.hidden = false; 
      if (fill) fill.style.width = "0%"; 
      if (msg) msg.textContent = "Initializing computation…"; 
    },
    update(st) { 
      if (fill) fill.style.width = `${Math.round((st.progress || 0) * 100)}%`; 
      if (msg) msg.textContent = `${st.message || st.state} (${Math.round((st.progress || 0) * 100)}%)`; 
    },
    hide() { 
      if (box) box.hidden = true; 
    },
  };
}

let _examples = null;
async function getExamples() {
  if (!_examples) {
    try { _examples = await api("/api/examples"); } catch (e) { _examples = { drugs: [], targets: [], docking: {} }; }
  }
  return _examples;
}

/* Target picker: example dropdown, UniProt fetch, or pasted sequence. */
class TargetPicker {
  constructor(root) {
    this.root = root;
    this.sel = $(".tp-example", root); this.acc = $(".tp-uniprot", root);
    this.seq = $(".tp-seq", root); this.status = $(".tp-status", root);
    this.name = "";
    const fetchBtn = $(".tp-fetch", root);
    if (fetchBtn) fetchBtn.addEventListener("click", () => this.fetch());
    if (this.sel) {
      this.sel.addEventListener("change", () => { 
        if (this.sel.value) { 
          this.acc.value = this.sel.value; 
          this.fetch(); 
        } 
      });
    }
    if (this.seq) {
      this.seq.addEventListener("input", () => { 
        this.name = ""; 
        if (this.status) {
          const len = this.seq.value.replace(/[^A-Za-z]/g, '').length;
          this.status.textContent = len > 0 ? `${len} residues entered.` : "";
        }
      });
    }
    getExamples().then((ex) => {
      if (this.sel) {
        (ex.targets || []).forEach((t) => {
          const o = document.createElement("option"); 
          o.value = t.uniprot; 
          o.textContent = `${t.name} (${t.uniprot})`;
          this.sel.appendChild(o);
        });
      }
    });
  }
  async fetch() {
    const acc = this.acc.value.trim();
    if (!acc) { 
      if (this.status) this.status.textContent = "Please enter a UniProt accession code (e.g. P00533)."; 
      return; 
    }
    if (this.status) this.status.textContent = "Querying UniProt REST database…";
    try {
      const r = await api(`/api/uniprot/${encodeURIComponent(acc)}`);
      this.seq.value = r.sequence; 
      this.name = r.name;
      if (this.status) this.status.textContent = `✓ ${r.name}: ${r.length} amino acid residues loaded.`;
    } catch (e) { 
      if (this.status) this.status.textContent = `✕ ${e.message}`; 
    }
  }
  value() { return this.seq ? this.seq.value.trim() : ""; }
  uniprot() { return this.acc ? this.acc.value.trim() : ""; }
}

function initTabs(scope = document) {
  const buttons = Array.from(scope.querySelectorAll(".tab-pill, .tab-btn"));
  buttons.forEach((btn) => btn.addEventListener("click", () => {
    const group = btn.parentElement;
    Array.from(group.querySelectorAll(".tab-pill, .tab-btn")).forEach((b) => b.classList.toggle("active", b === btn));
    const targetId = btn.dataset.tab;
    buttons.forEach((b) => {
      const t = document.getElementById(b.dataset.tab);
      if (t) t.hidden = b !== btn;
    });
    document.dispatchEvent(new CustomEvent("tabshown", { detail: targetId }));
  }));
}

/* Result table shared by Batch and Screen. */
function resultsTable(res, { thumbs = false, limit = 200 } = {}) {
  const eps = res.endpoints, rows = res.rows.slice(0, limit);
  let h = '<div class="table-container"><table class="bio-table"><thead><tr><th style="width:50px;">Rank</th>' + 
          (thumbs ? '<th style="width:140px;">2D Structure</th>' : "") + 
          '<th>Compound Name</th><th>SMILES String</th>';
  eps.forEach((ep) => { 
    h += `<th style="text-align:right;">${unitName(ep)}</th><th style="text-align:right;">${ep} Conc.</th><th style="text-align:center;">${ep} Potency</th>`; 
  });
  h += "<th>Training Familiarity</th></tr></thead><tbody>";

  rows.forEach((r) => {
    const rankBadge = r.rank === 1 ? '🥇 1' : r.rank === 2 ? '🥈 2' : r.rank === 3 ? '🥉 3' : `#${r.rank}`;
    h += `<tr><td style="font-weight:700; color:var(--text-main);">${rankBadge}</td>` + 
      (thumbs ? `<td><div class="mol-frame" style="min-height:90px; padding:4px;">${r.svg || ""}</div></td>` : "") +
      `<td><b>${esc(r.name || "Compound")}</b></td>
       <td><code style="font-size:0.78rem;">${esc(r.smiles.length > 45 ? r.smiles.slice(0, 45) + "…" : r.smiles)}</code></td>`;
    
    eps.forEach((ep) => { 
      h += `<td style="text-align:right; font-family:var(--font-mono); font-weight:700; color:var(--cyan-primary);">${r[ep + "_p"].toFixed(2)}</td>
            <td style="text-align:right; font-family:var(--font-mono); font-size:0.85rem;">&asymp; ${esc(r[ep + "_conc"])}</td>
            <td style="text-align:center;">${badge(r[ep + "_label"])}</td>`; 
    });
    
    const sim = esc(r.applicability || "–");
    const simBadge = sim === "high" ? '<span style="color:var(--strong-color); font-weight:600;">High</span>' :
                     sim === "medium" ? '<span style="color:var(--mod-color); font-weight:600;">Medium</span>' :
                     sim === "low" ? '<span style="color:var(--weak-color); font-weight:600;">Low (Extrapolation)</span>' : '–';
    h += `<td>${simBadge}</td></tr>`;
  });
  h += "</tbody></table></div>";
  if (res.rows.length > limit) {
    h += `<div style="margin-top:10px; font-size:0.82rem; color:var(--text-dim); text-align:center;">Showing the top ${limit} of ${res.rows.length.toLocaleString()} compounds. Download full CSV for all candidates.</div>`;
  }
  return h;
}

document.addEventListener("DOMContentLoaded", () => initTabs());
