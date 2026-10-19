/* Virtual screening page — Enhanced Molecular Studio Logic */
(function () {
  const picker = new TargetPicker($("#target"));
  const form = $("#form"), err = $("#error"), out = $("#out"), go = $("#go");
  const prog = bindProgress();

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault(); 
    showError(err, ""); 
    out.innerHTML = "";

    const file = $("#file").files[0];
    if (!file) return showError(err, "Please choose a chemical library file to screen.");
    if (!picker.value() && !picker.uniprot()) return showError(err, "Please select or provide a target protein.");

    const fd = new FormData();
    fd.append("file", file); 
    fd.append("endpoint", $("#endpoint").value); 
    fd.append("top_n", $("#top_n").value);
    if (picker.value()) fd.append("sequence", picker.value()); 
    else fd.append("uniprot", picker.uniprot());

    go.disabled = true; 
    prog.show();

    try {
      const start = await api("/api/screen", fd);
      const res = await pollJob(start.job_id, (st) => prog.update(st));
      render(res, start.job_id);
    } catch (e) { 
      showError(err, e.message); 
    } finally { 
      go.disabled = false; 
      prog.hide(); 
    }
  });

  function render(res, jobId) {
    const ep = res.endpoints[0], m = res.models[ep];
    const topHit = res.rows[0];
    const strongCount = res.rows.filter(r => (r[ep + "_p"] || 0) >= 8.0).length;

    let h = `
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px; margin-bottom:16px;">
        <div>
          <h3 class="card-title" style="font-size:1.4rem;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--strong-color);"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon></svg>
            Screening Leaderboard: Top ${res.rows.length} of ${res.n_scored.toLocaleString()} Compounds
          </h3>
          <div style="font-size:0.85rem; color:var(--text-muted); margin-top:4px;">
            Primary Ranking Metric: <b>${unitName(ep)}</b> &bull; Model: <b>${esc(m.model)}</b> (Scaffold Split: <code>${esc(m.split)}</code>)
          </div>
        </div>
        <a class="btn-primary" href="/api/download/${jobId}.csv" style="width:auto; padding:10px 20px;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
          Export Ranked Hits (CSV)
        </a>
      </div>

      <!-- Discovery Conclusion Box -->
      <div class="conclusion-box">
        <div class="conclusion-header">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
          Virtual Screening Conclusion &amp; Hit Prioritization
        </div>
        <p class="conclusion-body">
          ${topHit 
            ? `Top candidate <b>"${esc(topHit.name || 'Rank #1')}"</b> achieved a predicted ${unitName(ep)} of <b>${topHit[ep + '_p'].toFixed(2)}</b> (&asymp; ${esc(topHit[ep + '_conc'])}, <b>${esc(topHit[ep + '_label'])}</b>). 
               Across the library, <b>${strongCount} compound(s)</b> reached nanomolar binding threshold (&le; 10 nM). We recommend taking the top 10–20 unique scaffolds forward into in vitro binding assays.`
            : `Screening completed with no active hits detected.`}
        </p>
      </div>`;

    h += resultsTable(res, { thumbs: true, limit: 100 }) + "</div>";
    
    if (res.invalid && res.invalid.length) {
      h += `<div style="margin-top:10px; font-size:0.82rem; color:var(--text-dim); text-align:center;">Note: ${res.invalid.length} compound(s) with invalid SMILES structures were omitted from the ranked results.</div>`;
    }

    out.innerHTML = h;
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  }
})();
