/* Batch page — Enhanced Molecular Studio Logic */
(function () {
  const picker = new TargetPicker($("#target"));
  const form = $("#form"), err = $("#error"), out = $("#out"), go = $("#go");
  const prog = bindProgress();

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault(); 
    showError(err, ""); 
    out.innerHTML = "";
    
    const file = $("#file").files[0];
    const eps = $$('input[name="ep"]:checked').map((c) => c.value);
    if (!file) return showError(err, "Please choose a chemical library file to upload.");
    if (!picker.value() && !picker.uniprot()) return showError(err, "Please select or provide a target protein.");
    if (!eps.length) return showError(err, "Please select at least one endpoint (Ki or IC50).");

    const fd = new FormData();
    fd.append("file", file); 
    fd.append("endpoints", eps.join(","));
    if (picker.value()) fd.append("sequence", picker.value()); 
    else fd.append("uniprot", picker.uniprot());

    go.disabled = true; 
    prog.show();

    try {
      const start = await api("/api/batch", fd);
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
    const primaryEp = res.endpoints[0];
    const strongCount = res.rows.filter(r => (r[primaryEp + "_p"] || 0) >= 8.0).length;
    const modCount = res.rows.filter(r => (r[primaryEp + "_p"] || 0) >= 6.0 && (r[primaryEp + "_p"] || 0) < 8.0).length;

    let h = `
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px; margin-bottom:16px;">
        <div>
          <h3 class="card-title" style="font-size:1.4rem;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--cyan-primary);"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
            Batch Scoring Complete: ${res.n_scored.toLocaleString()} Compounds Processed
          </h3>
          <div style="font-size:0.85rem; color:var(--text-muted); margin-top:4px;">
            Ranked by primary endpoint: <b>${unitName(primaryEp)}</b> &bull; Models: ${Object.entries(res.models).map(([ep, m]) => `<b>${ep}</b>: ${esc(m.model)} (${esc(m.split)})`).join("; ")}
          </div>
        </div>
        <a class="btn-primary" href="/api/download/${jobId}.csv" style="width:auto; padding:10px 20px;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
          Download Results (CSV)
        </a>
      </div>

      <!-- Summary Metric Grid -->
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:20px;">
        <div class="guide-box">
          <div class="guide-label" style="color:var(--text-main);">Total Scored</div>
          <div style="font-size:1.6rem; font-weight:800; font-family:var(--font-display); color:#fff;">${res.n_scored.toLocaleString()}</div>
        </div>
        <div class="guide-box" style="border-color:var(--strong-border);">
          <div class="guide-label" style="color:var(--strong-color);">Strong Binders (&ge; 8.0)</div>
          <div style="font-size:1.6rem; font-weight:800; font-family:var(--font-display); color:var(--strong-color);">${strongCount.toLocaleString()}</div>
        </div>
        <div class="guide-box" style="border-color:var(--mod-border);">
          <div class="guide-label" style="color:var(--mod-color);">Moderate Binders (6.0 - 8.0)</div>
          <div style="font-size:1.6rem; font-weight:800; font-family:var(--font-display); color:var(--mod-color);">${modCount.toLocaleString()}</div>
        </div>
      </div>

      <!-- Biological Conclusion -->
      <div class="conclusion-box">
        <div class="conclusion-header">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
          Batch Assessment &amp; Discovery Conclusion
        </div>
        <p class="conclusion-body">
          ${strongCount > 0 
            ? `Found <b>${strongCount} potent compound(s)</b> predicted with nanomolar binding affinity (&le; 10 nM, pAffinity &ge; 8.0). These represent high-priority leads for confirmatory in vitro assays.`
            : modCount > 0 
              ? `No high-nanomolar binders found, but identified <b>${modCount} moderate hit(s)</b> (10 nM &ndash; 1 &micro;M). These are viable starting points for hit-to-lead chemical modification.`
              : `All evaluated compounds are predicted to have weak or negligible binding affinity (&gt; 1 &micro;M). Consider exploring alternate chemical series or expanding library structural diversity.`}
        </p>
      </div>`;

    h += resultsTable(res) + "</div>";

    if (res.invalid && res.invalid.length) {
      h += `
      <div class="card" style="border-color:var(--warn-border);">
        <h3 class="card-title" style="color:var(--warn-text);">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
          ${res.invalid.length} Molecule(s) Skipped Due to Parsing Errors
        </h3>
        <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:12px;">The following rows could not be parsed into valid chemical graphs by RDKit:</p>
        <div class="table-container">
          <table class="bio-table">
            <thead>
              <tr><th style="width:70px;">Row #</th><th>Invalid SMILES String</th><th>Parsing Error Reason</th></tr>
            </thead>
            <tbody>
              ${res.invalid.slice(0, 50).map((r) => `
              <tr>
                <td><b>${r.line}</b></td>
                <td><code style="color:var(--err-text);">${esc(r.smiles)}</code></td>
                <td style="color:var(--text-muted); font-size:0.8rem;">${esc(r.reason)}</td>
              </tr>`).join("")}
            </tbody>
          </table>
        </div>
      </div>`;
    }

    out.innerHTML = h;
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  }
})();
