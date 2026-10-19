/* Compare page — Enhanced Molecular Studio Logic */
(function () {
  const picker = new TargetPicker($("#target-a"));

  // Quick Demo Loaders
  const loadAnalogsBtn = $("#load-analogs-demo");
  if (loadAnalogsBtn) {
    loadAnalogsBtn.addEventListener("click", () => {
      $("#drugs").value = "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC\nCOc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1";
      const sel = $(".tp-uniprot", $("#target-a"));
      if (sel) sel.value = "P00533";
      const fetchBtn = $(".tp-fetch", $("#target-a"));
      if (fetchBtn) fetchBtn.click();
    });
  }

  const loadSelectBtn = $("#load-select-demo");
  if (loadSelectBtn) {
    loadSelectBtn.addEventListener("click", () => {
      $("#one-drug").value = "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC";
      $("#targets").value = "P00533, EGFR\nP00519, ABL1\nP04626, ERBB2\nP15056, BRAF";
    });
  }

  /* ---------- compare drugs ---------- */
  $("#go-drugs").addEventListener("click", async () => {
    const err = $("#err-drugs"), out = $("#out-drugs"), btn = $("#go-drugs");
    showError(err, ""); 
    out.innerHTML = "";
    
    const smiles = $("#drugs").value.split("\n").map((s) => s.trim()).filter(Boolean);
    if (smiles.length < 2) return showError(err, "Please enter at least two valid SMILES strings to compare.");
    if (!picker.value()) return showError(err, "Please choose or provide a target protein.");

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner">Evaluating Analogs…</span>';

    try {
      const r = await api("/api/compare", { smiles_list: smiles, sequence: picker.value() });
      
      // Determine top analog
      let bestIdx = 0, bestVal = -1;
      r.items.forEach((it, i) => {
        const kiVal = (it.results && it.results.Ki) ? it.results.Ki.pAffinity : 0;
        if (kiVal > bestVal) { bestVal = kiVal; bestIdx = i; }
      });

      let h = `
      <div class="card">
        <h3 class="card-title" style="font-size:1.3rem; margin-bottom:12px;">
          Analog Affinity Comparison (${r.items.length} Compounds)
        </h3>
        
        <!-- Biological Conclusion Box -->
        <div class="conclusion-box">
          <div class="conclusion-header">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
            Analog Ranking &amp; SAR Conclusion
          </div>
          <p class="conclusion-body">
            <b>Drug ${bestIdx + 1}</b> demonstrates the highest predicted binding affinity with pKi <b>${bestVal.toFixed(2)}</b> (&asymp; ${esc(r.items[bestIdx].results.Ki.conc)}). 
            Comparative differences highlight critical functional group contributions to pocket stabilization.
          </p>
        </div>

        <div class="grid-cards" style="margin-top:16px;">
          ${r.items.map((it, i) => `
            <div class="result-card" style="padding:18px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h4 style="margin:0; font-size:1.1rem; color:#fff;">Drug Candidate ${i + 1} ${i === bestIdx ? '👑 <span style="font-size:0.75rem; color:var(--cyan-primary);">Top Binder</span>' : ''}</h4>
              </div>
              <div class="mol-frame" style="min-height:160px; margin-bottom:12px;">${it.svg}</div>
              <div style="display:flex; flex-direction:column; gap:8px;">
                ${Object.entries(it.results).map(([ep, d]) => `
                  <div style="background:rgba(11,17,32,0.6); padding:8px 12px; border-radius:var(--radius-md); border:1px solid var(--border-card); display:flex; justify-content:space-between; align-items:center;">
                    <div>
                      <span style="font-weight:700; color:#fff; font-size:1rem;">${unitName(ep)} ${d.pAffinity.toFixed(2)}</span>
                      <div style="font-size:0.78rem; color:var(--text-muted);">&asymp; ${esc(d.conc)}</div>
                    </div>
                    <div>${badge(d.label)}</div>
                  </div>`).join("")}
              </div>
            </div>`).join("")}
        </div>
      </div>`;

      h += chart(r.items.map((it, i) => ({ name: `Drug ${i + 1}`, results: it.results })));
      out.innerHTML = h;
      out.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { 
      showError(err, e.message); 
    } finally { 
      btn.disabled = false; 
      btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Compare Analogs Against Target'; 
    }
  });

  /* ---------- selectivity ---------- */
  $("#go-select").addEventListener("click", async () => {
    const err = $("#err-select"), out = $("#out-select"), btn = $("#go-select");
    showError(err, ""); 
    out.innerHTML = "";

    const smiles = $("#one-drug").value.trim();
    const targets = $("#targets").value.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
      const [acc, ...name] = l.split(","); 
      return { uniprot: acc.trim(), name: name.join(",").trim() || undefined };
    });

    if (!smiles) return showError(err, "Please enter a valid query SMILES string.");
    if (!targets.length) return showError(err, "Please provide at least one target UniProt ID.");

    btn.disabled = true; 
    btn.innerHTML = '<span class="spinner">Profiling Across Receptor Panel…</span>';

    try {
      const r = await api("/api/selectivity", { smiles, targets });
      const good = r.rows.filter((x) => !x.error);
      
      // Compute selectivity ratio
      let maxAff = -1, minAff = 99, primaryTarget = "", secondaryTarget = "";
      good.forEach(g => {
        const val = g.results.Ki ? g.results.Ki.pAffinity : (g.results.IC50 ? g.results.IC50.pAffinity : 0);
        if (val > maxAff) { maxAff = val; primaryTarget = g.name; }
        if (val < minAff) { minAff = val; secondaryTarget = g.name; }
      });
      const foldDiff = Math.pow(10, Math.abs(maxAff - minAff));

      let h = `
      <div class="card">
        <h3 class="card-title" style="font-size:1.3rem; margin-bottom:12px;">
          Multi-Target Selectivity Profile
        </h3>

        <!-- Biological Conclusion Box -->
        <div class="conclusion-box">
          <div class="conclusion-header">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
            Selectivity Assessment &amp; Off-Target Margin
          </div>
          <p class="conclusion-body">
            Target <b>${esc(primaryTarget)}</b> shows the highest binding potency (pAffinity <b>${maxAff.toFixed(2)}</b>). 
            Selectivity window vs. lowest-affinity target (${esc(secondaryTarget)}): <b>&approx; ${foldDiff.toFixed(1)}&times; margin</b>.
            ${foldDiff > 100 
              ? 'Excellent target specificity (> 100-fold window); low anticipated off-target binding across this panel.' 
              : foldDiff > 10 
                ? 'Moderate selectivity window (10–100 fold); monitor cross-reactivity in downstream assays.' 
                : 'Broad promiscuous binding profile (< 10-fold window); potential pan-target inhibition risk.'}
          </p>
        </div>

        <div class="two-col-layout" style="margin-bottom:20px;">
          <div>
            <div style="font-weight:600; font-size:0.88rem; color:var(--text-muted); margin-bottom:8px;">Compound Under Evaluation:</div>
            <div class="mol-frame" style="min-height:220px;">${r.svg}</div>
          </div>
          <div>
            <div style="font-weight:600; font-size:0.88rem; color:var(--text-muted); margin-bottom:8px;">Comparative Panel Affinity:</div>
            ${chart(good)}
          </div>
        </div>

        <div class="table-container">
          <table class="bio-table">
            <thead>
              <tr>
                <th>Target Receptor</th>
                <th style="text-align:right;">pKi (Affinity)</th>
                <th style="text-align:right;">pIC50 (Potency)</th>
                <th style="text-align:center;">Potency Tier</th>
              </tr>
            </thead>
            <tbody>
              ${r.rows.map((x) => x.error ? `
                <tr>
                  <td><b>${esc(x.name)}</b></td>
                  <td colspan="3" style="color:var(--err-text); font-size:0.82rem;">${esc(x.error)}</td>
                </tr>` : `
                <tr>
                  <td><b>${esc(x.name)}</b></td>
                  <td style="text-align:right; font-family:var(--font-mono); font-weight:700; color:var(--cyan-primary);">
                    ${x.results.Ki ? `pKi ${x.results.Ki.pAffinity.toFixed(2)} <span style="font-size:0.78rem; color:var(--text-muted);">(&asymp; ${esc(x.results.Ki.conc)})</span>` : '&mdash;'}
                  </td>
                  <td style="text-align:right; font-family:var(--font-mono); font-weight:700; color:var(--cyan-primary);">
                    ${x.results.IC50 ? `pIC50 ${x.results.IC50.pAffinity.toFixed(2)} <span style="font-size:0.78rem; color:var(--text-muted);">(&asymp; ${esc(x.results.IC50.conc)})</span>` : '&mdash;'}
                  </td>
                  <td style="text-align:center;">
                    ${x.results.Ki ? badge(x.results.Ki.label) : (x.results.IC50 ? badge(x.results.IC50.label) : '&mdash;')}
                  </td>
                </tr>`).join("")}
            </tbody>
          </table>
        </div>
      </div>`;

      out.innerHTML = h;
      out.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { 
      showError(err, e.message); 
    } finally { 
      btn.disabled = false; 
      btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Evaluate Selectivity Across Panel'; 
    }
  });

  /* Horizontal bars of pAffinity (scale 3..11) */
  function chart(rows) {
    if (!rows.length) return "";
    const w = (p) => Math.max(2, Math.min(100, ((p - 3) / 8) * 100));
    let h = `
    <div style="background:rgba(11,17,32,0.7); border:1px solid var(--border-card); border-radius:var(--radius-md); padding:16px;">
      <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:var(--cyan-primary); margin-bottom:12px;">
        Affinity Spectrum (pAffinity Scale: 3 to 11)
      </div>`;
    
    rows.forEach((r) => {
      h += `<div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; font-size:0.86rem; margin-bottom:3px;">
          <b>${esc(r.name)}</b>
        </div>`;
      Object.entries(r.results).forEach(([ep, d]) => {
        h += `
        <div style="display:flex; align-items:center; gap:8px; margin:2px 0;">
          <span style="width:42px; font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono);">${ep}</span>
          <div style="flex:1; background:rgba(255,255,255,0.06); height:12px; border-radius:var(--radius-pill); overflow:hidden;">
            <div style="width:${w(d.pAffinity)}%; height:100%; background:linear-gradient(90deg, var(--cyan-primary), var(--indigo-primary)); border-radius:var(--radius-pill); box-shadow:0 0 8px var(--cyan-glow);"></div>
          </div>
          <span style="width:36px; text-align:right; font-family:var(--font-mono); font-weight:700; font-size:0.82rem; color:#fff;">${d.pAffinity.toFixed(2)}</span>
        </div>`;
      });
      h += `</div>`;
    });
    return h + `</div>`;
  }
})();
