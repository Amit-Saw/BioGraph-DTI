/* Predict page — Enhanced Molecular Studio Logic */
(function () {
  const smilesEl = $("#smiles"), msgEl = $("#smiles-msg"), previewEl = $("#preview");
  const goBtn = $("#go"), errEl = $("#error"), resultsEl = $("#results"), bannerEl = $("#banner");
  const detailEl = $("#detail");
  const picker = new TargetPicker($("#target"));
  let current = null;          // {smiles, sequence}
  let viewer = null, conformerFor = null;

  /* ---------- drug input with live preview ---------- */
  let timer = null;
  smilesEl.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(validate, 400); });
  async function validate() {
    const s = smilesEl.value.trim();
    if (!s) {
      msgEl.textContent = ""; 
      previewEl.innerHTML = '<span style="color:var(--text-dim); font-size:0.85rem;">Enter a SMILES string to render 2D graph structure...</span>'; 
      return false; 
    }
    try {
      const r = await api("/api/validate", { smiles: s });
      if (r.ok) { 
        msgEl.innerHTML = '<span style="color:var(--strong-color); font-weight:600;">&check; Valid chemical structure parsed</span>'; 
        previewEl.innerHTML = r.svg; 
        return true; 
      }
      msgEl.innerHTML = `<span style="color:var(--err-text); font-weight:500;">&times; ${esc(r.message)}</span>`; 
      previewEl.innerHTML = '<span style="color:var(--err-text); font-size:0.85rem;">Invalid chemical SMILES</span>'; 
      return false;
    } catch (e) { 
      msgEl.innerHTML = `<span style="color:var(--err-text);">${esc(e.message)}</span>`; 
      return false; 
    }
  }

  getExamples().then((ex) => {
    const box = $("#drug-examples");
    box.innerHTML = "";
    (ex.drugs || []).forEach((d) => {
      const b = document.createElement("button"); 
      b.type = "button"; 
      b.className = "chip-btn"; 
      b.textContent = d.name;
      b.addEventListener("click", () => { smilesEl.value = d.smiles; validate(); });
      box.appendChild(b);
    });
  });

  const params = new URLSearchParams(location.search);
  if (params.get("smiles")) { smilesEl.value = params.get("smiles"); validate(); }

  /* ---------- predict ---------- */
  goBtn.addEventListener("click", async () => {
    showError(errEl, "");
    const smiles = smilesEl.value.trim(), sequence = picker.value();
    if (!smiles) return showError(errEl, "Please enter a valid ligand SMILES string.");
    if (!sequence) return showError(errEl, "Please specify a target protein (fetch via UniProt ID or paste a sequence).");
    
    goBtn.disabled = true; 
    goBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="spinner" style="display:inline-block; vertical-align:-3px; margin-right:8px;"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 14 14"></polyline></svg> Evaluating GNN + ESM-2 Models…';
    resultsEl.hidden = true; 
    bannerEl.hidden = true;

    try {
      const r = await api("/api/predict", { smiles, sequence });
      current = { smiles: r.smiles, sequence };
      resultsEl.innerHTML = Object.entries(r.results).map(([ep, v]) => card(ep, v)).join("");
      resultsEl.hidden = false; 
      detailEl.hidden = false;

      loadProperties(); 
      loadSimilar();
      conformerFor = null; 
      $("#explain-out").innerHTML = "";
      if (!$("#t-3d").hidden) show3D();

      // Smooth scroll to results
      resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { 
      showError(errEl, e.message); 
    } finally { 
      goBtn.disabled = false; 
      goBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Predict Ki and IC50 Binding Affinity'; 
    }
  });

  function pos(p) { return Math.max(0, Math.min(100, ((p - 3) / 8) * 100)); }   // scale 3..11

  function getConclusion(ep, pAffinity, concStr) {
    if (pAffinity >= 8.0) {
      return {
        badgeClass: "strong",
        title: "High-Potency Nanomolar Binder",
        text: `The predicted ${ep === "Ki" ? "dissociation constant (pKi)" : "inhibition potency (pIC50)"} of ${pAffinity.toFixed(2)} indicates very strong nanomolar binding (≈ ${concStr}). This compound satisfies the typical binding threshold for a drug discovery lead candidate (≤ 10 nM).`
      };
    } else if (pAffinity >= 6.0) {
      return {
        badgeClass: "moderate",
        title: "Moderate Sub-Micromolar Binder",
        text: `The predicted ${ep === "Ki" ? "dissociation constant (pKi)" : "inhibition potency (pIC50)"} of ${pAffinity.toFixed(2)} represents moderate binding activity (≈ ${concStr}). Typical for early-stage screening hits (10 nM – 1 µM); structure-activity relationship (SAR) optimization is recommended to achieve nanomolar potency.`
      };
    } else {
      return {
        badgeClass: "weak",
        title: "Weak or Negligible Binding",
        text: `The predicted ${ep === "Ki" ? "dissociation constant (pKi)" : "inhibition potency (pIC50)"} of ${pAffinity.toFixed(2)} indicates weak affinity (≈ ${concStr}). The concentration required to bind or inhibit exceeds micromolar levels (> 1,000 nM), suggesting poor specific efficacy against this receptor.`
      };
    }
  }

  function card(ep, v) {
    if (v.error) {
      return `<div class="result-card"><h3 class="result-metric-title">${ep}</h3><p class="alert-box error">${esc(v.error)}</p></div>`;
    }
    const p = v.primary, u = unitName(ep);
    const lo = pos(p.pAffinity - (p.sigma || 0)), hi = pos(p.pAffinity + (p.sigma || 0));
    const concStr = esc(p.conc || fmtConc(p.nM));
    const conclusion = getConclusion(ep, p.pAffinity, concStr);

    let h = `
    <div class="result-card">
      <div class="result-card-header">
        <div>
          <h3 class="result-metric-title">
            ${u} 
            <span style="font-size:0.85rem; font-weight:500; color:var(--text-muted);">(${ep === "Ki" ? "Inhibition Constant" : "Half-Maximal Inhibition"})</span>
          </h3>
          <div class="result-metric-desc">
            ${ep === "Ki" ? "Thermodynamic equilibrium binding dissociation" : "Enzymatic / cellular functional inhibition concentration"}
          </div>
        </div>
        <span class="potency-badge ${conclusion.badgeClass}">
          <span style="width:7px; height:7px; border-radius:50%; background:currentColor; display:inline-block;"></span>
          ${esc(p.label)}
        </span>
      </div>

      <div class="result-value-row">
        <div class="result-value-big">${p.pAffinity.toFixed(2)}</div>
        <div class="result-conc-pill">&asymp; ${concStr}</div>
      </div>

      <div style="font-size:0.82rem; color:var(--text-muted); margin-bottom:12px;">
        Statistical 68% Confidence Interval (&plusmn;1&sigma;): <b>${esc(p.band_text || "n/a")}</b>
      </div>

      <!-- Potency Scale Gauge -->
      <div class="potency-meter">
        <div class="meter-band" style="left:${lo}%; width:${Math.max(hi - lo, 2)}%;"></div>
        <div class="meter-needle" style="left:${pos(p.pAffinity)}%;"></div>
      </div>
      <div class="meter-labels">
        <span>Weak (pAff 3 &bull; 1 mM)</span>
        <span>Moderate (pAff 6 &bull; 1 &micro;M)</span>
        <span>Strong (pAff 8 &bull; 10 nM)</span>
        <span>Ultra-Potent (pAff 11)</span>
      </div>

      <!-- Biological Conclusion Box -->
      <div class="conclusion-box">
        <div class="conclusion-header">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
          Conclusion &amp; Biological Interpretation
        </div>
        <p class="conclusion-body"><b>${conclusion.title}:</b> ${conclusion.text}</p>
      </div>

      <!-- Model Specifications -->
      <div class="reference-box">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span><b>Primary Model:</b> ${esc(p.model)} (Split: <code>${esc(p.split)}</code>)</span>
          <span style="font-family:var(--font-mono); font-size:0.75rem;">Test RMSE: ${p.test_rmse ? p.test_rmse.toFixed(2) : "n/a"}</span>
        </div>
        <div style="font-size:0.78rem; color:var(--text-dim); line-height:1.4;">
          Trained on novel Bemis-Murcko scaffolds. The uncertainty band reflects empirical validation error on unseen chemical series.
        </div>`;

    if (v.reference) {
      const r = v.reference;
      h += `
        <div style="margin-top:8px; padding-top:8px; border-top:1px solid var(--border-card); font-size:0.8rem;">
          <b>Reference Model (${esc(r.model)}, <code>${esc(r.split)}</code> split):</b> ${u} ${r.pAffinity.toFixed(2)} &asymp; ${esc(r.conc)}.
          <span style="color:var(--text-dim);">(Trained with random split; has seen similar chemical scaffolds in training).</span>
        </div>`;
      if (v.models_disagree) {
        h += `
        <div style="margin-top:8px; padding:6px 10px; background:var(--warn-bg); border:1px solid var(--warn-border); border-radius:var(--radius-sm); color:var(--warn-text); font-size:0.8rem;">
          &Delta; Model Disagreement: The cold-drug and random-split models differ by ${v.agreement_abs.toFixed(2)} log units. Treat this prediction with higher caution.
        </div>`;
      }
    }

    h += `</div></div>`;
    return h;
  }

  /* ---------- properties / similar ---------- */
  async function loadProperties() {
    const box = $("#t-props"); 
    box.innerHTML = '<div style="padding:24px; text-align:center; color:var(--text-muted);"><span class="spinner">Computing physicochemical descriptors…</span></div>';
    try {
      const r = await api("/api/properties", { smiles: current.smiles });
      const d = r.descriptors, l = r.lipinski;
      box.innerHTML = `
      <div class="two-col-layout">
        <div>
          <div style="font-weight:600; font-size:0.88rem; color:var(--text-muted); margin-bottom:8px;">2D Topological Structure Representation:</div>
          <div class="mol-frame" style="min-height:280px;">${r.svg}</div>
          <div style="margin-top:10px; font-size:0.78rem; color:var(--text-dim); word-break:break-all;">
            <b>Canonical SMILES:</b> <code>${esc(r.smiles)}</code>
          </div>
        </div>
        <div>
          <div style="font-weight:600; font-size:0.88rem; color:var(--text-muted); margin-bottom:8px;">Physicochemical Properties &amp; Drug-Likeness:</div>
          <div class="table-container">
            <table class="bio-table">
              <tr><th style="width:40%;">Molecular Formula</th><td><b>${esc(d.formula)}</b></td></tr>
              <tr><th>Molecular Weight</th><td>${d.mol_weight} g/mol <span style="font-size:0.75rem; color:var(--text-dim);">(Lipinski &le; 500)</span></td></tr>
              <tr><th>Calculated LogP (cLogP)</th><td>${d.logp} <span style="font-size:0.75rem; color:var(--text-dim);">(Lipinski &le; 5)</span></td></tr>
              <tr><th>H-Bond Donors / Acceptors</th><td>${d.h_donors} / ${d.h_acceptors} <span style="font-size:0.75rem; color:var(--text-dim);">(HBD &le; 5, HBA &le; 10)</span></td></tr>
              <tr><th>Topological Polar Surface Area</th><td>${d.tpsa} &Aring;&sup2; <span style="font-size:0.75rem; color:var(--text-dim);">(Polarity &le; 140 &Aring;&sup2;)</span></td></tr>
              <tr><th>Rotatable Bonds</th><td>${d.rotatable_bonds} <span style="font-size:0.75rem; color:var(--text-dim);">(Flexibility &le; 10)</span></td></tr>
              <tr><th>Aromatic Rings / Heavy Atoms</th><td>${d.rings} rings / ${d.heavy_atoms} heavy atoms</td></tr>
              <tr>
                <th>Lipinski's Rule of 5 Status</th>
                <td>${l.passes 
                  ? '<span class="potency-badge strong" style="font-size:0.75rem; padding:2px 8px;">Passes All Criteria</span>' 
                  : '<span class="potency-badge weak" style="font-size:0.75rem; padding:2px 8px;">Violations: ' + esc(l.violations.join(", ")) + '</span>'}
                </td>
              </tr>
            </table>
          </div>
        </div>
      </div>`;
    } catch (e) { 
      box.innerHTML = `<div class="alert-box error">${esc(e.message)}</div>`; 
    }
  }

  async function loadSimilar() {
    const box = $("#t-similar"); 
    box.innerHTML = '<div style="padding:24px; text-align:center; color:var(--text-muted);"><span class="spinner">Searching training set via Morgan Fingerprints (radius=2, 2048-bit)…</span></div>';
    try {
      const r = await api("/api/similar", { smiles: current.smiles, sequence: current.sequence });
      let h = "", worst = null;
      Object.entries(r.results).forEach(([ep, v]) => {
        if (!v.available) { 
          h += `<div style="margin-bottom:16px;"><h4>${ep} Reference Compounds</h4><p style="color:var(--text-muted); font-size:0.85rem;">Reference dataset file not indexed.</p></div>`; 
          return; 
        }
        if (v.level === "low" && !worst) worst = v;
        const levelBadge = v.level === "high" 
          ? '<span class="potency-badge strong" style="font-size:0.75rem;">High Familiarity</span>' 
          : v.level === "medium" 
            ? '<span class="potency-badge moderate" style="font-size:0.75rem;">Moderate Analogy</span>' 
            : '<span class="potency-badge weak" style="font-size:0.75rem;">Extrapolation Risk (Low)</span>';

        h += `
        <div style="margin-bottom:24px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <h4 style="margin:0; font-size:1.05rem; color:#fff;">
              ${ep}: Closest Training Compounds (Max Tanimoto: <b>${(v.max_tanimoto * 100).toFixed(1)}%</b>)
            </h4>
            ${levelBadge}
          </div>
          <div class="table-container">
            <table class="bio-table">
              <thead>
                <tr>
                  <th style="width:15%;">Tanimoto Similarity</th>
                  <th>Training Molecule SMILES</th>
                  <th>Observed on This Target</th>
                  <th>Median Across All Targets</th>
                </tr>
              </thead>
              <tbody>`;
        v.neighbours.forEach((n) => {
          h += `
                <tr>
                  <td><b>${(n.similarity * 100).toFixed(1)}%</b></td>
                  <td><code style="font-size:0.8rem;">${esc(n.smiles.length > 55 ? n.smiles.slice(0, 55) + "…" : n.smiles)}</code></td>
                  <td>${n.same_target_pAffinity !== null ? `<span style="color:var(--cyan-primary); font-weight:600;">${unitName(ep)} ${n.same_target_pAffinity.toFixed(2)}</span>` : '<span style="color:var(--text-dim);">&mdash;</span>'}</td>
                  <td>${n.median_pAffinity !== null && n.median_pAffinity !== undefined ? `pAff ${Number(n.median_pAffinity).toFixed(2)} (${n.n_targets} target${n.n_targets === 1 ? "" : "s"})` : '<span style="color:var(--text-dim);">&mdash;</span>'}</td>
                </tr>`;
        });
        h += `</tbody></table></div></div>`;
      });
      box.innerHTML = h || '<p style="color:var(--text-muted);">No reference data available.</p>';
      
      if (worst) {
        bannerEl.innerHTML = `
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0; margin-top:2px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
          <div>
            <b>Applicability Domain Alert:</b> This molecule has low structural similarity to compounds seen during training (&lt; 30% Tanimoto). 
            The neural network is extrapolating into novel chemical space; treat predicted affinities with wider uncertainty.
          </div>`;
        bannerEl.hidden = false;
      }
    } catch (e) { 
      box.innerHTML = `<div class="alert-box error">${esc(e.message)}</div>`; 
    }
  }

  /* ---------- 3D conformer ---------- */
  async function show3D() {
    if (!current || conformerFor === current.smiles) return;
    const el = $("#viewer3d");
    try {
      const r = await api("/api/conformer", { smiles: current.smiles });
      el.innerHTML = "";
      viewer = $3Dmol.createViewer(el, { backgroundColor: "#ffffff" });
      viewer.addModel(r.molblock, "sdf");
      viewer.setStyle({}, { stick: { radius: 0.16, colorscheme: "element" }, sphere: { scale: 0.24 } });
      viewer.zoomTo(); 
      viewer.render(); 
      conformerFor = current.smiles;
    } catch (e) { 
      el.innerHTML = `<div class="alert-box error" style="margin:20px;">${esc(e.message)}</div>`; 
    }
  }
  document.addEventListener("tabshown", (ev) => { if (ev.detail === "t-3d") show3D(); });

  /* ---------- explanation ---------- */
  $("#explain-go").addEventListener("click", async () => {
    if (!current) return;
    const out = $("#explain-out"), ep = $("#explain-ep").value;
    out.innerHTML = '<div style="padding:24px; text-align:center; color:var(--text-muted);"><span class="spinner">Iterating through all atoms with graph occlusion…</span></div>';
    try {
      const r = await api("/api/explain", { smiles: current.smiles, sequence: current.sequence, endpoint: ep });
      out.innerHTML = `
      <div class="two-col-layout">
        <div>
          <div style="font-weight:600; font-size:0.88rem; color:var(--text-muted); margin-bottom:8px;">Atom Sensitivity Colormap:</div>
          <div class="mol-frame" style="min-height:300px;">${r.svg}</div>
          <div style="display:flex; justify-content:space-around; margin-top:12px; font-size:0.82rem; background:rgba(15,23,42,0.8); padding:8px 14px; border-radius:var(--radius-pill); border:1px solid var(--border-card);">
            <span style="display:flex; align-items:center; gap:6px;"><i style="display:inline-block; width:12px; height:12px; border-radius:3px; background:rgb(26,102,217);"></i> Increases predicted ${unitName(ep)}</span>
            <span style="display:flex; align-items:center; gap:6px;"><i style="display:inline-block; width:12px; height:12px; border-radius:3px; background:rgb(235,128,20);"></i> Decreases predicted ${unitName(ep)}</span>
          </div>
        </div>
        <div>
          <div style="font-weight:600; font-size:0.88rem; color:var(--text-muted); margin-bottom:8px;">Most Influential Atoms Ranked by Impact:</div>
          <div class="table-container">
            <table class="bio-table">
              <thead>
                <tr>
                  <th>Atom Index</th>
                  <th>Element Symbol</th>
                  <th>&Delta; Affinity if Removed</th>
                  <th>Influence Direction</th>
                </tr>
              </thead>
              <tbody>
                ${r.top_atoms.map((a) => `
                <tr>
                  <td><b>#${a.index}</b></td>
                  <td><code>${esc(a.symbol)}</code></td>
                  <td style="font-family:var(--font-mono); font-weight:600; color:${a.score > 0 ? 'var(--cyan-primary)' : 'var(--mod-color)'};">
                    ${a.score > 0 ? "+" : ""}${a.score}
                  </td>
                  <td>
                    ${a.score > 0 
                      ? '<span style="color:var(--cyan-primary); font-size:0.8rem;">Raises binding potency</span>' 
                      : '<span style="color:var(--mod-color); font-size:0.8rem;">Lowers binding potency</span>'}
                  </td>
                </tr>`).join("")}
              </tbody>
            </table>
          </div>
          <div style="margin-top:12px; font-size:0.8rem; color:var(--text-dim); line-height:1.4;">
            <b>Scientific Note:</b> Occlusion measures what the graph neural network's message-passing layers are sensitive to. Removing an atom can generate an unphysical chemical fragment; results should be evaluated as model sensitivity rather than verified covalent binding energy.
          </div>
        </div>
      </div>`;
    } catch (e) { 
      out.innerHTML = `<div class="alert-box error">${esc(e.message)}</div>`; 
    }
  });
})();
