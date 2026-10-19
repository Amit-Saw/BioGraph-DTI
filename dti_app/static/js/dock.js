/* Dock page — Enhanced Molecular Studio Logic */
(function () {
  const err = $("#error"), prog = bindProgress();
  let ligands = [];

  getExamples().then((ex) => {
    if (ex.docking && ex.docking.available === false) {
      const u = $("#unavailable"); 
      u.hidden = false;
      u.innerHTML = `
        <div style="display:flex; align-items:flex-start; gap:10px;">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
          <div>
            <b>Structural Docking Not Active:</b> ${esc(ex.docking.message)}. 
            AutoDock Vina, Meeko, or OpenBabel are optional components. (Machine learning affinity predictions on /predict remain 100% operational).
          </div>
        </div>`;
    }
    const pdbExamplesBox = $("#pdb-examples");
    if (pdbExamplesBox) {
      pdbExamplesBox.innerHTML = "";
      [["1M17", "EGFR + Erlotinib"], ["1HSG", "HIV-1 Protease + Indinavir"]].forEach(([id, label]) => {
        const b = document.createElement("button"); 
        b.type = "button"; 
        b.className = "chip-btn"; 
        b.textContent = `${id} (${label})`;
        b.addEventListener("click", () => { $("#pdb").value = id; findLigands(); });
        pdbExamplesBox.appendChild(b);
      });
    }
  });

  const q = new URLSearchParams(location.search);
  if (q.get("smiles")) $("#smiles").value = q.get("smiles");

  $("#find").addEventListener("click", findLigands);
  async function findLigands() {
    showError(err, ""); 
    $("#lig-box").hidden = true;
    const id = $("#pdb").value.trim();
    if (!id) return showError(err, "Please enter a valid 4-character PDB ID (e.g. 1M17).");
    
    const findBtn = $("#find");
    findBtn.disabled = true;
    findBtn.innerHTML = '<span class="spinner">Querying RCSB PDB…</span>';

    try {
      const r = await api(`/api/pdb/${encodeURIComponent(id)}/ligands`);
      ligands = r.ligands;
      if (!ligands.length) return showError(err, "This crystallographic entry contains no drug-sized co-crystallized ligand to define pocket coordinates.");
      
      $("#lig").innerHTML = ligands.map((l, i) => `<option value="${i}">${esc(l.code)} (Chain ${esc(l.chain)}, Residue ${esc(l.resi)}, ${l.atoms} heavy atoms)</option>`).join("");
      $("#chains").innerHTML = r.chains.map((c) => `
        <label style="display:inline-flex; align-items:center; gap:6px; cursor:pointer; font-size:0.85rem; background:rgba(11,17,32,0.6); padding:4px 10px; border-radius:var(--radius-pill); border:1px solid var(--border-card);">
          <input type="checkbox" name="chain" value="${esc(c.chain)}" checked style="width:auto;"> 
          <span>Chain ${esc(c.chain)} (${c.residues} residues)</span>
        </label>`).join("");
      $("#lig-box").hidden = false;
    } catch (e) { 
      showError(err, e.message); 
    } finally {
      findBtn.disabled = false;
      findBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" style="margin-right:4px;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg> Inspect PDB Cavity';
    }
  }

  $("#go").addEventListener("click", async () => {
    showError(err, ""); 
    $("#out").hidden = true;
    const lig = ligands[Number($("#lig").value)];
    const chains = $$('input[name="chain"]:checked').map((c) => c.value);
    if (!lig) return showError(err, "Please choose a pocket ligand to define coordinates.");
    if (!chains.length) return showError(err, "Please keep at least one protein chain for receptor preparation.");

    const btn = $("#go"); 
    btn.disabled = true; 
    prog.show();

    try {
      const start = await api("/api/dock", {
        pdb_id: $("#pdb").value.trim(), 
        chains, 
        ligand: { code: lig.code, chain: lig.chain, resi: lig.resi },
        smiles: $("#smiles").value.trim(), 
        box: Number($("#box").value), 
        exhaustiveness: Number($("#exh").value),
      });
      const res = await pollJob(start.job_id, (st) => prog.update(st), 1500);
      show(res);
    } catch (e) { 
      showError(err, e.message); 
    } finally { 
      btn.disabled = false; 
      prog.hide(); 
    }
  });

  function show(res) {
    $("#out").hidden = false;
    const el = $("#viewer"); 
    el.innerHTML = "";
    const v = $3Dmol.createViewer(el, { backgroundColor: "#ffffff" });
    v.addModel(res.protein_pdb, "pdb");
    v.setStyle({ model: 0 }, { cartoon: { color: "#94a3b8", opacity: 0.85 } });
    
    // Highlight contact residues in cyan
    res.contacts.forEach((c) => v.addStyle({ model: 0, chain: c.chain, resi: c.resi }, { stick: { colorscheme: "cyanCarbon", radius: 0.14 } }));
    
    // Docked pose in green
    v.addModel(res.pose_pdbqt, "pdbqt");
    v.setStyle({ model: 1 }, { stick: { colorscheme: "greenCarbon", radius: 0.28 } });
    
    // Crystal ligand in orange for reference
    v.addModel(res.crystal_pdb, "pdb");
    v.setStyle({ model: 2 }, { stick: { colorscheme: "orangeCarbon", radius: 0.12 } });
    
    v.zoomTo({ model: 1 }); 
    v.render();

    const score = res.score;
    const conclusion = score <= -9.0 
      ? "Exceptional binding pose stability with strong negative free energy (≤ -9.0 kcal/mol)." 
      : score <= -7.0 
        ? "Favorable binding free energy (typical drug-like pocket fit, -7.0 to -9.0 kcal/mol)." 
        : "Weak physical complementarity (> -7.0 kcal/mol); significant steric clashes or suboptimal H-bonding.";

    $("#dock-info").innerHTML = `
      <div class="result-card-header" style="margin-bottom:8px;">
        <h3 class="card-title" style="font-size:1.3rem;">Docking Result Summary</h3>
        <span class="potency-badge ${score <= -7.0 ? 'strong' : 'moderate'}">
          ${score <= -7.0 ? 'Favorable Pose' : 'Marginal Fit'}
        </span>
      </div>

      <div class="result-value-row" style="margin-bottom:12px;">
        <div class="result-value-big" style="font-size:2.4rem;">${score.toFixed(2)}</div>
        <div class="result-conc-pill" style="font-size:0.95rem;">kcal / mol (Vina)</div>
      </div>

      <!-- Biological Conclusion Box -->
      <div class="conclusion-box" style="margin:14px 0;">
        <div class="conclusion-header">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
          Structural Fit &amp; Pose Conclusion
        </div>
        <p class="conclusion-body">
          ${conclusion} The docked pose identified <b>${res.contacts.length} pocket residue contact(s)</b> within 4 &Aring;.
        </p>
      </div>

      <div style="background:rgba(11,17,32,0.6); padding:12px 14px; border-radius:var(--radius-md); border:1px solid var(--border-card); margin-bottom:14px;">
        <div style="font-size:0.8rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">
          3D Visual Legend
        </div>
        <div style="display:flex; flex-direction:column; gap:6px; font-size:0.82rem;">
          <div style="display:flex; align-items:center; gap:8px;">
            <i style="display:inline-block; width:14px; height:14px; border-radius:3px; background:#10b981;"></i> 
            <span><b>Docked Ligand Pose (Green):</b> Lowest-energy predicted binding conformation.</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <i style="display:inline-block; width:14px; height:14px; border-radius:3px; background:#f59e0b;"></i> 
            <span><b>Native Crystal Ligand (Orange):</b> Original experimental co-crystallized pose.</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <i style="display:inline-block; width:14px; height:14px; border-radius:3px; background:#0ea5e9;"></i> 
            <span><b>Pocket Contact Residues (Cyan Sticks):</b> Amino acids within 4.0 &Aring; of docked pose.</span>
          </div>
        </div>
      </div>

      <div style="font-size:0.85rem; margin-bottom:8px;">
        <b>Key Contact Residues:</b> 
        <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:6px;">
          ${res.contacts.length ? res.contacts.map((c) => `
            <span style="font-family:var(--font-mono); font-size:0.75rem; background:rgba(14,165,233,0.15); color:var(--cyan-primary); border:1px solid rgba(14,165,233,0.3); padding:2px 8px; border-radius:var(--radius-pill);">
              ${esc(c.name)}${c.resi}(${esc(c.chain)})
            </span>`).join("") : '<span style="color:var(--text-dim);">No contacts &lt; 4 &Aring;</span>'}
        </div>
      </div>

      <div style="font-size:0.78rem; color:var(--text-dim); margin-top:14px; line-height:1.4;">
        <b>Physics Note:</b> AutoDock Vina scores reflect empirical intermolecular scoring functions (electrostatics, hydrophobics, and desolvation) and are not directly equivalent to statistical pKi or pIC50 machine learning predictions.
      </div>`;
    
    $("#out").scrollIntoView({ behavior: "smooth", block: "start" });
  }
})();
