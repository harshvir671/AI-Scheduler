

const ALGO_COLORS = {
  FCFS: '#5EEAD4',
  SJF: '#8B9EFF',
  SRTF: '#C29FFF',
  RR: '#F5A623',
  Priority: '#FF8FA3',
};

const ALGO_ORDER = ['FCFS', 'SJF', 'SRTF', 'RR', 'Priority'];

let currentProcesses = [];
let currentResults = null;
let activeGanttTab = 'FCFS';
let metricsChartCtx = null;

// ---------------- DOM refs ----------------
const el = (id) => document.getElementById(id);

const btnGenerate = el('btn-generate');
const btnResim = el('btn-resimulate');

// ---------------- helpers ----------------
function fmt(n) {
  if (n === null || n === undefined) return '—';
  return Number(n).toFixed(2);
}

function pidColor(pid, palette) {
  // deterministic color per PID for gantt charts, cycling a palette
  let hash = 0;
  for (let i = 0; i < pid.length; i++) hash = (hash * 31 + pid.charCodeAt(i)) >>> 0;
  return palette[hash % palette.length];
}

const PID_PALETTE = [
  '#5EEAD4', '#8B9EFF', '#C29FFF', '#F5A623', '#FF8FA3',
  '#7EE787', '#FFB454', '#79C0FF', '#F08CFF', '#E3B341',
];


async function generateWorkload() {
  const payload = {
    num_processes: parseInt(el('num-processes').value),
    max_arrival: parseInt(el('max-arrival').value),
    max_burst: parseInt(el('max-burst').value),
    max_priority: parseInt(el('max-priority').value),
    seed: el('seed-input').value.trim(),
  };

  btnGenerate.textContent = '… generating';
  btnGenerate.disabled = true;

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    currentProcesses = data.processes;
    renderProcessTable(currentProcesses);
    btnResim.disabled = false;
    await runSimulation();
  } catch (e) {
    console.error(e);
  } finally {
    btnGenerate.textContent = '▸ GENERATE WORKLOAD';
    btnGenerate.disabled = false;
  }
}

function renderProcessTable(processes) {
  const body = el('process-table-body');
  if (!processes.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty-row">no workload generated yet</td></tr>';
    return;
  }
  body.innerHTML = processes.map(p => `
    <tr>
      <td class="pid-cell">${p.pid}</td>
      <td>${p.arrival_time}</td>
      <td>${p.burst_time}</td>
      <td>${p.priority}</td>
      <td>${p.memory_required}</td>
    </tr>
  `).join('');
}



async function runSimulation() {
  if (!currentProcesses.length) return;

  const payload = {
    processes: currentProcesses,
    rr_quantum: parseInt(el('rr-quantum').value) || 4,
    use_aging: el('use-aging').checked,
  };

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    currentResults = data.results;
    renderMetricsTable(currentResults);
    renderMetricsChart(currentResults);
    renderGanttTabs(currentResults);
    renderGantt(activeGanttTab);
    renderStarvation(data.starvation);
    renderAIPrediction(data.ai_prediction);
    loadHistory();
  } catch (e) {
    console.error(e);
  }
}



function algoKeyOrder(results) {
  return ALGO_ORDER.filter(k => results[k]);
}

function renderMetricsTable(results) {
  const body = el('metrics-table-body');
  const keys = algoKeyOrder(results);
  body.innerHTML = keys.map(key => {
    const r = results[key];
    return `
      <tr>
        <td>
          <span class="algo-name-cell">
            <span class="algo-swatch" style="background:${ALGO_COLORS[key]}"></span>
            ${r.algorithm}
          </span>
        </td>
        <td>${fmt(r.avg_waiting_time)}</td>
        <td>${fmt(r.avg_turnaround_time)}</td>
        <td>${fmt(r.avg_response_time)}</td>
        <td>${fmt(r.cpu_utilization)}%</td>
      </tr>
    `;
  }).join('');
}

function renderMetricsChart(results) {
  const canvas = el('metrics-chart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 180 * dpr;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = '180px';
  ctx.scale(dpr, dpr);

  const keys = algoKeyOrder(results);
  const waitVals = keys.map(k => results[k].avg_waiting_time);
  const turnVals = keys.map(k => results[k].avg_turnaround_time);

  const W = rect.width, H = 180;
  const padL = 36, padB = 22, padT = 10, padR = 10;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const maxVal = Math.max(...waitVals, ...turnVals, 1) * 1.15;

  ctx.clearRect(0, 0, W, H);

  // gridlines
  ctx.strokeStyle = '#1F2A33';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#4A5862';
  ctx.font = '9px JetBrains Mono, monospace';
  for (let i = 0; i <= 4; i++) {
    const y = padT + plotH - (plotH * i / 4);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.fillText(Math.round(maxVal * i / 4), 2, y + 3);
  }

  const groupW = plotW / keys.length;
  const barW = groupW * 0.32;

  keys.forEach((key, i) => {
    const groupX = padL + i * groupW + groupW * 0.15;
    const waitH = (waitVals[i] / maxVal) * plotH;
    const turnH = (turnVals[i] / maxVal) * plotH;

    ctx.fillStyle = ALGO_COLORS[key];
    ctx.globalAlpha = 1;
    ctx.fillRect(groupX, padT + plotH - waitH, barW, waitH);

    ctx.globalAlpha = 0.45;
    ctx.fillRect(groupX + barW + 4, padT + plotH - turnH, barW, turnH);
    ctx.globalAlpha = 1;

    ctx.fillStyle = '#6B7C87';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(key, groupX + barW, H - 6);
  });
  ctx.textAlign = 'left';

  // legend
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.fillStyle = '#D7E2E9';
  ctx.fillRect(W - 150, 4, 8, 8);
  ctx.fillText('avg wait', W - 138, 11);
  ctx.globalAlpha = 0.45;
  ctx.fillRect(W - 80, 4, 8, 8);
  ctx.globalAlpha = 1;
  ctx.fillText('turnaround', W - 68, 11);
}



function renderGanttTabs(results) {
  const tabsEl = el('gantt-tabs');
  const keys = algoKeyOrder(results);
  if (!keys.includes(activeGanttTab)) activeGanttTab = keys[0];

  tabsEl.innerHTML = keys.map(key => `
    <div class="gantt-tab ${key === activeGanttTab ? 'active' : ''}" data-key="${key}">
      ${results[key].algorithm}
    </div>
  `).join('');

  tabsEl.querySelectorAll('.gantt-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      activeGanttTab = tab.dataset.key;
      renderGanttTabs(currentResults);
      renderGantt(activeGanttTab);
    });
  });
}

function renderGantt(key) {
  const container = el('gantt-container');
  if (!currentResults || !currentResults[key]) {
    container.innerHTML = '<div class="empty-row">run a simulation to see scheduling timelines</div>';
    return;
  }
  const result = currentResults[key];
  const gantt = result.gantt;
  if (!gantt.length) {
    container.innerHTML = '<div class="empty-row">no gantt data</div>';
    return;
  }

  const totalTime = result.total_time || gantt[gantt.length - 1].end;
  const pxPerUnit = Math.max(6, Math.min(28, 900 / totalTime));

  const blocks = gantt.map(g => {
    const width = (g.end - g.start) * pxPerUnit;
    const color = pidColor(g.pid, PID_PALETTE);
    return `<div class="gantt-block" style="width:${width}px; background:${color}"
                  data-tooltip="${g.pid}: ${g.start}→${g.end}">${width > 22 ? g.pid : ''}</div>`;
  }).join('');

  // axis ticks every ~10% of total width, snapped to gantt boundaries
  const boundaries = [...new Set(gantt.flatMap(g => [g.start, g.end]))].sort((a, b) => a - b);
  const axisTicks = boundaries.map(t => {
    const x = 60 + t * pxPerUnit;
    return `<span class="gantt-axis-tick" style="left:${x}px">${t}</span>`;
  }).join('');

  const uniquePids = [...new Set(gantt.map(g => g.pid))];
  const legend = uniquePids.map(pid => `
    <div class="gantt-legend-item">
      <span class="algo-swatch" style="background:${pidColor(pid, PID_PALETTE)}"></span>${pid}
    </div>
  `).join('');

  container.innerHTML = `
    <div class="gantt-row">
      <div class="gantt-row-label">${key}</div>
      <div class="gantt-track">${blocks}</div>
    </div>
    <div class="gantt-axis">${axisTicks}</div>
    <div class="gantt-legend">${legend}</div>
  `;
}



function renderStarvation(flags) {
  const body = el('starvation-body');
  if (!flags || flags.length === 0) {
    body.innerHTML = '<div class="starvation-ok">✓ No starvation detected — no process waited disproportionately longer than higher-priority processes.</div>';
    return;
  }
  body.innerHTML = flags.map(f => `
    <div class="starvation-flag">
      <span><strong>${f.pid}</strong> (priority ${f.priority}) waited ${f.waiting_time} units vs ${f.higher_priority_avg_wait} avg for higher-priority jobs</span>
      <span class="sev">${f.severity}×</span>
    </div>
  `).join('') + `<div style="margin-top:8px; font-size:10.5px; color:var(--dim2);">
      Tip: enable "Priority aging" in the control panel to prevent this.
    </div>`;
}



function renderAIPrediction(pred) {
  if (!pred) {
    el('ai-empty').classList.remove('hidden');
    el('ai-result').classList.add('hidden');
    return;
  }
  el('ai-empty').classList.add('hidden');
  el('ai-result').classList.remove('hidden');

  el('ai-algo-name').textContent = pred.recommended_algorithm;
  el('ai-algo-name').style.color = ALGO_COLORS[pred.recommended_algorithm] || '#5EEAD4';

  const pct = Math.round(pred.confidence * 100);
  el('confidence-bar-fill').style.width = pct + '%';
  el('confidence-pct').textContent = pct + '%';

  const probsEl = el('ai-probs');
  probsEl.innerHTML = Object.entries(pred.all_probabilities).map(([algo, p]) => `
    <div class="ai-prob-row">
      <span>${algo}</span>
      <span class="ai-prob-bar-track"><span class="ai-prob-bar-fill" style="width:${(p*100).toFixed(0)}%; background:${ALGO_COLORS[algo]}"></span></span>
      <span>${(p*100).toFixed(1)}%</span>
    </div>
  `).join('');

  const verdictEl = el('ai-verdict');
  if (pred.actual_best_by_waiting) {
    const correct = pred.actual_best_by_waiting === pred.recommended_algorithm;
    verdictEl.className = 'ai-verdict ' + (correct ? 'correct' : 'incorrect');
    verdictEl.textContent = correct
      ? `✓ Matches measured optimum (lowest avg waiting time was also ${pred.actual_best_by_waiting})`
      : `△ Measured optimum for avg waiting time was ${pred.actual_best_by_waiting} — the AI optimizes a blended objective (wait + turnaround + response + fairness), not waiting time alone.`;
  } else {
    verdictEl.textContent = '';
  }
}



async function loadModelInfo() {
  try {
    const res = await fetch('/api/model-info');
    const data = await res.json();
    const statusDot = el('model-status-dot');
    const statusText = el('model-status-text');

    if (!data.available) {
      statusDot.className = 'status-dot offline';
      statusText.textContent = 'model not trained — run ml/train.py';
      el('model-info-body').innerHTML = '<div class="empty-row">No trained model found. Run <code>python ml/train.py</code> to enable AI predictions.</div>';
      return;
    }
    statusDot.className = 'status-dot online';
    statusText.textContent = 'model online';

    const m = data.metrics;
    const body = el('model-info-body');
    body.innerHTML = `
      <div class="model-stat-grid">
        <div class="model-stat">
          <div class="model-stat-value">${(m.test_accuracy*100).toFixed(1)}%</div>
          <div class="model-stat-label">TEST ACCURACY</div>
        </div>
        <div class="model-stat">
          <div class="model-stat-value">${(m.cv_accuracy_mean*100).toFixed(1)}%</div>
          <div class="model-stat-label">5-FOLD CV ACCURACY</div>
        </div>
        <div class="model-stat">
          <div class="model-stat-value">${(m.macro_f1*100).toFixed(1)}%</div>
          <div class="model-stat-label">MACRO F1</div>
        </div>
        <div class="model-stat">
          <div class="model-stat-value">${m.n_train + m.n_test}</div>
          <div class="model-stat-label">TRAINING SAMPLES</div>
        </div>
      </div>
      <div style="font-size:11px; color:var(--dim); margin-bottom:6px;">FEATURE IMPORTANCE</div>
      ${Object.entries(m.feature_importances).map(([name, imp]) => `
        <div class="fi-bar-row">
          <span>${name}</span>
          <span class="fi-bar-track"><span class="fi-bar-fill" style="width:${(imp*100*5).toFixed(0)}%"></span></span>
          <span>${(imp*100).toFixed(1)}%</span>
        </div>
      `).join('')}
    `;
  } catch (e) {
    console.error(e);
  }
}



async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const data = await res.json();
    const summaryEl = el('history-summary');
    const listEl = el('history-list');

    if (!data.runs.length) {
      summaryEl.textContent = 'no runs logged yet';
      listEl.innerHTML = '';
      return;
    }

    summaryEl.innerHTML = `${data.runs.length} runs logged · AI matched the measured optimum <span class="acc-highlight">${(data.accuracy*100).toFixed(1)}%</span> of the time`;

    listEl.innerHTML = data.runs.slice(0, 12).map(r => {
      const hit = r.was_correct === 1;
      return `
        <div class="history-item">
          <span>${new Date(r.timestamp).toLocaleTimeString()}</span>
          <span>${r.num_processes} procs, burst~${fmt(r.avg_burst)}</span>
          <span style="color:${ALGO_COLORS[r.recommended_algorithm] || '#fff'}">${r.recommended_algorithm}</span>
          <span class="${hit ? 'hit' : 'miss'}">${hit ? '✓ match' : '△ ' + r.actual_best_algorithm}</span>
        </div>
      `;
    }).join('');
  } catch (e) {
    console.error(e);
  }
}



btnGenerate.addEventListener('click', generateWorkload);
btnResim.addEventListener('click', runSimulation);

window.addEventListener('resize', () => {
  if (currentResults) renderMetricsChart(currentResults);
});

loadModelInfo();
loadHistory();
