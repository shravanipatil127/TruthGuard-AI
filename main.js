/**
 * TruthGuard AI - Main JavaScript
 * Handles: API calls, UI animations, chart rendering, file uploads
 */

// ─── UTILITIES ──────────────────────────────────────────────

const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function showAlert(msg, type = 'info', container = null) {
  const icons = { success: '✓', error: '✕', info: 'ℹ' };
  const el = document.createElement('div');
  el.className = `alert alert-${type} animate-fade-in`;
  el.innerHTML = `<span>${icons[type] || 'ℹ'}</span> ${msg}`;
  el.style.cursor = 'pointer';
  el.onclick = () => el.remove();
  const target = container || $('#flash-container') || document.body;
  target.prepend(el);
  setTimeout(() => el.remove(), 5000);
}

function formatDate(isoStr) {
  if (!isoStr) return 'N/A';
  const d = new Date(isoStr.replace('T', ' '));
  return d.toLocaleDateString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit'
  });
}

function animateNumber(el, from, to, duration = 1000, suffix = '') {
  const start = performance.now();
  function update(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // Ease out cubic
    el.textContent = Math.round(from + (to - from) * eased) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// ─── AUTO-DISMISS FLASH MESSAGES ───────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  $$('.alert[data-auto-dismiss]').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity 0.5s';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 500);
    }, 4000);
  });

  // Animate stat numbers
  $$('[data-count]').forEach(el => {
    const target = parseInt(el.dataset.count, 10);
    const suffix = el.dataset.suffix || '';
    animateNumber(el, 0, target, 1200, suffix);
  });
});

// ─── TEXT DETECTION ────────────────────────────────────────

function initTextDetection() {
  const form         = $('#text-detect-form');
  const textarea     = $('#article-input');
  const charCounter  = $('#char-counter');
  const analyzeBtn   = $('#analyze-btn');
  const loader       = $('#text-loader');
  const resultArea   = $('#text-result');
  const clearBtn     = $('#clear-btn');

  if (!form) return;

  // Character counter
  textarea?.addEventListener('input', () => {
    const len = textarea.value.length;
    if (charCounter) {
      charCounter.textContent = `${len} / 5000`;
      charCounter.style.color = len > 4800
        ? 'var(--neon-red)'
        : len > 4000
          ? 'var(--neon-orange)'
          : 'var(--text-dim)';
    }
    if (analyzeBtn) analyzeBtn.disabled = len < 10 || len > 5000;
  });

  // Clear
  clearBtn?.addEventListener('click', () => {
    textarea.value = '';
    if (charCounter) charCounter.textContent = '0 / 5000';
    if (analyzeBtn) analyzeBtn.disabled = true;
    if (resultArea) resultArea.innerHTML = '';
  });

  // Submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = textarea.value.trim();
    if (text.length < 10) {
      showAlert('Please enter at least 10 characters.', 'error');
      return;
    }

    analyzeBtn.disabled = true;
    if (loader) loader.classList.add('active');
    if (resultArea) resultArea.innerHTML = '';

    try {
      const res = await fetch('/api/analyze/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await res.json();

      if (!res.ok) {
        showAlert(data.error || 'Analysis failed.', 'error');
        return;
      }

      renderTextResult(data, resultArea);
    } catch (err) {
      showAlert('Network error. Please try again.', 'error');
    } finally {
      analyzeBtn.disabled = false;
      if (loader) loader.classList.remove('active');
    }
  });

  // Sample texts
  $$('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      textarea.value = btn.dataset.text;
      textarea.dispatchEvent(new Event('input'));
    });
  });
}

function renderTextResult(data, container) {
  const resultClass = data.result.toLowerCase().replace(/[\s-]/g, '-');
  const icons = { 'Fake': '⚠', 'Real': '✓' };
  const icon = icons[data.result] || '?';

  const suspiciousHtml = data.suspicious_found?.length
    ? `<div class="suspicious-tags">
        ${data.suspicious_found.map(w => `<span class="suspicious-tag">${w}</span>`).join('')}
       </div>`
    : '';

  container.innerHTML = `
    <div class="result-panel ${resultClass} animate-fade-up">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;flex-wrap:wrap">
        <div>
          <div class="result-verdict ${resultClass}">${icon} ${data.result}</div>
          <div style="color:var(--text-secondary);font-size:.9rem;margin-top:4px">
            ${data.result === 'Fake'
              ? 'This content shows signs of misinformation.'
              : 'This content appears to be legitimate news.'}
          </div>
        </div>
        <div style="margin-left:auto;display:flex;gap:8px;flex-wrap:wrap">
          ${data.scan_id
            ? `<a href="/api/report/${data.scan_id}" class="btn btn-secondary btn-sm" target="_blank">⬇ PDF Report</a>`
            : ''}
        </div>
      </div>

      <div class="confidence-bar-wrap">
        <div class="confidence-bar-label">
          <span>Confidence</span>
          <span style="color:${data.result==='Fake'?'var(--neon-red)':'var(--neon-green)'}">${data.confidence}%</span>
        </div>
        <div class="confidence-bar">
          <div class="confidence-bar-fill ${resultClass}" id="conf-bar-fill" style="width:0%"></div>
        </div>
      </div>

      <div class="prob-grid">
        <div class="prob-item">
          <div class="prob-item-label">Real Probability</div>
          <div class="prob-item-value" style="color:var(--neon-green)">${data.real_prob}%</div>
        </div>
        <div class="prob-item">
          <div class="prob-item-label">Fake Probability</div>
          <div class="prob-item-value" style="color:var(--neon-red)">${data.fake_prob}%</div>
        </div>
        <div class="prob-item">
          <div class="prob-item-label">Word Count</div>
          <div class="prob-item-value" style="color:var(--neon-cyan)">${data.word_count || 0}</div>
        </div>
      </div>

      ${data.suspicious_found?.length
        ? `<div style="margin-top:20px">
            <div style="font-size:.82rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">
              🔴 Suspicious Words Detected
            </div>
            ${suspiciousHtml}
           </div>`
        : ''}

      ${data.highlighted_text
        ? `<div style="margin-top:24px">
            <div style="font-size:.82rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px">
              Analyzed Text
            </div>
            <div style="font-family:var(--font-mono);font-size:.88rem;line-height:1.8;color:var(--text-primary);
                        background:rgba(0,0,0,.4);padding:16px;border-radius:10px;border:1px solid var(--border-dim)">
              ${data.highlighted_text}
            </div>
           </div>`
        : ''}
    </div>
  `;

  // Animate confidence bar
  setTimeout(() => {
    const fill = $('#conf-bar-fill');
    if (fill) fill.style.width = data.confidence + '%';
  }, 100);
}

// ─── IMAGE DETECTION ────────────────────────────────────────

function initImageDetection() {
  const uploadZone  = $('#upload-zone');
  const fileInput   = $('#image-input');
  const preview     = $('#image-preview');
  const analyzeBtn  = $('#analyze-image-btn');
  const loader      = $('#image-loader');
  const resultArea  = $('#image-result');
  const clearBtn    = $('#clear-image-btn');

  if (!uploadZone) return;

  // Drag & drop
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });

  ['dragleave', 'dragend'].forEach(evt => {
    uploadZone.addEventListener(evt, () => uploadZone.classList.remove('dragover'));
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleImageFile(file);
  });

  fileInput?.addEventListener('change', () => {
    if (fileInput.files[0]) handleImageFile(fileInput.files[0]);
  });

  clearBtn?.addEventListener('click', () => {
    if (preview) { preview.src = ''; preview.style.display = 'none'; }
    if (fileInput) fileInput.value = '';
    if (analyzeBtn) analyzeBtn.disabled = true;
    if (resultArea) resultArea.innerHTML = '';
    $('#upload-zone-content')?.style && (uploadZoneContent.style.display = 'block');
  });

  analyzeBtn?.addEventListener('click', async () => {
    if (!fileInput?.files[0]) return;

    analyzeBtn.disabled = true;
    if (loader) loader.classList.add('active');
    if (resultArea) resultArea.innerHTML = '';

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    try {
      const res = await fetch('/api/analyze/image', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (!res.ok) {
        showAlert(data.error || 'Analysis failed.', 'error');
        return;
      }
      renderImageResult(data, resultArea);
    } catch (err) {
      showAlert('Network error. Please try again.', 'error');
    } finally {
      analyzeBtn.disabled = false;
      if (loader) loader.classList.remove('active');
    }
  });
}

function handleImageFile(file) {
  const allowed = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showAlert('Invalid file type. Please upload an image.', 'error');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showAlert('File too large (max 16 MB).', 'error');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const preview = $('#image-preview');
    const zoneContent = $('#upload-zone-content');
    if (preview) {
      preview.src = e.target.result;
      preview.style.display = 'block';
    }
    if (zoneContent) zoneContent.style.display = 'none';

    // Also set the file input via DataTransfer
    const dt = new DataTransfer();
    dt.items.add(file);
    const fileInput = $('#image-input');
    if (fileInput) fileInput.files = dt.files;

    const analyzeBtn = $('#analyze-image-btn');
    if (analyzeBtn) analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function renderImageResult(data, container) {
  const classMap = {
    'Real Photo':    { cls: 'real',         color: 'var(--neon-green)',  icon: '📷' },
    'AI-Generated':  { cls: 'ai-generated', color: '#a78bff',            icon: '🤖' },
    'Manipulated':   { cls: 'manipulated',  color: 'var(--neon-orange)', icon: '⚠' }
  };
  const info = classMap[data.result] || { cls: 'real', color: 'var(--neon-cyan)', icon: '?' };

  const scoresHtml = data.class_scores
    ? Object.entries(data.class_scores).map(([cls, score]) => {
        const ci = classMap[cls] || { color: 'var(--neon-cyan)' };
        return `
          <div style="margin-bottom:14px">
            <div style="display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:6px">
              <span style="color:var(--text-secondary)">${cls}</span>
              <span style="color:${ci.color};font-weight:700">${score}%</span>
            </div>
            <div class="confidence-bar">
              <div class="confidence-bar-fill ${cls.toLowerCase().replace(/[\s-]/g,'')}"
                   style="width:${score}%;background:${ci.color};box-shadow:0 0 10px ${ci.color}40"></div>
            </div>
          </div>`;
      }).join('')
    : '';

  container.innerHTML = `
    <div class="result-panel ${info.cls} animate-fade-up">
      <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;margin-bottom:20px">
        <div>
          <div class="result-verdict ${info.cls}">${info.icon} ${data.result}</div>
          <div style="color:var(--text-secondary);font-size:.9rem;margin-top:4px">
            ${getImageVerdict(data.result)}
          </div>
        </div>
        <div style="margin-left:auto;display:flex;gap:8px;flex-wrap:wrap">
          ${data.scan_id
            ? `<a href="/api/report/${data.scan_id}" class="btn btn-secondary btn-sm" target="_blank">⬇ PDF Report</a>`
            : ''}
        </div>
      </div>

      <div class="confidence-bar-wrap">
        <div class="confidence-bar-label">
          <span>Prediction Confidence</span>
          <span style="color:${info.color}">${data.confidence}%</span>
        </div>
        <div class="confidence-bar">
          <div class="confidence-bar-fill ${info.cls}" id="img-conf-bar" style="width:0%;background:${info.color};box-shadow:0 0 12px ${info.color}80"></div>
        </div>
      </div>

      <div style="margin-top:24px">
        <div style="font-size:.82rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px">
          Class Probabilities
        </div>
        ${scoresHtml}
      </div>

      <div class="prob-grid" style="margin-top:16px">
        <div class="prob-item">
          <div class="prob-item-label">Image Size</div>
          <div style="font-family:var(--font-mono);font-size:1rem;color:var(--neon-cyan)">${data.image_size || 'N/A'}</div>
        </div>
        <div class="prob-item">
          <div class="prob-item-label">Method</div>
          <div style="font-family:var(--font-mono);font-size:.8rem;color:var(--text-secondary)">${data.analysis_method || 'N/A'}</div>
        </div>
      </div>
    </div>
  `;

  setTimeout(() => {
    const bar = $('#img-conf-bar');
    if (bar) bar.style.transition = 'width 1s cubic-bezier(0.4,0,0.2,1)';
    if (bar) bar.style.width = data.confidence + '%';
  }, 100);
}

function getImageVerdict(result) {
  const verdicts = {
    'Real Photo':   'This image appears to be an authentic, unmodified photograph.',
    'AI-Generated': 'This image shows patterns consistent with AI generation (GAN/Diffusion).',
    'Manipulated':  'This image shows signs of digital manipulation or compositing.'
  };
  return verdicts[result] || 'Analysis complete.';
}

// ─── DASHBOARD CHARTS ───────────────────────────────────────

function initDashboardCharts(stats) {
  if (typeof Chart === 'undefined') return;

  Chart.defaults.color = '#8ba8bc';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';

  const neonColors = {
    fake:      '#ff2d55',
    real:      '#00ff88',
    aiGen:     '#a78bff',
    manip:     '#ff6b35',
    text:      '#00d4ff',
    image:     '#7b2fff'
  };

  // ── Result Pie Chart ──
  const pieCtx = $('#result-pie')?.getContext('2d');
  if (pieCtx) {
    const resultData = [
      stats.real_count || 0,
      stats.fake_count || 0,
      stats.ai_generated_count || 0,
      stats.manipulated_count || 0
    ];
    const total = resultData.reduce((a, b) => a + b, 0);

    new Chart(pieCtx, {
      type: 'doughnut',
      data: {
        labels: ['Real', 'Fake', 'AI-Generated', 'Manipulated'],
        datasets: [{
          data: total ? resultData : [1, 0, 0, 0],
          backgroundColor: [
            neonColors.real + '99',
            neonColors.fake + '99',
            neonColors.aiGen + '99',
            neonColors.manip + '99'
          ],
          borderColor: [neonColors.real, neonColors.fake, neonColors.aiGen, neonColors.manip],
          borderWidth: 2,
          hoverBorderWidth: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        cutout: '65%',
        plugins: {
          legend: { position: 'bottom', labels: { padding: 16, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const val = ctx.raw;
                const pct = total ? ((val / total) * 100).toFixed(1) : 0;
                return ` ${ctx.label}: ${val} (${pct}%)`;
              }
            }
          }
        }
      }
    });
  }

  // ── Scan Type Bar Chart ──
  const barCtx = $('#type-bar')?.getContext('2d');
  if (barCtx) {
    new Chart(barCtx, {
      type: 'bar',
      data: {
        labels: ['Text Scans', 'Image Scans'],
        datasets: [{
          label: 'Scans',
          data: [stats.text_scans || 0, stats.image_scans || 0],
          backgroundColor: [neonColors.text + '33', neonColors.image + '33'],
          borderColor: [neonColors.text, neonColors.image],
          borderWidth: 2,
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { stepSize: 1 }
          },
          x: { grid: { display: false } }
        }
      }
    });
  }

  // ── Daily Activity Line Chart ──
  const lineCtx = $('#activity-line')?.getContext('2d');
  if (lineCtx && stats.daily_activity) {
    // Fill missing days in last 7
    const days7 = [];
    for (let i = 6; i >= 0; i--) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      days7.push(d.toISOString().slice(0, 10));
    }
    const actMap = Object.fromEntries(stats.daily_activity.map(a => [a.day, a.count]));
    const counts = days7.map(d => actMap[d] || 0);
    const labels = days7.map(d => {
      const dt = new Date(d + 'T00:00:00');
      return dt.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
    });

    new Chart(lineCtx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Scans',
          data: counts,
          borderColor: neonColors.text,
          backgroundColor: neonColors.text + '15',
          fill: true,
          tension: 0.4,
          pointBackgroundColor: neonColors.text,
          pointBorderColor: '#0a1628',
          pointBorderWidth: 2,
          pointRadius: 5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { stepSize: 1 }
          },
          x: { grid: { display: false } }
        }
      }
    });
  }
}

// ─── PASSWORD STRENGTH ──────────────────────────────────────

function initPasswordStrength() {
  const pwInput = $('#password');
  const bar = $('.password-strength-bar');
  if (!pwInput || !bar) return;

  pwInput.addEventListener('input', () => {
    const pw = pwInput.value;
    let strength = 0;
    if (pw.length >= 8) strength++;
    if (/[A-Z]/.test(pw)) strength++;
    if (/[0-9]/.test(pw)) strength++;
    if (/[^A-Za-z0-9]/.test(pw)) strength++;

    const widths  = ['0%', '25%', '50%', '75%', '100%'];
    const colors  = ['', '#ff2d55', '#ff6b35', '#ffe600', '#00ff88'];
    bar.style.width  = widths[strength];
    bar.style.background = colors[strength] || '';
  });
}

// ─── INIT ────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initTextDetection();
  initImageDetection();
  initPasswordStrength();
});
