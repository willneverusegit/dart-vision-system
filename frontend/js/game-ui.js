import { drawBoard, drawHit, drawHits, getScoreFromClick } from './board-canvas.js';

let totalScore = 0;
const throws = [];
const CONFIDENCE_REVIEW_THRESHOLD = 0.6;
let debugMode = false;

export function initGamePage() {
  const canvas = document.getElementById('dartboard');
  const btnStart = document.getElementById('btn-start-game');
  const btnStop = document.getElementById('btn-stop-game');
  const btnDebug = document.getElementById('btn-toggle-debug');

  drawBoard(canvas);

  if (btnDebug) {
    btnDebug.addEventListener('click', () => {
      const active = toggleDebugMode();
      btnDebug.textContent = active ? 'Debug (AN)' : 'Debug';
    });
  }

  canvas.addEventListener('click', (e) => {
    if (!btnStop.disabled) {
      handleThrow(canvas, e);
    }
  });

  btnStart.addEventListener('click', () => {
    totalScore = 0;
    throws.length = 0;
    updateScoreDisplay();
    updateHistory();
    drawBoard(canvas);
    btnStart.disabled = true;
    btnStop.disabled = false;
  });

  btnStop.addEventListener('click', () => {
    btnStart.disabled = false;
    btnStop.disabled = true;
  });
}

function handleThrow(canvas, event) {
  const result = getScoreFromClick(canvas, event);
  if (result.score === 0 && result.multiplier === 0) return;

  drawHit(canvas, event, result.field);
  throws.push(result);
  totalScore += result.score;

  const lastThrow = document.getElementById('last-throw');
  lastThrow.textContent = `${result.field} \u2192 +${result.score}`;

  animateScore(result.score, result.field);
  updateScoreDisplay();
  updateHistory();
}

export function animateScore(scoreValue, fieldName) {
  const container = document.getElementById('score-popup-container');
  if (!container) return;

  const popup = document.createElement('div');
  popup.className = 'score-popup';
  popup.textContent = `+${scoreValue} ${fieldName}`;
  container.appendChild(popup);

  popup.addEventListener('animationend', () => {
    popup.remove();
  });
}

function updateScoreDisplay() {
  const el = document.getElementById('score-value');
  el.textContent = totalScore;
  el.classList.remove('score-pulse');
  // Force reflow to restart animation
  void el.offsetWidth;
  el.classList.add('score-pulse');
}

function updateHistory() {
  const list = document.getElementById('history-list');
  list.replaceChildren();
  for (let i = throws.length - 1; i >= 0; i--) {
    const t = throws[i];
    const li = document.createElement('li');

    const fieldSpan = document.createElement('span');
    fieldSpan.className = 'field';
    fieldSpan.textContent = `#${i + 1} ${t.field}`;

    const pointsSpan = document.createElement('span');
    pointsSpan.className = 'points';
    pointsSpan.textContent = `+${t.score}`;

    if (t.confidence !== undefined) {
      const confSpan = document.createElement('span');
      confSpan.className = getConfidenceClass(t.confidence);
      confSpan.textContent = ` ${Math.round(t.confidence * 100)}%`;
      pointsSpan.appendChild(confSpan);
    }

    li.appendChild(fieldSpan);
    li.appendChild(pointsSpan);
    list.appendChild(li);
  }
}

function getConfidenceClass(confidence) {
  if (confidence >= 0.8) return 'hit-green';
  if (confidence >= 0.6) return 'hit-yellow';
  return 'hit-red';
}

export function showConfidence(confidence) {
  const el = document.getElementById('confidence-indicator');
  if (!el) return;
  const pct = Math.round(confidence * 100);
  el.textContent = `${pct}%`;
  el.className = `confidence-indicator ${getConfidenceClass(confidence)}`;
}

export function showReviewDialog(field, alternatives, onSelect) {
  // Remove existing dialog if any
  const existing = document.getElementById('review-dialog');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'review-dialog';
  overlay.className = 'review-overlay';

  const dialog = document.createElement('div');
  dialog.className = 'review-dialog-box';

  const title = document.createElement('h3');
  title.textContent = 'Treffer bestaetigen?';
  dialog.appendChild(title);

  const desc = document.createElement('p');
  desc.textContent = `Erkannt: ${field} — Niedrige Confidence`;
  dialog.appendChild(desc);

  const btnContainer = document.createElement('div');
  btnContainer.className = 'review-buttons';

  const allOptions = [field, ...alternatives.filter(a => a !== field)];
  for (const option of allOptions) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-primary review-btn';
    btn.textContent = option;
    btn.addEventListener('click', () => {
      overlay.remove();
      if (onSelect) onSelect(option);
    });
    btnContainer.appendChild(btn);
  }

  // Miss button
  const missBtn = document.createElement('button');
  missBtn.className = 'btn btn-danger review-btn';
  missBtn.textContent = 'Miss';
  missBtn.addEventListener('click', () => {
    overlay.remove();
    if (onSelect) onSelect('MISS');
  });
  btnContainer.appendChild(missBtn);

  dialog.appendChild(btnContainer);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
}

export function toggleDebugMode() {
  debugMode = !debugMode;
  const panel = document.getElementById('debug-panel');
  if (panel) {
    panel.style.display = debugMode ? 'grid' : 'none';
  }
  return debugMode;
}

export function showDebugThumbnails(debugOutput) {
  if (!debugMode || !debugOutput) return;

  const panel = document.getElementById('debug-panel');
  if (!panel) return;
  panel.replaceChildren();

  const stages = [
    { label: 'Grauwert', data: debugOutput.grayscale_b64 },
    { label: 'Differenz', data: debugOutput.diff_b64 },
    { label: 'Canny', data: debugOutput.canny_b64 },
    { label: 'Konturen', data: debugOutput.contours_b64 },
  ];

  for (const stage of stages) {
    if (!stage.data) continue;

    const container = document.createElement('div');
    container.className = 'debug-thumb';

    const label = document.createElement('span');
    label.className = 'debug-label';
    label.textContent = stage.label;

    const img = document.createElement('img');
    img.src = `data:image/jpeg;base64,${stage.data}`;
    img.alt = stage.label;

    container.appendChild(label);
    container.appendChild(img);
    panel.appendChild(container);
  }
}
