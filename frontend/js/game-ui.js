import { drawBoard, drawHit, drawHits, getScoreFromClick } from './board-canvas.js';

let totalScore = 0;
const throws = [];

export function initGamePage() {
  const canvas = document.getElementById('dartboard');
  const btnStart = document.getElementById('btn-start-game');
  const btnStop = document.getElementById('btn-stop-game');

  drawBoard(canvas);

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

    li.appendChild(fieldSpan);
    li.appendChild(pointsSpan);
    list.appendChild(li);
  }
}
