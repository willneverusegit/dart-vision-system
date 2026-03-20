import { drawBoard, drawHit, getScoreFromClick } from './board-canvas.js';

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
  lastThrow.textContent = `${result.field} → +${result.score}`;

  updateScoreDisplay();
  updateHistory();
}

function updateScoreDisplay() {
  document.getElementById('score-value').textContent = totalScore;
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
