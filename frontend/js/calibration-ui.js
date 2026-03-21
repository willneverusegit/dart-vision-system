/**
 * Calibration UI — ROI Cropper and calibration wizard controls.
 */

let roiState = {
  active: false,
  startX: 0, startY: 0,
  x: 50, y: 50, w: 400, h: 400,
  dragging: false,
  resizing: false,
  dragOffsetX: 0, dragOffsetY: 0,
};

const HANDLE_SIZE = 10;

export function initCalibrationPage() {
  const container = document.getElementById('page-calibration');
  if (!container) return;

  container.replaceChildren();

  const h2 = document.createElement('h2');
  h2.textContent = 'Kalibrierung';
  container.appendChild(h2);

  // Calibration steps
  const steps = document.createElement('div');
  steps.className = 'calib-steps';
  steps.textContent = 'Schritt 1: Kamera auswählen → Schritt 2: ROI festlegen → Schritt 3: Board-Fitting';
  container.appendChild(steps);

  // ROI Canvas
  const canvasWrap = document.createElement('div');
  canvasWrap.className = 'roi-canvas-wrap';

  const canvas = document.createElement('canvas');
  canvas.id = 'roi-canvas';
  canvas.width = 640;
  canvas.height = 480;
  canvasWrap.appendChild(canvas);
  container.appendChild(canvasWrap);

  // Controls
  const controls = document.createElement('div');
  controls.className = 'calib-controls';

  const btnRoi = document.createElement('button');
  btnRoi.className = 'btn btn-primary';
  btnRoi.textContent = 'ROI auswählen';
  btnRoi.addEventListener('click', () => toggleRoiMode(canvas));
  controls.appendChild(btnRoi);

  const btnSave = document.createElement('button');
  btnSave.className = 'btn btn-primary';
  btnSave.textContent = 'ROI speichern';
  btnSave.addEventListener('click', () => saveRoi());
  controls.appendChild(btnSave);

  const roiInfo = document.createElement('div');
  roiInfo.id = 'roi-info';
  roiInfo.className = 'roi-info';
  controls.appendChild(roiInfo);

  container.appendChild(controls);

  // Draw placeholder
  drawRoiCanvas(canvas);

  // Mouse events for ROI
  canvas.addEventListener('mousedown', (e) => onMouseDown(canvas, e));
  canvas.addEventListener('mousemove', (e) => onMouseMove(canvas, e));
  canvas.addEventListener('mouseup', () => onMouseUp());
}

function toggleRoiMode(canvas) {
  roiState.active = !roiState.active;
  drawRoiCanvas(canvas);
}

function drawRoiCanvas(canvas) {
  const ctx = canvas.getContext('2d');

  // Draw placeholder background (simulated camera image)
  ctx.fillStyle = '#2a2a4a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#444';
  ctx.font = '16px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Kamerabild (Platzhalter)', canvas.width / 2, canvas.height / 2);

  if (!roiState.active) return;

  const { x, y, w, h } = roiState;

  // Dim area outside ROI
  ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
  ctx.fillRect(0, 0, canvas.width, y);
  ctx.fillRect(0, y + h, canvas.width, canvas.height - y - h);
  ctx.fillRect(0, y, x, h);
  ctx.fillRect(x + w, y, canvas.width - x - w, h);

  // ROI border
  ctx.strokeStyle = '#4ecca3';
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);

  // Resize handle (bottom-right)
  ctx.fillStyle = '#4ecca3';
  ctx.fillRect(x + w - HANDLE_SIZE, y + h - HANDLE_SIZE, HANDLE_SIZE, HANDLE_SIZE);

  // ROI dimensions text
  ctx.fillStyle = '#4ecca3';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`ROI: ${w}×${h} at (${x}, ${y})`, x, y - 5);

  updateRoiInfo();
}

function updateRoiInfo() {
  const info = document.getElementById('roi-info');
  if (info) {
    info.textContent = `x: ${roiState.x}, y: ${roiState.y}, w: ${roiState.w}, h: ${roiState.h}`;
  }
}

function onMouseDown(canvas, e) {
  if (!roiState.active) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const { x, y, w, h } = roiState;

  // Check resize handle
  if (mx >= x + w - HANDLE_SIZE && mx <= x + w && my >= y + h - HANDLE_SIZE && my <= y + h) {
    roiState.resizing = true;
    return;
  }

  // Check inside ROI for drag
  if (mx >= x && mx <= x + w && my >= y && my <= y + h) {
    roiState.dragging = true;
    roiState.dragOffsetX = mx - x;
    roiState.dragOffsetY = my - y;
  }
}

function onMouseMove(canvas, e) {
  if (!roiState.active) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  if (roiState.dragging) {
    roiState.x = Math.max(0, Math.min(canvas.width - roiState.w, mx - roiState.dragOffsetX));
    roiState.y = Math.max(0, Math.min(canvas.height - roiState.h, my - roiState.dragOffsetY));
    drawRoiCanvas(canvas);
  } else if (roiState.resizing) {
    roiState.w = Math.max(50, Math.min(canvas.width - roiState.x, mx - roiState.x));
    roiState.h = Math.max(50, Math.min(canvas.height - roiState.y, my - roiState.y));
    drawRoiCanvas(canvas);
  }
}

function onMouseUp() {
  roiState.dragging = false;
  roiState.resizing = false;
}

async function saveRoi() {
  const { x, y, w, h } = roiState;
  try {
    const resp = await fetch('/api/calibrate/roi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_id: '0', x, y, w, h }),
    });
    const data = await resp.json();
    const info = document.getElementById('roi-info');
    if (info) {
      info.textContent = `ROI gespeichert: ${JSON.stringify(data)}`;
    }
  } catch (err) {
    console.error('Failed to save ROI:', err);
  }
}
