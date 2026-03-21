/**
 * Multi-Camera UI — camera thumbnails, stereo wizard, diagnostics panel.
 */

const API_BASE = '/api';

let cameras = [];
let diagnosticsInterval = null;

export function initMulticamPage() {
  const btnScan = document.getElementById('btn-scan-cameras');
  const btnStereo = document.getElementById('btn-start-stereo');

  btnScan.addEventListener('click', scanCameras);
  btnStereo.addEventListener('click', startStereoWizard);
}

async function scanCameras() {
  const grid = document.getElementById('cam-grid');
  grid.replaceChildren();

  try {
    const resp = await fetch(`${API_BASE}/cameras`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    cameras = await resp.json();

    if (cameras.length === 0) {
      const msg = document.createElement('p');
      msg.textContent = 'Keine Kameras gefunden.';
      grid.appendChild(msg);
      return;
    }

    for (const cam of cameras) {
      const card = createCameraCard(cam);
      grid.appendChild(card);
    }

    const btnStereo = document.getElementById('btn-start-stereo');
    btnStereo.disabled = cameras.length < 2;

    startDiagnostics();
  } catch (err) {
    const errMsg = document.createElement('p');
    errMsg.textContent = `Fehler: ${err.message}`;
    errMsg.className = 'error-msg';
    grid.appendChild(errMsg);
  }
}

function createCameraCard(cam) {
  const card = document.createElement('div');
  card.className = 'cam-card';
  card.dataset.camId = cam.id;

  const header = document.createElement('div');
  header.className = 'cam-card-header';

  const name = document.createElement('span');
  name.className = 'cam-name';
  name.textContent = cam.name || `Camera ${cam.id}`;

  const status = document.createElement('span');
  status.className = cam.available ? 'cam-status online' : 'cam-status offline';
  status.textContent = cam.available ? 'Online' : 'Offline';

  header.appendChild(name);
  header.appendChild(status);

  const preview = document.createElement('div');
  preview.className = 'cam-preview';
  preview.textContent = 'Preview';

  const info = document.createElement('div');
  info.className = 'cam-info';
  info.textContent = 'FPS: -- | Reproj: --';

  card.appendChild(header);
  card.appendChild(preview);
  card.appendChild(info);
  return card;
}

function startDiagnostics() {
  if (diagnosticsInterval) clearInterval(diagnosticsInterval);
  diagnosticsInterval = setInterval(updateDiagnostics, 2000);
  updateDiagnostics();
}

async function updateDiagnostics() {
  const content = document.getElementById('diag-content');
  if (!content) return;

  content.replaceChildren();

  for (const cam of cameras) {
    const row = document.createElement('div');
    row.className = 'diag-row';

    const label = document.createElement('span');
    label.textContent = cam.name || `Camera ${cam.id}`;

    const statusSpan = document.createElement('span');
    statusSpan.className = cam.available ? 'hit-green' : 'hit-red';
    statusSpan.textContent = cam.available ? 'OK' : 'FAIL';

    row.appendChild(label);
    row.appendChild(statusSpan);
    content.appendChild(row);
  }

  if (cameras.length >= 2) {
    const stereoRow = document.createElement('div');
    stereoRow.className = 'diag-row';

    const stereoLabel = document.createElement('span');
    stereoLabel.textContent = 'Stereo-Paar';

    const stereoStatus = document.createElement('span');
    stereoStatus.textContent = `${cameras[0].name || cameras[0].id} + ${cameras[1].name || cameras[1].id}`;

    stereoRow.appendChild(stereoLabel);
    stereoRow.appendChild(stereoStatus);
    content.appendChild(stereoRow);
  }
}

async function startStereoWizard() {
  if (cameras.length < 2) return;

  const grid = document.getElementById('cam-grid');
  const wizard = document.createElement('div');
  wizard.className = 'stereo-wizard';

  const title = document.createElement('h3');
  title.textContent = 'Stereo-Kalibrierung';
  wizard.appendChild(title);

  const instructions = document.createElement('p');
  instructions.textContent =
    'Halte das ChArUco-Board so, dass es von beiden Kameras sichtbar ist. ' +
    'Nimm mindestens 15 Aufnahmen aus verschiedenen Winkeln auf.';
  wizard.appendChild(instructions);

  const progress = document.createElement('div');
  progress.className = 'wizard-progress';
  progress.id = 'stereo-progress';
  progress.textContent = 'Aufnahmen: 0 / 15';
  wizard.appendChild(progress);

  const btnCapture = document.createElement('button');
  btnCapture.className = 'btn btn-primary';
  btnCapture.textContent = 'Aufnahme';
  btnCapture.addEventListener('click', () => captureStereoPair(progress));
  wizard.appendChild(btnCapture);

  const btnFinish = document.createElement('button');
  btnFinish.className = 'btn btn-primary';
  btnFinish.textContent = 'Kalibrieren';
  btnFinish.disabled = true;
  btnFinish.id = 'btn-finish-stereo';
  btnFinish.addEventListener('click', finishStereoCalibration);
  wizard.appendChild(btnFinish);

  const btnCancel = document.createElement('button');
  btnCancel.className = 'btn btn-danger';
  btnCancel.textContent = 'Abbrechen';
  btnCancel.addEventListener('click', () => {
    wizard.remove();
  });
  wizard.appendChild(btnCancel);

  grid.parentNode.insertBefore(wizard, grid.nextSibling);
}

let stereoFrameCount = 0;

async function captureStereoPair(progressEl) {
  stereoFrameCount++;
  progressEl.textContent = `Aufnahmen: ${stereoFrameCount} / 15`;

  const btnFinish = document.getElementById('btn-finish-stereo');
  if (btnFinish && stereoFrameCount >= 15) {
    btnFinish.disabled = false;
  }
}

async function finishStereoCalibration() {
  const btnFinish = document.getElementById('btn-finish-stereo');
  if (btnFinish) {
    btnFinish.textContent = 'Kalibriere...';
    btnFinish.disabled = true;
  }

  try {
    const resp = await fetch(`${API_BASE}/calibrate/stereo`, { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const result = await resp.json();

    const wizard = document.querySelector('.stereo-wizard');
    if (wizard) {
      wizard.replaceChildren();
      const done = document.createElement('p');
      done.className = 'hit-green';
      done.textContent = `Stereo-Kalibrierung abgeschlossen! Reprojection Error: ${result.reprojection_error?.toFixed(4) ?? 'N/A'}`;
      wizard.appendChild(done);
    }
  } catch (err) {
    if (btnFinish) {
      btnFinish.textContent = `Fehler: ${err.message}`;
    }
  }
}

export function destroyMulticamPage() {
  if (diagnosticsInterval) {
    clearInterval(diagnosticsInterval);
    diagnosticsInterval = null;
  }
}
