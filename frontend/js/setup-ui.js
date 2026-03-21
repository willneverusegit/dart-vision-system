/**
 * Setup page — Profile management UI.
 * Loads, saves, and deletes CameraProfiles via REST API.
 */

const API = '/api/profiles';

const els = {};

function getEls() {
  els.select = document.getElementById('profile-select');
  els.loadBtn = document.getElementById('btn-load-profile');
  els.deleteBtn = document.getElementById('btn-delete-profile');
  els.nameInput = document.getElementById('profile-name');
  els.saveBtn = document.getElementById('btn-save-profile');
  els.status = document.getElementById('profile-status');
}

function showStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.className = isError ? 'status-msg error' : 'status-msg success';
  setTimeout(() => { els.status.textContent = ''; }, 3000);
}

async function refreshProfiles() {
  try {
    const resp = await fetch(API);
    const profiles = await resp.json();

    // Clear and rebuild dropdown
    while (els.select.firstChild) els.select.removeChild(els.select.firstChild);

    if (profiles.length === 0) {
      const opt = document.createElement('option');
      opt.textContent = '— Keine Profile —';
      opt.disabled = true;
      opt.selected = true;
      els.select.appendChild(opt);
      els.loadBtn.disabled = true;
      els.deleteBtn.disabled = true;
      return;
    }

    els.loadBtn.disabled = false;
    els.deleteBtn.disabled = false;

    for (const p of profiles) {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.id + (p.active ? ' (aktiv)' : '');
      if (p.active) opt.selected = true;
      els.select.appendChild(opt);
    }
  } catch {
    showStatus('Fehler beim Laden der Profile', true);
  }
}

async function loadProfile() {
  const id = els.select.value;
  if (!id) return;
  try {
    const resp = await fetch(`${API}/${id}`);
    if (!resp.ok) throw new Error();
    const profile = await resp.json();
    showStatus(`Profil "${profile.id}" geladen (${profile.role})`);
  } catch {
    showStatus('Profil konnte nicht geladen werden', true);
  }
}

async function saveProfile() {
  const name = els.nameInput.value.trim();
  if (!name) {
    showStatus('Bitte Profilnamen eingeben', true);
    return;
  }
  const profile = {
    id: name,
    role: 'left',
    resolution: [1920, 1080],
  };
  try {
    const resp = await fetch(API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profile),
    });
    if (!resp.ok) throw new Error();
    els.nameInput.value = '';
    showStatus(`Profil "${name}" gespeichert`);
    await refreshProfiles();
  } catch {
    showStatus('Speichern fehlgeschlagen', true);
  }
}

async function deleteProfile() {
  const id = els.select.value;
  if (!id) return;
  try {
    const resp = await fetch(`${API}/${id}`, { method: 'DELETE' });
    if (!resp.ok) throw new Error();
    showStatus(`Profil "${id}" gelöscht`);
    await refreshProfiles();
  } catch {
    showStatus('Löschen fehlgeschlagen', true);
  }
}

export function initSetupPage() {
  getEls();
  els.loadBtn.addEventListener('click', loadProfile);
  els.saveBtn.addEventListener('click', saveProfile);
  els.deleteBtn.addEventListener('click', deleteProfile);
  refreshProfiles();
}
