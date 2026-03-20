import { initGamePage } from './game-ui.js';

const AppState = {
  currentPage: 'game',
};

function navigate(page) {
  AppState.currentPage = page;
  document.querySelectorAll('.page').forEach((p) => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach((l) => l.classList.remove('active'));

  const pageEl = document.getElementById(`page-${page}`);
  const linkEl = document.querySelector(`[data-page="${page}"]`);
  if (pageEl) pageEl.classList.add('active');
  if (linkEl) linkEl.classList.add('active');
}

function handleHash() {
  const hash = location.hash.slice(1) || 'game';
  navigate(hash);
}

// Init
window.addEventListener('hashchange', handleHash);
window.addEventListener('DOMContentLoaded', () => {
  handleHash();
  initGamePage();
});
