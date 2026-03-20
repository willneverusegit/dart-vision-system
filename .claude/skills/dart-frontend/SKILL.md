---
name: dart-frontend
description: Guidance for web UI implementation (Setup, Calibration, Game pages, WebSocket client, Canvas board)
user_invocable: true
---

# Dart Frontend Skill

## Wann verwenden
Bei Arbeit an der Web-Oberfläche: Seiten, Komponenten, Canvas-Rendering, WebSocket-Client.

## Seitenstruktur

### 1. Setup-Seite (`#setup`)
- DeviceList: Verfügbare Kameras anzeigen (ID, Vorschau, Rolle zuweisen)
- Profilverwaltung: Dropdown mit gespeicherten Profilen, Laden/Speichern
- Button: "Kalibrierung starten"

### 2. Kalibrierungs-Assistent (`#calibration`)
- Schrittanzeige: Lens Cal → ROI → Board Fit → Fine Tune
- CalibrationOverlay: Live-Bild mit Marker-Overlay via Canvas
- RoiCropper: Draggable/resizable Rechteck auf Canvas
- BoardFitter: Ellipsen-/Vierpunkt-Tool
- Fortschrittsbalken + Statusmeldungen

### 3. Multi-Cam-Seite (`#multicam`)
- Extrinsische Kalibrierung starten
- Kamera aktivieren/deaktivieren
- DiagnosticsPanel: Reprojection Error, Marker-Count, FPS

### 4. Game-Seite (`#game`)
- Board-Canvas: Top-Down projiziertes Dartboard (Polar-Koordinaten)
- Treffer-Marker mit Score + Confidence
- PreviewThumbnail: kleines Live-Kamerabild
- Scoreboard: Spieler, Punkte, Wurf-Historie
- ModeToggle: Eco / Normal / Debug

## WebSocket-Client
```javascript
const ws = new WebSocket(`ws://${location.host}/ws/stream`);
ws.binaryType = 'arraybuffer';
ws.onmessage = (event) => {
  if (typeof event.data === 'string') {
    // JSON Event (score, status, calibration)
    const msg = JSON.parse(event.data);
    handleEvent(msg);
  } else {
    // Binary frame (JPEG)
    const blob = new Blob([event.data], { type: 'image/jpeg' });
    previewImg.src = URL.createObjectURL(blob);
  }
};
```

## Canvas Board-Rendering
```javascript
function drawBoard(ctx, centerX, centerY, radius) {
  const sectors = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5];
  const sectorAngle = (2 * Math.PI) / 20;
  // Ringe: Bull (6.35mm), Outer Bull (15.9mm), Triple, Double
  // Sektoren zeichnen mit alternierenden Farben
}
function drawHit(ctx, x, y, score, confidence) {
  // Treffer-Marker mit Farbkodierung nach Confidence
}
```

## State Management (einfach)
```javascript
const AppState = {
  currentPage: 'setup',  // setup | calibration | multicam | game
  mode: 'normal',        // eco | normal | debug
  cameras: [],
  calibration: { step: null, progress: 0 },
  game: { players: [], currentPlayer: 0, throws: [] }
};
```

## Dateien
- `frontend/index.html` — SPA Shell + Hash-Routing
- `frontend/js/app.js` — State + Routing
- `frontend/js/ws-client.js` — WebSocket
- `frontend/js/board-canvas.js` — Canvas Board
- `frontend/js/calibration-ui.js` — Kalibrierungs-Wizard
- `frontend/js/game-ui.js` — Spiel-Ansicht
- `frontend/css/style.css` — Dark Theme

## Design
- Dark Theme (Bar/Pub-Umgebung)
- Responsive, funktioniert auf Tablet
- Große Touch-Targets für Spielbetrieb
