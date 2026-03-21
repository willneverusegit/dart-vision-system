const SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5];
const SECTOR_ANGLE = (2 * Math.PI) / 20;

// Ring radii as fraction of canvas radius
const RINGS = {
  bull: 0.037,
  outerBull: 0.094,
  tripleInner: 0.582,
  tripleOuter: 0.629,
  doubleInner: 0.953,
  doubleOuter: 1.0,
};

const COLORS = {
  black: '#1a1a1a',
  white: '#f5f0e1',
  red: '#e94560',
  green: '#2d6a4f',
};

export function drawBoard(canvas) {
  const ctx = canvas.getContext('2d');
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const r = Math.min(cx, cy) * 0.92;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw sectors
  for (let i = 0; i < 20; i++) {
    const startAngle = -Math.PI / 2 + i * SECTOR_ANGLE - SECTOR_ANGLE / 2;
    const endAngle = startAngle + SECTOR_ANGLE;
    const isEven = i % 2 === 0;

    // Outer single (between triple and double)
    drawSectorRing(ctx, cx, cy, r, RINGS.tripleOuter, RINGS.doubleInner, startAngle, endAngle,
      isEven ? COLORS.black : COLORS.white);

    // Double ring
    drawSectorRing(ctx, cx, cy, r, RINGS.doubleInner, RINGS.doubleOuter, startAngle, endAngle,
      isEven ? COLORS.red : COLORS.green);

    // Inner single (between outer bull and triple)
    drawSectorRing(ctx, cx, cy, r, RINGS.outerBull, RINGS.tripleInner, startAngle, endAngle,
      isEven ? COLORS.black : COLORS.white);

    // Triple ring
    drawSectorRing(ctx, cx, cy, r, RINGS.tripleInner, RINGS.tripleOuter, startAngle, endAngle,
      isEven ? COLORS.red : COLORS.green);
  }

  // Bull rings
  drawCircle(ctx, cx, cy, r * RINGS.outerBull, COLORS.green);
  drawCircle(ctx, cx, cy, r * RINGS.bull, COLORS.red);

  // Sector numbers
  ctx.fillStyle = '#ccc';
  ctx.font = `bold ${r * 0.07}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < 20; i++) {
    const angle = -Math.PI / 2 + i * SECTOR_ANGLE;
    const numR = r * 1.08;
    const nx = cx + numR * Math.cos(angle);
    const ny = cy + numR * Math.sin(angle);
    ctx.fillText(SECTORS[i].toString(), nx, ny);
  }

  // Wire lines
  ctx.strokeStyle = '#666';
  ctx.lineWidth = 0.5;
  for (let i = 0; i < 20; i++) {
    const angle = -Math.PI / 2 + i * SECTOR_ANGLE - SECTOR_ANGLE / 2;
    ctx.beginPath();
    ctx.moveTo(cx + r * RINGS.bull * Math.cos(angle), cy + r * RINGS.bull * Math.sin(angle));
    ctx.lineTo(cx + r * RINGS.doubleOuter * Math.cos(angle), cy + r * RINGS.doubleOuter * Math.sin(angle));
    ctx.stroke();
  }
}

function drawSectorRing(ctx, cx, cy, r, innerFrac, outerFrac, startAngle, endAngle, color) {
  ctx.beginPath();
  ctx.arc(cx, cy, r * outerFrac, startAngle, endAngle);
  ctx.arc(cx, cy, r * innerFrac, endAngle, startAngle, true);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function drawCircle(ctx, cx, cy, radius, color) {
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

export function getScoreFromClick(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (event.clientX - rect.left) * scaleX;
  const py = (event.clientY - rect.top) * scaleY;

  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const r = Math.min(cx, cy) * 0.92;

  const dx = px - cx;
  const dy = py - cy;
  const dist = Math.sqrt(dx * dx + dy * dy) / r;
  let angle = Math.atan2(dy, dx) + Math.PI / 2;
  if (angle < 0) angle += 2 * Math.PI;

  // Miss
  if (dist > RINGS.doubleOuter) return { score: 0, field: 'MISS', multiplier: 0 };

  // Bull
  if (dist <= RINGS.bull) return { score: 50, field: 'DBULL', multiplier: 2 };
  if (dist <= RINGS.outerBull) return { score: 25, field: 'BULL', multiplier: 1 };

  // Sector
  const sectorIdx = Math.floor(((angle + SECTOR_ANGLE / 2) % (2 * Math.PI)) / SECTOR_ANGLE);
  const sectorValue = SECTORS[sectorIdx];

  // Ring
  if (dist >= RINGS.doubleInner) return { score: sectorValue * 2, field: `D${sectorValue}`, multiplier: 2 };
  if (dist >= RINGS.tripleInner && dist <= RINGS.tripleOuter) return { score: sectorValue * 3, field: `T${sectorValue}`, multiplier: 3 };

  return { score: sectorValue, field: `${sectorValue}`, multiplier: 1 };
}

function getConfidenceColor(confidence) {
  if (confidence >= 0.8) return '#4ecca3';
  if (confidence >= 0.5) return '#f5c518';
  return '#e94560';
}

export function drawHit(canvas, event, field, confidence = 1.0) {
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (event.clientX - rect.left) * scaleX;
  const py = (event.clientY - rect.top) * scaleY;

  ctx.beginPath();
  ctx.arc(px, py, 8, 0, 2 * Math.PI);
  ctx.fillStyle = getConfidenceColor(confidence);
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Label
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 11px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText(field, px, py - 12);
}

export function drawHits(canvas, hits) {
  const ctx = canvas.getContext('2d');
  const AGE_OPACITY = [1.0, 0.6, 0.3];

  for (const hit of hits) {
    const opacity = AGE_OPACITY[hit.age] ?? 0.3;
    ctx.globalAlpha = opacity;

    // Filled circle
    ctx.beginPath();
    ctx.arc(hit.x, hit.y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = getConfidenceColor(hit.confidence);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(hit.field, hit.x, hit.y - 12);
  }

  ctx.globalAlpha = 1.0;
}
