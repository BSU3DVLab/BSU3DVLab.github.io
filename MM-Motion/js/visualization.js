// Foot Pressure Visualization
let rightSensorPositions = [
  {id: 'R1', x: 728, y: 38}, {id: 'R2', x: 764, y: 21}, {id: 'R3', x: 795, y: 28}, {id: 'R4', x: 822, y: 48},
  {id: 'R5', x: 719, y: 78}, {id: 'R6', x: 763, y: 71}, {id: 'R7', x: 804, y: 80}, {id: 'R8', x: 846, y: 90}, {id: 'R9', x: 853, y: 131},
  {id: 'R10', x: 723, y: 120}, {id: 'R11', x: 768, y: 122}, {id: 'R12', x: 810, y: 126}, {id: 'R13', x: 771, y: 165}, {id: 'R14', x: 862, y: 175},
  {id: 'R15', x: 724, y: 164}, {id: 'R16', x: 785, y: 211}, {id: 'R17', x: 820, y: 213}, {id: 'R18', x: 858, y: 219}, {id: 'R19', x: 816, y: 173},
  {id: 'R20', x: 747, y: 208}, {id: 'R21', x: 753, y: 249}, {id: 'R22', x: 786, y: 253}, {id: 'R23', x: 824, y: 257}, {id: 'R24', x: 861, y: 259},
  {id: 'R25', x: 753, y: 347}, {id: 'R26', x: 762, y: 299}, {id: 'R27', x: 790, y: 302}, {id: 'R28', x: 818, y: 304}, {id: 'R29', x: 849, y: 307},
  {id: 'R30', x: 754, y: 394}, {id: 'R31', x: 784, y: 399}, {id: 'R32', x: 785, y: 350}, {id: 'R33', x: 815, y: 350}, {id: 'R34', x: 849, y: 354},
  {id: 'R35', x: 751, y: 439}, {id: 'R36', x: 814, y: 403}, {id: 'R37', x: 781, y: 445}, {id: 'R38', x: 811, y: 449}, {id: 'R39', x: 843, y: 402},
  {id: 'R40', x: 749, y: 483}, {id: 'R41', x: 777, y: 494}, {id: 'R42', x: 807, y: 496}, {id: 'R43', x: 836, y: 492}, {id: 'R44', x: 841, y: 447},
  {id: 'R45', x: 749, y: 526}, {id: 'R46', x: 770, y: 547}, {id: 'R47', x: 804, y: 550}, {id: 'R48', x: 833, y: 534}
];

let leftSensorPositions = [
  {id: 'L1', x: 498, y: 48}, {id: 'L2', x: 530, y: 29}, {id: 'L3', x: 560, y: 18}, {id: 'L4', x: 597, y: 36},
  {id: 'L5', x: 480, y: 91}, {id: 'L6', x: 517, y: 80}, {id: 'L7', x: 562, y: 73}, {id: 'L8', x: 599, y: 78}, {id: 'L9', x: 603, y: 120},
  {id: 'L10', x: 473, y: 131}, {id: 'L11', x: 510, y: 126}, {id: 'L12', x: 558, y: 120}, {id: 'L13', x: 550, y: 165}, {id: 'L14', x: 593, y: 162},
  {id: 'L15', x: 463, y: 215}, {id: 'L16', x: 501, y: 213}, {id: 'L17', x: 540, y: 208}, {id: 'L18', x: 505, y: 170}, {id: 'L19', x: 579, y: 204},
  {id: 'L20', x: 466, y: 175}, {id: 'L21', x: 535, y: 253}, {id: 'L22', x: 464, y: 259}, {id: 'L23', x: 497, y: 256}, {id: 'L24', x: 574, y: 250},
  {id: 'L25', x: 475, y: 354}, {id: 'L26', x: 474, y: 307}, {id: 'L27', x: 511, y: 302}, {id: 'L28', x: 536, y: 300}, {id: 'L29', x: 569, y: 297},
  {id: 'L30', x: 483, y: 402}, {id: 'L31', x: 514, y: 401}, {id: 'L32', x: 509, y: 351}, {id: 'L33', x: 537, y: 346}, {id: 'L34', x: 568, y: 345},
  {id: 'L35', x: 543, y: 398}, {id: 'L36', x: 575, y: 392}, {id: 'L37', x: 575, y: 439}, {id: 'L38', x: 514, y: 448}, {id: 'L39', x: 481, y: 450},
  {id: 'L40', x: 485, y: 491}, {id: 'L41', x: 522, y: 496}, {id: 'L42', x: 553, y: 491}, {id: 'L43', x: 581, y: 482}, {id: 'L44', x: 547, y: 445},
  {id: 'L45', x: 492, y: 533}, {id: 'L46', x: 521, y: 548}, {id: 'L47', x: 557, y: 545}, {id: 'L48', x: 580, y: 525}
];

const rightFootPerimeter = ['R1','R2','R3','R4','R8','R9','R14','R18','R24','R29','R34','R39','R44','R48','R47','R46','R45','R40','R35','R30','R25','R21','R20','R15','R10','R5'];
const leftFootPerimeter = ['L4','L3','L2','L1','L5','L10','L20','L15','L22','L26','L25','L30','L39','L40','L45','L46','L47','L48','L43','L37','L36','L34','L29','L24','L19','L14','L9','L8'];

let pressureData = [];
let footCurrentFrame = 0;
let sensorsVisible = true;
let showCoP = true;
let heatmapCanvas = null;
let heatmapCtx = null;
let maskCanvas = null;
let maskCtx = null;
let tempCanvas = null;
let tempCtx = null;

const COLOR_STOPS = [
  { v: 0.0, c: [41, 51, 117] },
  { v: 0.2, c: [84, 158, 191] },
  { v: 0.4, c: [132, 193, 115] },
  { v: 0.6, c: [222, 215, 86] },
  { v: 0.8, c: [216, 117, 50] },
  { v: 1.0, c: [159, 39, 36] }
];

function interpolateColor(intensity) {
  intensity = Math.max(0, Math.min(1, intensity));
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    if (intensity >= COLOR_STOPS[i].v && intensity <= COLOR_STOPS[i+1].v) {
      const range = COLOR_STOPS[i+1].v - COLOR_STOPS[i].v;
      const ratio = (intensity - COLOR_STOPS[i].v) / range;
      const r = Math.round(COLOR_STOPS[i].c[0] + ratio * (COLOR_STOPS[i+1].c[0] - COLOR_STOPS[i].c[0]));
      const g = Math.round(COLOR_STOPS[i].c[1] + ratio * (COLOR_STOPS[i+1].c[1] - COLOR_STOPS[i].c[1]));
      const b = Math.round(COLOR_STOPS[i].c[2] + ratio * (COLOR_STOPS[i+1].c[2] - COLOR_STOPS[i].c[2]));
      return { r, g, b };
    }
  }
  return { r: 255, g: 0, b: 0 };
}

function getHeatmapColor(intensity) {
  return interpolateColor(intensity);
}

function initFootCanvas() {
  heatmapCanvas = document.getElementById('heatmap-canvas');
  const container = document.getElementById('foot-container');
  const w = container.offsetWidth;
  const h = container.offsetHeight;
  
  heatmapCanvas.width = w;
  heatmapCanvas.height = h;
  heatmapCtx = heatmapCanvas.getContext('2d');
  
  if (!tempCanvas) {
    tempCanvas = document.createElement('canvas');
    tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
  }
  tempCanvas.width = w;
  tempCanvas.height = h;
  
  if (!maskCanvas) {
    maskCanvas = document.createElement('canvas');
    maskCtx = maskCanvas.getContext('2d');
  }
  maskCanvas.width = w;
  maskCanvas.height = h;
  
  // 计算缩放比例和偏移量
  const originalWidth = 1350;
  const originalHeight = 600;
  const scaleX = w / originalWidth;
  const scaleY = h / originalHeight;
  const scale = Math.min(scaleX, scaleY) * 0.85;
  const offsetX = (w - originalWidth * scale) / 2;
  const offsetY = (h - originalHeight * scale) / 2;
  
  // 缩放传感器位置
  rightSensorPositions = rightSensorPositions.map(p => ({
    id: p.id,
    x: p.x * scale + offsetX,
    y: p.y * scale + offsetY,
    originalX: p.x,
    originalY: p.y
  }));
  
  leftSensorPositions = leftSensorPositions.map(p => ({
    id: p.id,
    x: p.x * scale + offsetX,
    y: p.y * scale + offsetY,
    originalX: p.x,
    originalY: p.y
  }));
  
  // 存储缩放参数供热图绘制使用
  window.footScale = scale;
  window.footOffsetX = offsetX;
  window.footOffsetY = offsetY;
}

function defineSmoothPath(ctx, points) {
  if (points.length < 3) return;
  ctx.beginPath();
  let startX = (points[points.length - 1].x + points[0].x) / 2;
  let startY = (points[points.length - 1].y + points[0].y) / 2;
  ctx.moveTo(startX, startY);

  for (let i = 0; i < points.length; i++) {
    let p1 = points[i];
    let p2 = points[(i + 1) % points.length];
    let midX = (p1.x + p2.x) / 2;
    let midY = (p1.y + p2.y) / 2;
    ctx.quadraticCurveTo(p1.x, p1.y, midX, midY);
  }
  ctx.closePath();
}

function drawFootBase(ctx, points, scale = 1) {
  if (points.length === 0) return;
  defineSmoothPath(ctx, points);
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  
  ctx.lineWidth = 55 * scale; 
  ctx.strokeStyle = '#cccccc';
  ctx.stroke();
  
  ctx.lineWidth = 51 * scale;
  ctx.strokeStyle = '#f4f6f8';
  ctx.stroke();
  ctx.fillStyle = '#f4f6f8';
  ctx.fill();
}

function drawFootMask(ctx, points, scale = 1) {
  if (points.length === 0) return;
  defineSmoothPath(ctx, points);
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  
  ctx.lineWidth = 51 * scale;
  ctx.strokeStyle = '#000000';
  ctx.stroke();
  ctx.fillStyle = '#000000';
  ctx.fill();
}

function calculateCoP(dataArray, side) {
  let sumPx = 0;
  let sumPy = 0;
  let sumP = 0;
  const positions = side === 'right' ? rightSensorPositions : leftSensorPositions;
  const prefix = side === 'right' ? 'R' : 'L';

  if (!dataArray || dataArray.length === 0) return null;

  dataArray.forEach((val, idx) => {
    if (val > 5) { 
      const pos = positions.find(p => p.id === `${prefix}${idx + 1}`);
      if (pos) {
        sumPx += val * (pos.x + 17.5);
        sumPy += val * (pos.y + 17.5);
        sumP += val;
      }
    }
  });

  if (sumP < 10) return null; 
  return { x: sumPx / sumP, y: sumPy / sumP };
}

function drawArrow(ctx, p1, p2) {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const angle = Math.atan2(dy, dx);
  const midX = (p1.x + p2.x) / 2;
  const midY = (p1.y + p2.y) / 2;
  const headlen = 8; 
  
  ctx.beginPath();
  ctx.moveTo(midX, midY); 
  ctx.lineTo(midX - headlen * Math.cos(angle - Math.PI / 7), midY - headlen * Math.sin(angle - Math.PI / 7));
  ctx.lineTo(midX - headlen * Math.cos(angle + Math.PI / 7), midY - headlen * Math.sin(angle + Math.PI / 7));
  ctx.closePath();
  
  ctx.fillStyle = 'rgba(128, 128, 128, 0.9)'; 
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
  ctx.stroke();
}

function drawCoPTrajectory(ctx, upToFrame, side, lineColor) {
  if (pressureData.length === 0) return;

  const path = [];
  const maxSafeFrame = Math.min(upToFrame, pressureData.length - 1);
  
  for (let i = 0; i <= maxSafeFrame; i++) {
    const frame = pressureData[i];
    if (!frame) continue;
    
    const dataArray = side === 'right' ? (frame.right || []) : (frame.left || []);
    const cop = calculateCoP(dataArray, side);
    if (cop) {
      path.push(cop);
    }
  }

  if (path.length === 0) return;

  ctx.beginPath();
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round';
  ctx.moveTo(path[0].x, path[0].y);
  for (let i = 1; i < path.length; i++) {
    ctx.lineTo(path[i].x, path[i].y);
  }
  ctx.stroke();

  let lastArrowPoint = path[0];
  for (let i = 0; i < path.length; i++) {
    ctx.beginPath();
    ctx.fillStyle = lineColor;
    ctx.arc(path[i].x, path[i].y, 2.5, 0, Math.PI * 2);
    ctx.fill();

    if (i > 0) {
      const p1 = path[i - 1];
      const p2 = path[i];
      const dist = Math.hypot(p2.x - lastArrowPoint.x, p2.y - lastArrowPoint.y);
      if (dist > 12) { 
        drawArrow(ctx, lastArrowPoint, p2);
        lastArrowPoint = p2;
      }
    }
  }

  ctx.fillStyle = '#4CAF50';
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 1.5;
  ctx.fillRect(path[0].x - 6, path[0].y - 6, 12, 12);
  ctx.strokeRect(path[0].x - 6, path[0].y - 6, 12, 12);

  const last = path[path.length - 1];
  ctx.beginPath();
  ctx.fillStyle = lineColor;
  ctx.arc(last.x, last.y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawHeatmap(frame) {
  if (!heatmapCtx) return;
  const width = heatmapCanvas.width;
  const height = heatmapCanvas.height;
  
  heatmapCtx.clearRect(0, 0, width, height);
  tempCtx.clearRect(0, 0, width, height);
  maskCtx.clearRect(0, 0, width, height);
  
  const scale = window.footScale || 1;
  const rightPts = rightFootPerimeter.map(id => rightSensorPositions.find(p => p.id === id)).filter(p => p).map(p => ({x: p.x + 17.5 * scale, y: p.y + 17.5 * scale}));
  const leftPts = leftFootPerimeter.map(id => leftSensorPositions.find(p => p.id === id)).filter(p => p).map(p => ({x: p.x + 17.5 * scale, y: p.y + 17.5 * scale}));

  drawFootBase(heatmapCtx, rightPts, scale);
  drawFootBase(heatmapCtx, leftPts, scale);

  drawFootMask(maskCtx, rightPts, scale);
  drawFootMask(maskCtx, leftPts, scale);
  
  const rightData = (frame && frame.right) ? frame.right : [];
  rightData.forEach((value, index) => {
    const sensorId = index + 1;
    const pos = rightSensorPositions.find(p => p.id === `R${sensorId}`);
    if (pos && value > 0) {
      drawHeatPoint(tempCtx, pos.x + 17.5 * scale, pos.y + 17.5 * scale, value, 80 * scale);
    }
  });
  
  const leftData = (frame && frame.left) ? frame.left : [];
  leftData.forEach((value, index) => {
    const sensorId = index + 1;
    const pos = leftSensorPositions.find(p => p.id === `L${sensorId}`);
    if (pos && value > 0) {
      drawHeatPoint(tempCtx, pos.x + 17.5 * scale, pos.y + 17.5 * scale, value, 80 * scale);
    }
  });
  
  const imageData = tempCtx.getImageData(0, 0, width, height);
  const data = imageData.data;
  
  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3];
    if (alpha > 0) {
      const intensity = alpha / 255;
      const color = getHeatmapColor(intensity);
      data[i] = color.r;
      data[i + 1] = color.g;
      data[i + 2] = color.b;
      data[i + 3] = Math.min(alpha * 2.5, 255);
    }
  }
  tempCtx.putImageData(imageData, 0, 0);
  
  tempCtx.globalCompositeOperation = 'destination-in';
  tempCtx.drawImage(maskCanvas, 0, 0);
  tempCtx.globalCompositeOperation = 'source-over';
  
  heatmapCtx.drawImage(tempCanvas, 0, 0);

  if (showCoP && pressureData.length > 0) {
    drawCoPTrajectory(heatmapCtx, footCurrentFrame, 'left', '#2196F3'); 
    drawCoPTrajectory(heatmapCtx, footCurrentFrame, 'right', '#F44336'); 
  }
}
    
function drawHeatPoint(ctx, x, y, value, radius) {
  const maxValue = 2000;
  const intensity = Math.min(value / maxValue, 1);
  const actualRadius = radius * (0.8 + intensity * 0.6);
  
  const gradient = ctx.createRadialGradient(x, y, 0, x, y, actualRadius);
  gradient.addColorStop(0, `rgba(255, 255, 255, ${intensity})`);
  gradient.addColorStop(0.3, `rgba(255, 255, 255, ${intensity * 0.8})`);
  gradient.addColorStop(0.7, `rgba(255, 255, 255, ${intensity * 0.3})`);
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
  
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(x, y, actualRadius, 0, Math.PI * 2);
  ctx.fill();
}

function loadFootCSVData(csvContent) {
  const lines = csvContent.trim().split('\n');
  pressureData = [];
  const headers = lines[0].split(',').map(h => h.trim());
  
  const rIndices = [];
  const lIndices = [];
  
  for (let i = 0; i < headers.length; i++) {
    const header = headers[i];
    if (header.match(/^R\d+\(g\)$/)) {
      rIndices.push(i);
    } else if (header.match(/^L\d+\(g\)$/)) {
      lIndices.push(i);
    }
  }

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',');
    if (values.length >= Math.max(...rIndices, ...lIndices) + 1) {
      const frameData = {
        time: values[0] || 0,
        right: rIndices.map(idx => parseFloat(values[idx]) || 0),
        left: lIndices.map(idx => parseFloat(values[idx]) || 0)
      };
      pressureData.push(frameData);
    }
  }

  const slider = document.getElementById('time-slider');
  slider.max = pressureData.length > 0 ? pressureData.length - 1 : 0;
  slider.value = 0;

  // Initialize sensor DOM elements
  initFootSensorElements();
  
  updateFootVisualization(0);
  
  // Set global flag
  if (typeof footDataLoaded !== 'undefined') {
    footDataLoaded = true;
    if (typeof checkAllDataLoaded === 'function') {
      checkAllDataLoaded();
    }
  }
}

function updateFootVisualization(frameIndex) {
  if (frameIndex < 0 || frameIndex >= pressureData.length) return;
  
  footCurrentFrame = frameIndex; 
  
  const frame = pressureData[footCurrentFrame];
  drawHeatmap(frame);

  // 更新传感器UI
  frame.right.forEach((value, index) => updateFootSensorUI('right-R', index, value));
  frame.left.forEach((value, index) => updateFootSensorUI('left-L', index, value));

  document.getElementById('time-display').textContent = `时间: ${frame.time}s`;
  document.getElementById('time-slider').value = footCurrentFrame;
}

function toggleFootSensors() {
  sensorsVisible = !sensorsVisible;
  const container = document.getElementById('foot-container');
  const btn = document.getElementById('toggle-sensors-btn');
  
  if (sensorsVisible) {
    container.classList.remove('hide-sensors');
    btn.textContent = '隐藏数据框';
    btn.style.backgroundColor = '#3f51b5';
  } else {
    container.classList.add('hide-sensors');
    btn.textContent = '显示数据框';
    btn.style.backgroundColor = '#795548';
  }
}

function toggleFootCoP() {
  showCoP = !showCoP;
  const btn = document.getElementById('toggle-cop-btn');
  if (showCoP) {
    btn.textContent = '隐藏压力中心';
    btn.style.backgroundColor = '#009688';
  } else {
    btn.textContent = '显示压力中心';
    btn.style.backgroundColor = '#607D8B';
  }
  if (pressureData.length > 0) {
    updateFootVisualization(footCurrentFrame); 
  }
}

// 创建传感器DOM元素
function initFootSensorElements() {
  const container = document.getElementById('foot-container');
  const canvas = document.getElementById('heatmap-canvas');
  
  // 清除旧的传感器元素但保留canvas
  const oldSensors = container.querySelectorAll('.sensor');
  oldSensors.forEach(s => s.remove());
  
  // 创建传感器元素
  rightSensorPositions.forEach(pos => createFootSensorElement(container, pos, 'right'));
  leftSensorPositions.forEach(pos => createFootSensorElement(container, pos, 'left'));
}

function createFootSensorElement(container, pos, side) {
  const sensor = document.createElement('div');
  sensor.className = 'sensor';
  sensor.id = `${side}-${pos.id}`;
  sensor.style.left = `${pos.x}px`;
  sensor.style.top = `${pos.y}px`;
  sensor.textContent = '0';
  sensor.dataset.side = side;
  sensor.dataset.id = pos.id;
  container.appendChild(sensor);
}

// 更新传感器UI
function updateFootSensorUI(sensorIdPrefix, index, value) {
  const sensorId = index + 1;
  const sensor = document.getElementById(`${sensorIdPrefix}${sensorId}`);
  if (sensor) {
    sensor.textContent = Math.round(value);
    
    if (value < 5) {
      sensor.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
      sensor.style.color = '#666';
      sensor.style.border = '1px solid #ddd';
    } else {
      const color = interpolateColor(value / 2000);
      sensor.style.backgroundColor = `rgba(${color.r}, ${color.g}, ${color.b}, 0.85)`;
      
      const brightness = (color.r * 299 + color.g * 587 + color.b * 114) / 1000;
      sensor.style.color = brightness > 125 ? '#000' : '#fff';
      sensor.style.border = '1px solid rgba(0,0,0,0.2)';
    }
  }
}

// 在DOM加载完成后添加事件监听器
function initFootEventListeners() {
  const toggleSensorsBtn = document.getElementById('toggle-sensors-btn');
  const toggleCopBtn = document.getElementById('toggle-cop-btn');
  const timeSlider = document.getElementById('time-slider');
  
  if (toggleSensorsBtn) {
    toggleSensorsBtn.addEventListener('click', toggleFootSensors);
  }
  if (toggleCopBtn) {
    toggleCopBtn.addEventListener('click', toggleFootCoP);
  }
  if (timeSlider) {
    timeSlider.addEventListener('input', (e) => {
      updateFootVisualization(parseInt(e.target.value));
    });
  }
}

window.addEventListener('load', () => {
  initFootCanvas();
  initFootEventListeners();
  fetch('data-demo/1701F2.csv')
    .then(response => response.text())
    .then(data => loadFootCSVData(data))
    .catch(err => console.log('Default foot pressure file loading failed', err));
  
  // Initialize IMU visualization
  initScene_IMU();
  fetch('data-demo/1701K2Mskel.csv')
    .then(response => response.text())
    .then(data => {
      parseCSV_IMU(data);
      performAnalysis_IMU();
      imuDataLoaded = true;
      checkAllDataLoaded();
    })
    .catch(err => console.log('Default IMU file loading failed', err));
  
  // Initialize Kinect visualization
  initThree_Kinect();
  initCharts_Kinect();
  fetch('data-demo/1701K2Mskel.csv')
    .then(response => response.text())
    .then(data => {
      loadCSVData_Kinect(data);
      kinectDataLoaded = true;
      checkAllDataLoaded();
    })
    .catch(err => console.log('Default Kinect file loading failed', err));
});

// ============================================
// IMU Data Visualization
// ============================================

const FPS_IMU = 90;
const frameInterval_IMU = 1000 / FPS_IMU;
const SKELETON_COLOR_IMU = 0x00e5ff; 
const TRAJECTORY_COLOR_LEFT_IMU = 0x10b981;
const TRAJECTORY_COLOR_RIGHT_IMU = 0xf43f5e;
const JOINT_RADIUS_IMU = 0.018; 
const BONE_RADIUS_IMU = 0.0045; 
const AXES_HELPER_SCALE_IMU = 0.6;

let charts_IMU = { velocity: null, tilt: null, rose: null };
let trajectoryLines_IMU = { left: null, right: null };

const skeletonDefinition_IMU = [
  ["Hips", "Hips", 0, -1, ["Hips-Joint-Posi-x", "Hips-Joint-Posi-y", "Hips-Joint-Posi-z"]],
  ["RightUpLeg", "RightUpLeg", 1, 0, ["RightUpLeg-Joint-Posi-x", "RightUpLeg-Joint-Posi-y", "RightUpLeg-Joint-Posi-z"]],
  ["RightLeg", "RightLeg", 2, 1, ["RightLeg-Joint-Posi-x", "RightLeg-Joint-Posi-y", "RightLeg-Joint-Posi-z"]],
  ["RightFoot", "RightFoot", 3, 2, ["RightFoot-Joint-Posi-x", "RightFoot-Joint-Posi-y", "RightFoot-Joint-Posi-z"]],
  ["LeftUpLeg", "LeftUpLeg", 4, 0, ["LeftUpLeg-Joint-Posi-x", "LeftUpLeg-Joint-Posi-y", "LeftUpLeg-Joint-Posi-z"]],
  ["LeftLeg", "LeftLeg", 5, 4, ["LeftLeg-Joint-Posi-x", "LeftLeg-Joint-Posi-y", "LeftLeg-Joint-Posi-z"]],
  ["LeftFoot", "LeftFoot", 6, 5, ["LeftFoot-Joint-Posi-x", "LeftFoot-Joint-Posi-y", "LeftFoot-Joint-Posi-z"]],
  ["Spine", "Spine", 7, 0, ["Spine-Joint-Posi-x", "Spine-Joint-Posi-y", "Spine-Joint-Posi-z"]],
  ["Spine1", "Spine1", 8, 7, ["Spine1-Joint-Posi-x", "Spine1-Joint-Posi-y", "Spine1-Joint-Posi-z"]],
  ["Spine2", "Spine2", 9, 8, ["Spine2-Joint-Posi-x", "Spine2-Joint-Posi-y", "Spine2-Joint-Posi-z"]],
  ["Neck", "Neck", 10, 9, ["Neck-Joint-Posi-x", "Neck-Joint-Posi-y", "Neck-Joint-Posi-z"]],
  ["Neck1", "Neck1", 11, 10, ["Neck1-Joint-Posi-x", "Neck1-Joint-Posi-y", "Neck1-Joint-Posi-z"]],
  ["Head", "Head", 12, 11, ["Head-Joint-Posi-x", "Head-Joint-Posi-y", "Head-Joint-Posi-z"]],
  ["RightShoulder", "RightShoulder", 13, 8, ["RightShoulder-Joint-Posi-x", "RightShoulder-Joint-Posi-y", "RightShoulder-Joint-Posi-z"]],
  ["RightArm", "RightArm", 14, 13, ["RightArm-Joint-Posi-x", "RightArm-Joint-Posi-y", "RightArm-Joint-Posi-z"]],
  ["RightForeArm", "RightForeArm", 15, 14, ["RightForeArm-Joint-Posi-x", "RightForeArm-Joint-Posi-y", "RightForeArm-Joint-Posi-z"]],
  ["RightHand", "RightHand", 16, 15, ["RightHand-Joint-Posi-x", "RightHand-Joint-Posi-y", "RightHand-Joint-Posi-z"]],
  ["RightHandThumb1", "RightHandThumb1", 17, 16, ["RightHandThumb1-Joint-Posi-x", "RightHandThumb1-Joint-Posi-y", "RightHandThumb1-Joint-Posi-z"]],
  ["RightHandThumb2", "RightHandThumb2", 18, 17, ["RightHandThumb2-Joint-Posi-x", "RightHandThumb2-Joint-Posi-y", "RightHandThumb2-Joint-Posi-z"]],
  ["RightHandThumb3", "RightHandThumb3", 19, 18, ["RightHandThumb3-Joint-Posi-x", "RightHandThumb3-Joint-Posi-y", "RightHandThumb3-Joint-Posi-z"]],
  ["RightInHandIndex", "RightInHandIndex", 20, 16, ["RightInHandIndex-Joint-Posi-x", "RightInHandIndex-Joint-Posi-y", "RightInHandIndex-Joint-Posi-z"]],
  ["RightHandIndex1", "RightHandIndex1", 21, 20, ["RightHandIndex1-Joint-Posi-x", "RightHandIndex1-Joint-Posi-y", "RightHandIndex1-Joint-Posi-z"]],
  ["RightHandIndex2", "RightHandIndex2", 22, 21, ["RightHandIndex2-Joint-Posi-x", "RightHandIndex2-Joint-Posi-y", "RightHandIndex2-Joint-Posi-z"]],
  ["RightHandIndex3", "RightHandIndex3", 23, 22, ["RightHandIndex3-Joint-Posi-x", "RightHandIndex3-Joint-Posi-y", "RightHandIndex3-Joint-Posi-z"]],
  ["RightInHandMiddle", "RightInHandMiddle", 24, 16, ["RightInHandMiddle-Joint-Posi-x", "RightInHandMiddle-Joint-Posi-y", "RightInHandMiddle-Joint-Posi-z"]],
  ["RightHandMiddle1", "RightHandMiddle1", 25, 24, ["RightHandMiddle1-Joint-Posi-x", "RightHandMiddle1-Joint-Posi-y", "RightHandMiddle1-Joint-Posi-z"]],
  ["RightHandMiddle2", "RightHandMiddle2", 26, 25, ["RightHandMiddle2-Joint-Posi-x", "RightHandMiddle2-Joint-Posi-y", "RightHandMiddle2-Joint-Posi-z"]],
  ["RightHandMiddle3", "RightHandMiddle3", 27, 26, ["RightHandMiddle3-Joint-Posi-x", "RightHandMiddle3-Joint-Posi-y", "RightHandMiddle3-Joint-Posi-z"]],
  ["RightInHandRing", "RightInHandRing", 28, 16, ["RightInHandRing-Joint-Posi-x", "RightInHandRing-Joint-Posi-y", "RightInHandRing-Joint-Posi-z"]],
  ["RightHandRing1", "RightHandRing1", 29, 28, ["RightHandRing1-Joint-Posi-x", "RightHandRing1-Joint-Posi-y", "RightHandRing1-Joint-Posi-z"]],
  ["RightHandRing2", "RightHandRing2", 30, 29, ["RightHandRing2-Joint-Posi-x", "RightHandRing2-Joint-Posi-y", "RightHandRing2-Joint-Posi-z"]],
  ["RightHandRing3", "RightHandRing3", 31, 30, ["RightHandRing3-Joint-Posi-x", "RightHandRing3-Joint-Posi-y", "RightHandRing3-Joint-Posi-z"]],
  ["RightInHandPinky", "RightInHandPinky", 32, 16, ["RightInHandPinky-Joint-Posi-x", "RightInHandPinky-Joint-Posi-y", "RightInHandPinky-Joint-Posi-z"]],
  ["RightHandPinky1", "RightHandPinky1", 33, 32, ["RightHandPinky1-Joint-Posi-x", "RightHandPinky1-Joint-Posi-y", "RightHandPinky1-Joint-Posi-z"]],
  ["RightHandPinky2", "RightHandPinky2", 34, 33, ["RightHandPinky2-Joint-Posi-x", "RightHandPinky2-Joint-Posi-y", "RightHandPinky2-Joint-Posi-z"]],
  ["RightHandPinky3", "RightHandPinky3", 35, 34, ["RightHandPinky3-Joint-Posi-x", "RightHandPinky3-Joint-Posi-y", "RightHandPinky3-Joint-Posi-z"]],
  ["LeftShoulder", "LeftShoulder", 36, 8, ["LeftShoulder-Joint-Posi-x", "LeftShoulder-Joint-Posi-y", "LeftShoulder-Joint-Posi-z"]],
  ["LeftArm", "LeftArm", 37, 36, ["LeftArm-Joint-Posi-x", "LeftArm-Joint-Posi-y", "LeftArm-Joint-Posi-z"]],
  ["LeftForeArm", "LeftForeArm", 38, 37, ["LeftForeArm-Joint-Posi-x", "LeftForeArm-Joint-Posi-y", "LeftForeArm-Joint-Posi-z"]],
  ["LeftHand", "LeftHand", 39, 38, ["LeftHand-Joint-Posi-x", "LeftHand-Joint-Posi-y", "LeftHand-Joint-Posi-z"]],
  ["LeftHandThumb1", "LeftHandThumb1", 40, 39, ["LeftHandThumb1-Joint-Posi-x", "LeftHandThumb1-Joint-Posi-y", "LeftHandThumb1-Joint-Posi-z"]],
  ["LeftHandThumb2", "LeftHandThumb2", 41, 40, ["LeftHandThumb2-Joint-Posi-x", "LeftHandThumb2-Joint-Posi-y", "LeftHandThumb2-Joint-Posi-z"]],
  ["LeftHandThumb3", "LeftHandThumb3", 42, 41, ["LeftHandThumb3-Joint-Posi-x", "LeftHandThumb3-Joint-Posi-y", "LeftHandThumb3-Joint-Posi-z"]],
  ["LeftInHandIndex", "LeftInHandIndex", 43, 39, ["LeftInHandIndex-Joint-Posi-x", "LeftInHandIndex-Joint-Posi-y", "LeftInHandIndex-Joint-Posi-z"]],
  ["LeftHandIndex1", "LeftHandIndex1", 44, 43, ["LeftHandIndex1-Joint-Posi-x", "LeftHandIndex1-Joint-Posi-y", "LeftHandIndex1-Joint-Posi-z"]],
  ["LeftHandIndex2", "LeftHandIndex2", 45, 44, ["LeftHandIndex2-Joint-Posi-x", "LeftHandIndex2-Joint-Posi-y", "LeftHandIndex2-Joint-Posi-z"]],
  ["LeftHandIndex3", "LeftHandIndex3", 46, 45, ["LeftHandIndex3-Joint-Posi-x", "LeftHandIndex3-Joint-Posi-y", "LeftHandIndex3-Joint-Posi-z"]],
  ["LeftInHandMiddle", "LeftInHandMiddle", 47, 39, ["LeftInHandMiddle-Joint-Posi-x", "LeftInHandMiddle-Joint-Posi-y", "LeftInHandMiddle-Joint-Posi-z"]],
  ["LeftHandMiddle1", "LeftHandMiddle1", 48, 47, ["LeftHandMiddle1-Joint-Posi-x", "LeftHandMiddle1-Joint-Posi-y", "LeftHandMiddle1-Joint-Posi-z"]],
  ["LeftHandMiddle2", "LeftHandMiddle2", 49, 48, ["LeftHandMiddle2-Joint-Posi-x", "LeftHandMiddle2-Joint-Posi-y", "LeftHandMiddle2-Joint-Posi-z"]],
  ["LeftHandMiddle3", "LeftHandMiddle3", 50, 49, ["LeftHandMiddle3-Joint-Posi-x", "LeftHandMiddle3-Joint-Posi-y", "LeftHandMiddle3-Joint-Posi-z"]],
  ["LeftInHandRing", "LeftInHandRing", 51, 39, ["LeftInHandRing-Joint-Posi-x", "LeftInHandRing-Joint-Posi-y", "LeftInHandRing-Joint-Posi-z"]],
  ["LeftHandRing1", "LeftHandRing1", 52, 51, ["LeftHandRing1-Joint-Posi-x", "LeftHandRing1-Joint-Posi-y", "LeftHandRing1-Joint-Posi-z"]],
  ["LeftHandRing2", "LeftHandRing2", 53, 52, ["LeftHandRing2-Joint-Posi-x", "LeftHandRing2-Joint-Posi-y", "LeftHandRing2-Joint-Posi-z"]],
  ["LeftHandRing3", "LeftHandRing3", 54, 53, ["LeftHandRing3-Joint-Posi-x", "LeftHandRing3-Joint-Posi-y", "LeftHandRing3-Joint-Posi-z"]],
  ["LeftInHandPinky", "LeftInHandPinky", 55, 39, ["LeftInHandPinky-Joint-Posi-x", "LeftInHandPinky-Joint-Posi-y", "LeftInHandPinky-Joint-Posi-z"]],
  ["LeftHandPinky1", "LeftHandPinky1", 56, 55, ["LeftHandPinky1-Joint-Posi-x", "LeftHandPinky1-Joint-Posi-y", "LeftHandPinky1-Joint-Posi-z"]],
  ["LeftHandPinky2", "LeftHandPinky2", 57, 56, ["LeftHandPinky2-Joint-Posi-x", "LeftHandPinky2-Joint-Posi-y", "LeftHandPinky2-Joint-Posi-z"]],
  ["LeftHandPinky3", "LeftHandPinky3", 58, 57, ["LeftHandPinky3-Joint-Posi-x", "LeftHandPinky3-Joint-Posi-y", "LeftHandPinky3-Joint-Posi-z"]],
];

const boneConnections_IMU = skeletonDefinition_IMU.filter(bone => bone[3] !== -1).map(bone => [bone[2], bone[3]]);
var allColumns_IMU = [];
skeletonDefinition_IMU.forEach(bone => allColumns_IMU.push(...bone[4]));

var scene_IMU, camera_IMU, renderer_IMU, controls_IMU;
var jointMeshes_IMU = [], boneMeshes_IMU = [];
var gridHelper_IMU, axesHelper_IMU;

var motionData_IMU = [], totalFrames_IMU = 0, currentFrame_IMU = 0, isPlaying_IMU = false, lastTimestamp_IMU = 0;
var analysisResults_IMU = { velocitiesLeft: [], velocitiesRight: [], tilts: [], directions: [] };

function initScene_IMU() {
  const container = document.getElementById('visualization-container');
  if (!container) return;

  scene_IMU = new THREE.Scene();
  scene_IMU.background = new THREE.Color(0x05070a);
  
  camera_IMU = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera_IMU.position.set(3, 1.8, 3.5);
  
  try {
    renderer_IMU = new THREE.WebGLRenderer({ 
      antialias: true, 
      preserveDrawingBuffer: true,
      powerPreference: "high-performance" 
    });
    renderer_IMU.setSize(container.clientWidth, container.clientHeight);
    renderer_IMU.shadowMap.enabled = true; 
    container.appendChild(renderer_IMU.domElement);
  } catch (e) {
    console.error("WebGL Initialization Failed:", e);
    return;
  }

  controls_IMU = new THREE.OrbitControls(camera_IMU, renderer_IMU.domElement);
  controls_IMU.target.set(0, 0.9, 0);
  controls_IMU.enableDamping = true;
  controls_IMU.dampingFactor = 0.05;
  controls_IMU.update();

  scene_IMU.add(new THREE.AmbientLight(0xffffff, 0.3));
  const spotLight = new THREE.SpotLight(0xffffff, 1.2);
  spotLight.position.set(5, 15, 10);
  spotLight.castShadow = true;
  scene_IMU.add(spotLight);
  
  axesHelper_IMU = new THREE.AxesHelper(AXES_HELPER_SCALE_IMU);
  axesHelper_IMU.position.set(-1.5, 0, -1.5); 
  scene_IMU.add(axesHelper_IMU); 

  window.addEventListener('resize', onWindowResize_IMU, false);
  initCharts_IMU();
  animate_IMU(0);
}

function onWindowResize_IMU() {
  const container = document.getElementById('visualization-container');
  if (!container || !renderer_IMU) return;
  camera_IMU.aspect = container.clientWidth / container.clientHeight;
  camera_IMU.updateProjectionMatrix();
  renderer_IMU.setSize(container.clientWidth, container.clientHeight);
}

function initCharts_IMU() {
  const chartStyles = (x, y) => ({
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { title: { display: true, text: x, color: '#64748b', font: { size: 9 } }, ticks: { display: false }, grid: { color: '#0f172a' } },
      y: { title: { display: true, text: y, color: '#64748b', font: { size: 9 } }, grid: { color: '#0f172a' }, ticks: { color: '#475569', font: { size: 8 } } }
    },
    plugins: { 
      legend: { 
        display: true, 
        position: 'top', 
        labels: { color: '#64748b', boxWidth: 10, font: { size: 8 } } 
      } 
    },
    animation: false
  });

  const vCtx = document.getElementById('velocityChart_IMU');
  const tCtx = document.getElementById('tiltChart_IMU');
  const rCtx = document.getElementById('roseChart_IMU');

  if (vCtx) {
    charts_IMU.velocity = new Chart(vCtx, {
      type: 'line',
      data: { 
        labels: [], 
        datasets: [
          { label: 'Left Leg', data: [], borderColor: '#10b981', borderWidth: 1.5, pointRadius: 0, fill: false },
          { label: 'Right Leg', data: [], borderColor: '#f43f5e', borderWidth: 1.5, pointRadius: 0, fill: false }
        ] 
      },
      options: chartStyles('Frame Index', 'rad/s'),
      plugins: [{ id: 'vline', afterDraw: drawVerticalLine_IMU }]
    });
  }

  if (tCtx) {
    charts_IMU.tilt = new Chart(tCtx, {
      type: 'line',
      data: { labels: [], datasets: [{ data: [], borderColor: '#00e5ff', borderWidth: 1.5, pointRadius: 0, fill: true, backgroundColor: 'rgba(0, 229, 255, 0.05)' }] },
      options: Object.assign(chartStyles('Frame Index', 'Degrees'), { plugins: { legend: { display: false } } }),
      plugins: [{ id: 'vline', afterDraw: drawVerticalLine_IMU }]
    });
  }

  if (rCtx) {
    charts_IMU.rose = new Chart(rCtx, {
      type: 'polarArea',
      data: {
        labels: ['Front', 'Front-Right', 'Right', 'Back-Right', 'Back', 'Back-Left', 'Left', 'Front-Left'],
        datasets: [{ data: Array(8).fill(0), backgroundColor: ['#ef4444aa', '#f59e0baa', '#10b981aa', '#06b6d4aa', '#00e5ffaa', '#8b5cf6aa', '#ec4899aa', '#f43f5eaa'] }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { r: { grid: { color: '#1e293b' }, ticks: { display: false } } },
        plugins: { legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 9 } } } }
      }
    });
  }
}

function drawVerticalLine_IMU(chart) {
  if (totalFrames_IMU === 0 || !chart.scales.x) return;
  const ctx = chart.ctx;
  const x = chart.scales.x.getPixelForValue(currentFrame_IMU);
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x, chart.chartArea.top);
  ctx.lineTo(x, chart.chartArea.bottom);
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#f43f5e';
  ctx.stroke();
  ctx.restore();
}

function handleFileUpload_IMU(event) {
  const file = event.target.files[0];
  if (!file) return;
  const loadingOverlay = document.getElementById('loadingOverlay');
  if (loadingOverlay) loadingOverlay.style.display = 'flex';
  const reader = new FileReader();
  reader.onload = (e) => {
    parseCSV_IMU(e.target.result);
    performAnalysis_IMU();
    if (loadingOverlay) loadingOverlay.style.display = 'none';
  };
  reader.readAsText(file);
}

function parseCSV_IMU(csvText) {
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return;
  const header = lines[0].split(',').map(h => h.trim());

  motionData_IMU = [];
  let minC = new THREE.Vector3(Infinity, Infinity, Infinity), maxC = new THREE.Vector3(-Infinity, -Infinity, -Infinity);

  // Kinect CSV format: joint_X_x, joint_X_y, joint_X_z
  // Map IMU skeleton joints to Kinect joint indices
  const kinectJointMap = {
    0: 0,   // Hips -> joint_0 (pelvis)
    1: 1,   // RightUpLeg -> joint_1
    2: 2,   // RightLeg -> joint_2
    3: 3,   // RightFoot -> joint_3
    4: 4,   // LeftUpLeg -> joint_4
    5: 5,   // LeftLeg -> joint_5
    6: 6,   // LeftFoot -> joint_6
    7: 7,   // Spine -> joint_7
    8: 8,   // Spine1 -> joint_8
    9: 9,   // Spine2 -> joint_9
    10: 10, // Neck -> joint_10
    11: 11, // Neck1 -> joint_11
    12: 12, // Head -> joint_12
    13: 13, // RightShoulder -> joint_13
    14: 14, // RightArm -> joint_14
    15: 15, // RightForeArm -> joint_15
    16: 16, // RightHand -> joint_16
    17: 17, // RightHandThumb1 -> joint_17
    18: 18, // RightHandThumb2 -> joint_18
    19: 19, // RightHandThumb3 -> joint_19
    20: 20, // RightInHandIndex -> joint_20
    21: 21, // RightHandIndex1 -> joint_21
    22: 22, // RightHandIndex2 -> joint_22
    23: 23, // RightHandIndex3 -> joint_23
    24: 24, // RightInHandMiddle -> joint_24
    25: 25, // RightHandMiddle1 -> joint_25
    26: 26, // RightHandMiddle2 -> joint_26
    27: 27, // RightHandMiddle3 -> joint_27
    28: 28, // RightInHandRing -> joint_28
    29: 29, // RightHandRing1 -> joint_29
    30: 30, // RightHandRing2 -> joint_30
    31: 31, // RightHandRing3 -> joint_31
    32: 32, // RightInHandPinky -> joint_32 (may not exist)
    33: 33, // RightHandPinky1 -> joint_33
    34: 34, // RightHandPinky2 -> joint_34
    35: 35, // RightHandPinky3 -> joint_35
    36: 36, // LeftShoulder -> joint_36 (may not exist)
    37: 37, // LeftArm -> joint_37
    38: 38, // LeftForeArm -> joint_38
    39: 39, // LeftHand -> joint_39
    40: 40, // LeftHandThumb1 -> joint_40
    41: 41, // LeftHandThumb2 -> joint_41
    42: 42, // LeftHandThumb3 -> joint_42
    43: 43, // LeftInHandIndex -> joint_43
    44: 44, // LeftHandIndex1 -> joint_44
    45: 45, // LeftHandIndex2 -> joint_45
    46: 46, // LeftHandIndex3 -> joint_46
    47: 47, // LeftInHandMiddle -> joint_47
    48: 48, // LeftHandMiddle1 -> joint_48
    49: 49, // LeftHandMiddle2 -> joint_49
    50: 50, // LeftHandMiddle3 -> joint_50
    51: 51, // LeftInHandRing -> joint_51
    52: 52, // LeftHandRing1 -> joint_52
    53: 53, // LeftHandRing2 -> joint_53
    54: 54, // LeftHandRing3 -> joint_54
    55: 55, // LeftInHandPinky -> joint_55
    56: 56, // LeftHandPinky1 -> joint_56
    57: 57, // LeftHandPinky2 -> joint_57
    58: 58  // LeftHandPinky3 -> joint_58
  };

  // Find column indices for Kinect format
  const colIndices = [];
  for (let j = 0; j < skeletonDefinition_IMU.length; j++) {
    const kinectIdx = kinectJointMap[j];
    const xIdx = header.indexOf(`joint_${kinectIdx}_x`);
    const yIdx = header.indexOf(`joint_${kinectIdx}_y`);
    const zIdx = header.indexOf(`joint_${kinectIdx}_z`);
    colIndices.push([xIdx, yIdx, zIdx]);
  }

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => parseFloat(v.trim()));
    const framePos = new Array(skeletonDefinition_IMU.length);
    let valid = false;
    for (let j = 0; j < skeletonDefinition_IMU.length; j++) {
      const [xIdx, yIdx, zIdx] = colIndices[j];
      if (xIdx !== -1 && !isNaN(values[xIdx])) {
        const p = new THREE.Vector3(values[xIdx], values[yIdx], values[zIdx]);
        minC.min(p); maxC.max(p);
        framePos[j] = p;
        valid = true;
      }
    }
    if (valid) motionData_IMU.push(framePos);
  }
  
  totalFrames_IMU = motionData_IMU.length;
  if (totalFrames_IMU > 0) {
    const center = minC.clone().add(maxC).multiplyScalar(0.5);
    const scale = 1.8 / (maxC.y - minC.y || 1); 
    motionData_IMU.forEach(f => f.forEach(p => p && p.sub(center).multiplyScalar(scale).add(new THREE.Vector3(0, 0.9, 0))));

    setupSkeletonMeshes_IMU();
    const slider = document.getElementById('frameSlider');
    if (slider) {
      slider.max = totalFrames_IMU - 1; 
      slider.disabled = false;
    }
    const totalFramesDisplay = document.getElementById('totalFramesDisplay');
    if (totalFramesDisplay) totalFramesDisplay.textContent = totalFrames_IMU;
    const playPauseButton = document.getElementById('playPauseButton');
    if (playPauseButton) playPauseButton.disabled = false;
    const resetButton = document.getElementById('resetButton');
    if (resetButton) resetButton.disabled = false;
    const exportGifButton = document.getElementById('exportGifButton');
    if (exportGifButton) exportGifButton.disabled = false;
    currentFrame_IMU = 0;
    isPlaying_IMU = true;
  }
}

function setupSkeletonMeshes_IMU() {
  jointMeshes_IMU.forEach(m => scene_IMU.remove(m)); 
  boneMeshes_IMU.forEach(m => scene_IMU.remove(m));
  if (gridHelper_IMU) scene_IMU.remove(gridHelper_IMU);
  if (trajectoryLines_IMU.left) scene_IMU.remove(trajectoryLines_IMU.left);
  if (trajectoryLines_IMU.right) scene_IMU.remove(trajectoryLines_IMU.right);
  
  jointMeshes_IMU = []; 
  boneMeshes_IMU = [];

  const mat = new THREE.MeshStandardMaterial({ 
    color: SKELETON_COLOR_IMU, 
    emissive: SKELETON_COLOR_IMU, 
    emissiveIntensity: 0.6, 
    metalness: 0.9, 
    roughness: 0.1 
  });

  skeletonDefinition_IMU.forEach((bone, i) => {
    const r = (bone[0].includes("Thumb") || bone[0].includes("Foot")) ? JOINT_RADIUS_IMU * 0.5 : JOINT_RADIUS_IMU;
    const m = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 12), mat);
    m.castShadow = true; 
    scene_IMU.add(m); 
    jointMeshes_IMU.push(m);
  });

  boneConnections_IMU.forEach(() => {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(BONE_RADIUS_IMU, BONE_RADIUS_IMU, 1, 8), mat);
    m.castShadow = true; 
    scene_IMU.add(m); 
    boneMeshes_IMU.push(m);
  });

  const createTrajectory = (color) => {
    const geo = new THREE.BufferGeometry();
    const lineMat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.6 });
    const line = new THREE.Line(geo, lineMat);
    const trajectoryVisibility = document.getElementById('trajectoryVisibility');
    line.visible = trajectoryVisibility ? trajectoryVisibility.checked : true;
    scene_IMU.add(line);
    return line;
  };
  trajectoryLines_IMU.left = createTrajectory(TRAJECTORY_COLOR_LEFT_IMU);
  trajectoryLines_IMU.right = createTrajectory(TRAJECTORY_COLOR_RIGHT_IMU);
  
  gridHelper_IMU = new THREE.GridHelper(10, 50, 0x1e293b, 0x0f172a);
  scene_IMU.add(gridHelper_IMU);
  updateSkeleton_IMU(currentFrame_IMU);
}

const _v_IMU = new THREE.Vector3(), _q_IMU = new THREE.Quaternion();
function updateSkeleton_IMU(idx) {
  if (motionData_IMU.length === 0) return;
  const pos = motionData_IMU[idx];
  if (!pos) return;

  pos.forEach((p, i) => {
    if (p && jointMeshes_IMU[i]) { jointMeshes_IMU[i].position.copy(p); jointMeshes_IMU[i].visible = true; }
    else if (jointMeshes_IMU[i]) { jointMeshes_IMU[i].visible = false; }
  });
  boneConnections_IMU.forEach(([cIdx, pIdx], i) => {
    const c = pos[cIdx], p = pos[pIdx];
    if (c && p && boneMeshes_IMU[i]) {
      boneMeshes_IMU[i].position.copy(p).lerp(c, 0.5);
      _v_IMU.subVectors(c, p).normalize();
      _q_IMU.setFromUnitVectors(new THREE.Vector3(0, 1, 0), _v_IMU);
      boneMeshes_IMU[i].scale.set(1, c.distanceTo(p), 1);
      boneMeshes_IMU[i].quaternion.copy(_q_IMU);
      boneMeshes_IMU[i].visible = true;
    } else if (boneMeshes_IMU[i]) { boneMeshes_IMU[i].visible = false; }
  });

  const updatePath = (line, jointIdx) => {
    const points = [];
    for(let k=0; k<=idx; k++) {
      const p = motionData_IMU[k][jointIdx];
      if(p) points.push(p.x, p.y, p.z);
    }
    line.geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    line.geometry.attributes.position.needsUpdate = true;
  };
  updatePath(trajectoryLines_IMU.right, 3); 
  updatePath(trajectoryLines_IMU.left, 6);  
  
  if (charts_IMU.velocity) charts_IMU.velocity.draw(); 
  if (charts_IMU.tilt) charts_IMU.tilt.draw();
}

function performAnalysis_IMU() {
  analysisResults_IMU = { velocitiesLeft: [0], velocitiesRight: [0], tilts: [], directions: Array(8).fill(0) };
  for (let i = 0; i < totalFrames_IMU; i++) {
    const frame = motionData_IMU[i];
    const hips = frame[0], spine = frame[9]; 
    
    if (hips && spine) {
      const trunkVec = new THREE.Vector3().subVectors(spine, hips);
      analysisResults_IMU.tilts.push(trunkVec.angleTo(new THREE.Vector3(0, 1, 0)) * (180 / Math.PI));
      const angle = Math.atan2(trunkVec.x, trunkVec.z);
      let bin = Math.round(angle / (Math.PI / 4));
      if (bin < 0) bin += 8;
      analysisResults_IMU.directions[bin % 8]++;
    }

    if (i > 0) {
      const pF = motionData_IMU[i-1];
      
      const rKnee = frame[2], rAnkle = frame[3], prK = pF[2], prA = pF[3];
      if (rKnee && rAnkle && prK && prA) {
        const v1 = new THREE.Vector3().subVectors(rAnkle, rKnee).normalize();
        const v2 = new THREE.Vector3().subVectors(prA, prK).normalize();
        analysisResults_IMU.velocitiesRight.push(v1.angleTo(v2) * FPS_IMU);
      } else analysisResults_IMU.velocitiesRight.push(0);

      const lKnee = frame[5], lAnkle = frame[6], plK = pF[5], plA = pF[6];
      if (lKnee && lAnkle && plK && plA) {
        const v1 = new THREE.Vector3().subVectors(lAnkle, lKnee).normalize();
        const v2 = new THREE.Vector3().subVectors(plA, plK).normalize();
        analysisResults_IMU.velocitiesLeft.push(v1.angleTo(v2) * FPS_IMU);
      } else analysisResults_IMU.velocitiesLeft.push(0);
    }
  }
  const labels = Array.from({length: totalFrames_IMU}, (_, i) => i);
  if (charts_IMU.velocity) { 
    charts_IMU.velocity.data.labels = labels; 
    charts_IMU.velocity.data.datasets[0].data = analysisResults_IMU.velocitiesLeft; 
    charts_IMU.velocity.data.datasets[1].data = analysisResults_IMU.velocitiesRight; 
    charts_IMU.velocity.update(); 
  }
  if (charts_IMU.tilt) { charts_IMU.tilt.data.labels = labels; charts_IMU.tilt.data.datasets[0].data = analysisResults_IMU.tilts; charts_IMU.tilt.update(); }
  if (charts_IMU.rose) { charts_IMU.rose.data.datasets[0].data = analysisResults_IMU.directions; charts_IMU.rose.update(); }
}

function animate_IMU(ts) {
  requestAnimationFrame(animate_IMU);
  if (isPlaying_IMU && totalFrames_IMU > 0) {
    const delta = ts - lastTimestamp_IMU;
    if (delta >= frameInterval_IMU) {
      lastTimestamp_IMU = ts - (delta % frameInterval_IMU);
      currentFrame_IMU = (currentFrame_IMU + 1) % totalFrames_IMU;
      updateSkeleton_IMU(currentFrame_IMU); 
      const frameSlider = document.getElementById('frameSlider');
      if (frameSlider) frameSlider.value = currentFrame_IMU;
      const frameInfo = document.getElementById('frame-info');
      if (frameInfo) frameInfo.textContent = `FRAME: ${currentFrame_IMU + 1} / ${totalFrames_IMU}`;
    }
  }
  if (controls_IMU) controls_IMU.update(); 
  if (renderer_IMU && scene_IMU && camera_IMU) renderer_IMU.render(scene_IMU, camera_IMU);
}

function handleFrameScrubbing_IMU(v) { 
  isPlaying_IMU = false; 
  currentFrame_IMU = parseInt(v); 
  updateSkeleton_IMU(currentFrame_IMU); 
  const playPauseText = document.getElementById('playPauseText');
  if (playPauseText) playPauseText.textContent = 'Play'; 
}

function toggleAnimation_IMU() { 
  isPlaying_IMU = !isPlaying_IMU; 
  const playPauseText = document.getElementById('playPauseText');
  if (playPauseText) playPauseText.textContent = isPlaying_IMU ? 'Pause' : 'Play'; 
}

function resetAnimation_IMU() { 
  isPlaying_IMU = false; 
  currentFrame_IMU = 0; 
  updateSkeleton_IMU(0); 
  const playPauseText = document.getElementById('playPauseText');
  if (playPauseText) playPauseText.textContent = 'Play'; 
}

function toggleAxesVisibility_IMU(v) { if(axesHelper_IMU) axesHelper_IMU.visible = v; }
function toggleTrajectoryVisibility_IMU(v) { 
  if(trajectoryLines_IMU.left) trajectoryLines_IMU.left.visible = v; 
  if(trajectoryLines_IMU.right) trajectoryLines_IMU.right.visible = v; 
}

// ============================================
// Kinect Data Visualization
// ============================================

let frames_Kinect = [];
let charts_Kinect = {};
let currentIndex_Kinect = 0;
let isPlaying_Kinect = false;
let animationId_Kinect = null;
let lastTimestamp_Kinect = 0;

let scene_Kinect, camera_Kinect, renderer_Kinect, controls_Kinect;
let jointMeshes_Kinect = [];
let boneLines_Kinect = [];
let highLightBall_Kinect;

const connections_Kinect = [
  [26, 3], [3, 2], [2, 1], [1, 0], 
  [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [8, 10],
  [3, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [15, 17],
  [0, 18], [18, 19], [19, 20], [20, 21],
  [0, 22], [22, 23], [23, 24], [24, 25],
  [26, 27], [27, 28], [27, 30] 
];

function initThree_Kinect() {
  const container = document.getElementById('skeleton3d');
  if (!container) return;
  const w = container.offsetWidth;
  const h = container.offsetHeight;

  scene_Kinect = new THREE.Scene();
  scene_Kinect.background = new THREE.Color(0x050505);

  camera_Kinect = new THREE.PerspectiveCamera(45, w / h, 1, 30000);
  camera_Kinect.position.set(0, 1200, 3500);

  renderer_Kinect = new THREE.WebGLRenderer({ antialias: true });
  renderer_Kinect.setSize(w, h);
  renderer_Kinect.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer_Kinect.domElement);

  controls_Kinect = new THREE.OrbitControls(camera_Kinect, renderer_Kinect.domElement);
  controls_Kinect.enableDamping = true;
  controls_Kinect.dampingFactor = 0.1;

  scene_Kinect.add(new THREE.AmbientLight(0xffffff, 0.7));
  const pointLight = new THREE.PointLight(0xffffff, 0.8);
  pointLight.position.set(0, 5000, 5000);
  scene_Kinect.add(pointLight);

  const gridHelper = new THREE.GridHelper(10000, 40, 0x333333, 0x111111);
  gridHelper.position.y = 0; 
  scene_Kinect.add(gridHelper);

  const sphereGeo = new THREE.SphereGeometry(20, 16, 16);
  for (let i = 0; i < 32; i++) {
    const mat = new THREE.MeshPhongMaterial({ color: i === 26 ? 0xffdd00 : 0xeeeeee });
    const mesh = new THREE.Mesh(sphereGeo, mat);
    mesh.visible = false;
    scene_Kinect.add(mesh);
    jointMeshes_Kinect.push(mesh);
  }

  const hlGeo = new THREE.SphereGeometry(55, 32, 32);
  const hlMat = new THREE.MeshBasicMaterial({ 
    color: 0xff3333, 
    transparent: true, 
    opacity: 0.4,
    wireframe: true 
  });
  highLightBall_Kinect = new THREE.Mesh(hlGeo, hlMat);
  highLightBall_Kinect.visible = false;
  scene_Kinect.add(highLightBall_Kinect);

  const lineMat = new THREE.LineBasicMaterial({ color: 0x3b82f6, linewidth: 3 });
  connections_Kinect.forEach(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    const line = new THREE.Line(geometry, lineMat);
    line.visible = false;
    scene_Kinect.add(line);
    boneLines_Kinect.push(line);
  });

  function animate_Kinect(time) {
    requestAnimationFrame(animate_Kinect);
    if(highLightBall_Kinect.visible) {
      const s = 1 + Math.sin(time * 0.008) * 0.15;
      highLightBall_Kinect.scale.set(s, s, s);
    }
    controls_Kinect.update();
    renderer_Kinect.render(scene_Kinect, camera_Kinect);
  }
  animate_Kinect(0);
}

function initCharts_Kinect() {
  const vLine = {
    id: 'vLine',
    afterDraw: (chart) => {
      if (chart.tooltip?._active?.length || (!isPlaying_Kinect && currentIndex_Kinect === 0)) return;
      const ctx = chart.ctx;
      const x = chart.scales.x.getPixelForValue(currentIndex_Kinect);
      if (x < chart.chartArea.left || x > chart.chartArea.right) return;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(x, chart.chartArea.top);
      ctx.lineTo(x, chart.chartArea.bottom);
      ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)'; ctx.stroke();
      ctx.restore();
    }
  };

  const baseCfg = (col) => ({
    type: 'line', plugins: [vLine],
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: { 
        x: { display: false }, 
        y: { grid: { color: 'rgba(75, 85, 99, 0.2)' }, ticks: { color: '#9ca3af', font: {size: 10} } } 
      }
    }
  });

  const speedChart = document.getElementById('speedChart_Kinect');
  const accChart = document.getElementById('accChart_Kinect');
  const posChart = document.getElementById('posChart_Kinect');

  if (speedChart) {
    charts_Kinect.speed = new Chart(speedChart, {
      ...baseCfg('#3b82f6'),
      data: { labels: [], datasets: [{ data: [], borderColor: '#3b82f6', borderWidth: 2, pointRadius: 0, fill: true, backgroundColor: 'rgba(59, 130, 246, 0.1)' }] }
    });
  }
  if (accChart) {
    charts_Kinect.acc = new Chart(accChart, {
      ...baseCfg('#ef4444'),
      data: { labels: [], datasets: [{ data: [], borderColor: '#ef4444', borderWidth: 2, pointRadius: 0 }] }
    });
  }
  if (posChart) {
    charts_Kinect.pos = new Chart(posChart, {
      ...baseCfg('#10b981'),
      options: { ...baseCfg('#10b981').options, plugins: { legend: { display: true, labels: { color: '#9ca3af', boxWidth: 8, font: {size: 10} } } } },
      data: { labels: [], datasets: [
        { label: 'X', data: [], borderColor: '#ef4444', borderWidth: 1, pointRadius: 0 },
        { label: 'Y', data: [], borderColor: '#10b981', borderWidth: 1, pointRadius: 0 },
        { label: 'Z', data: [], borderColor: '#3b82f6', borderWidth: 1, pointRadius: 0 }
      ]}
    });
  }
}

function updateThreeSkeleton_Kinect(idx) {
  const frame = frames_Kinect[idx];
  if (!frame) return;

  const jointSelect = document.getElementById('jointSelect');
  const targetId = jointSelect ? parseInt(jointSelect.value) : 20;
  
  let minY = Infinity;
  for (let i = 0; i < 32; i++) {
    const y = frame[`joint_${i}_y`];
    if (y !== undefined) {
      if (-y < minY) minY = -y;
    }
  }
  const yOffset = -minY;

  for (let i = 0; i < 32; i++) {
    const x = frame[`joint_${i}_x`], y = frame[`joint_${i}_y`], z = frame[`joint_${i}_z`];
    if (x !== undefined) {
      const posX = -x;
      const posY = -y + yOffset; 
      const posZ = z;

      jointMeshes_Kinect[i].position.set(posX, posY, posZ);
      jointMeshes_Kinect[i].visible = true;
      
      if (i === targetId) {
        highLightBall_Kinect.position.set(posX, posY, posZ);
        highLightBall_Kinect.visible = true;
      }
    } else { jointMeshes_Kinect[i].visible = false; }
  }

  connections_Kinect.forEach((conn, i) => {
    const j1 = jointMeshes_Kinect[conn[0]], j2 = jointMeshes_Kinect[conn[1]];
    if (j1 && j2 && j1.visible && j2.visible) {
      const posAttr = boneLines_Kinect[i].geometry.attributes.position;
      posAttr.array[0] = j1.position.x; posAttr.array[1] = j1.position.y; posAttr.array[2] = j1.position.z;
      posAttr.array[3] = j2.position.x; posAttr.array[4] = j2.position.y; posAttr.array[5] = j2.position.z;
      posAttr.needsUpdate = true;
      boneLines_Kinect[i].visible = true;
    } else { if(boneLines_Kinect[i]) boneLines_Kinect[i].visible = false; }
  });

  if (jointMeshes_Kinect[0].visible) {
    controls_Kinect.target.lerp(jointMeshes_Kinect[0].position, 0.2); 
  }
}

function calculatePhysics_Kinect() {
  const jointSelect = document.getElementById('jointSelect');
  const jointId = jointSelect ? jointSelect.value : 20;
  const labels = frames_Kinect.map(f => f.frame_id);
  const speeds = [0], accs = [0], pX = [], pY = [], pZ = [];
  const sX = frames_Kinect[0][`joint_${jointId}_x`]||0, sY = frames_Kinect[0][`joint_${jointId}_y`]||0, sZ = frames_Kinect[0][`joint_${jointId}_z`]||0;

  for (let i = 1; i < frames_Kinect.length; i++) {
    const f1 = frames_Kinect[i-1], f2 = frames_Kinect[i];
    const dt = (f2.timestamp_usec - f1.timestamp_usec) / 1000000;
    const x1 = f1[`joint_${jointId}_x`]||0, y1 = f1[`joint_${jointId}_y`]||0, z1 = f1[`joint_${jointId}_z`]||0;
    const x2 = f2[`joint_${jointId}_x`]||0, y2 = f2[`joint_${jointId}_y`]||0, z2 = f2[`joint_${jointId}_z`]||0;

    const dist = Math.sqrt(Math.pow(x2-x1, 2) + Math.pow(y2-y1, 2) + Math.pow(z2-z1, 2)) / 1000;
    const v = dt > 0 ? dist / dt : 0;
    speeds.push(v);
    accs.push(dt > 0 ? Math.abs(v - (speeds[i-1]||0))/dt : 0);
    
    pX.push((-f2[`joint_${jointId}_x`] - (-sX))/1000);
    pY.push((-f2[`joint_${jointId}_y`] - (-sY))/1000);
    pZ.push((f2[`joint_${jointId}_z`]-sZ)/1000);
  }

  if (charts_Kinect.speed) {
    charts_Kinect.speed.data.labels = labels; 
    charts_Kinect.speed.data.datasets[0].data = speeds; 
    charts_Kinect.speed.update('none');
  }
  if (charts_Kinect.acc) {
    charts_Kinect.acc.data.labels = labels; 
    charts_Kinect.acc.data.datasets[0].data = accs; 
    charts_Kinect.acc.update('none');
  }
  if (charts_Kinect.pos) {
    charts_Kinect.pos.data.labels = labels;
    charts_Kinect.pos.data.datasets[0].data = pX; 
    charts_Kinect.pos.data.datasets[1].data = pY; 
    charts_Kinect.pos.data.datasets[2].data = pZ;
    charts_Kinect.pos.update('none');
  }
}

function playbackLoop_Kinect(timestamp) {
  if (!isPlaying_Kinect) return;
  if (timestamp - lastTimestamp_Kinect > 33) {
    if (currentIndex_Kinect >= frames_Kinect.length - 1) togglePlay_Kinect(false);
    else { currentIndex_Kinect++; updateUI_Kinect(); }
    lastTimestamp_Kinect = timestamp;
  }
  animationId_Kinect = requestAnimationFrame(playbackLoop_Kinect);
}

function updateUI_Kinect() {
  const frame = frames_Kinect[currentIndex_Kinect];
  if (!frame) return;
  const progressBar = document.getElementById('progressBar');
  if (progressBar) progressBar.value = currentIndex_Kinect;
  const curFrameId = document.getElementById('curFrameId');
  if (curFrameId) curFrameId.innerText = `#${frame.frame_id}`;
  const currentTime = document.getElementById('currentTime');
  if (currentTime) currentTime.innerText = (frame.timestamp_usec / 1000000).toFixed(2) + "s";
  const curSpeed = document.getElementById('curSpeed');
  if (curSpeed && charts_Kinect.speed) curSpeed.innerText = (charts_Kinect.speed.data.datasets[0].data[currentIndex_Kinect]||0).toFixed(3);
  const curAcc = document.getElementById('curAcc');
  if (curAcc && charts_Kinect.acc) curAcc.innerText = (charts_Kinect.acc.data.datasets[0].data[currentIndex_Kinect]||0).toFixed(3);
  Object.values(charts_Kinect).forEach(c => c.update('none'));
  updateThreeSkeleton_Kinect(currentIndex_Kinect);
}

function togglePlay_Kinect(state) {
  isPlaying_Kinect = state;
  const playIcon = document.getElementById('playIcon');
  const pauseIcon = document.getElementById('pauseIcon');
  if (playIcon) playIcon.style.display = isPlaying_Kinect ? 'none' : 'block';
  if (pauseIcon) pauseIcon.style.display = isPlaying_Kinect ? 'block' : 'none';
  if (isPlaying_Kinect) { 
    lastTimestamp_Kinect = performance.now(); 
    animationId_Kinect = requestAnimationFrame(playbackLoop_Kinect); 
  }
  else { cancelAnimationFrame(animationId_Kinect); }
}

function handleFileUpload_Kinect(e) {
  const file = e.target.files[0];
  if (!file) return;
  
  const display = document.getElementById('fileNameDisplay');
  if (display) display.innerText = file.name;

  Papa.parse(file, {
    header: true, dynamicTyping: true, skipEmptyLines: true,
    complete: (results) => {
      frames_Kinect = results.data; 
      currentIndex_Kinect = 0;
      const progressBar = document.getElementById('progressBar');
      if (progressBar) progressBar.max = frames_Kinect.length - 1;
      const duration = document.getElementById('duration');
      if (duration) duration.innerText = (frames_Kinect[frames_Kinect.length-1].timestamp_usec / 1000000).toFixed(2) + "s";
      const playbackControls = document.getElementById('playbackControls');
      if (playbackControls) {
        playbackControls.classList.remove('opacity-50');
        playbackControls.style.pointerEvents = 'auto';
      }
      calculatePhysics_Kinect(); 
      updateUI_Kinect(); 
      togglePlay_Kinect(true);
    }
  });
}

function resetCam_Kinect() {
  if(jointMeshes_Kinect[0].visible) {
    const p = jointMeshes_Kinect[0].position;
    camera_Kinect.position.set(p.x, p.y + 1200, p.z + 3500);
    controls_Kinect.target.copy(p);
  } else {
    camera_Kinect.position.set(0, 1200, 3500);
    controls_Kinect.target.set(0, 0, 0);
  }
  controls_Kinect.update();
}

function onWindowResize_Kinect() {
  const container = document.getElementById('skeleton3d');
  if (!container || !camera_Kinect || !renderer_Kinect) return;
  camera_Kinect.aspect = container.offsetWidth / container.offsetHeight;
  camera_Kinect.updateProjectionMatrix();
  renderer_Kinect.setSize(container.offsetWidth, container.offsetHeight);
}

// ============================================
// 统一控制支持函数
// ============================================

// IMU 统一更新函数
function updateIMUVisualization(frame) {
  if (motionData_IMU.length === 0 || frame >= motionData_IMU.length) return;
  currentFrame_IMU = frame;
  updateSkeleton_IMU(frame);
  
  // 更新图表垂直线
  if (charts_IMU.velocity) charts_IMU.velocity.draw();
  if (charts_IMU.tilt) charts_IMU.tilt.draw();
}

// Kinect 统一更新函数
function updateKinectVisualization(frame) {
  if (!frames_Kinect || frames_Kinect.length === 0 || frame >= frames_Kinect.length) return;
  currentIndex_Kinect = frame;
  updateUI_Kinect();
}

// 更新 Kinect UI（用于统一控制）
function updateUI_Kinect() {
  const frame = frames_Kinect[currentIndex_Kinect];
  if (!frame) return;
  
  // 更新统计信息
  const curSpeed = document.getElementById('curSpeed_Kinect');
  const curAcc = document.getElementById('curAcc_Kinect');
  const curFrameId = document.getElementById('curFrameId_Kinect');
  
  if (curFrameId) curFrameId.innerText = `#${frame.frame_id || currentIndex_Kinect}`;
  if (curSpeed && charts_Kinect.speed && charts_Kinect.speed.data.datasets[0].data.length > currentIndex_Kinect) {
    curSpeed.innerText = (charts_Kinect.speed.data.datasets[0].data[currentIndex_Kinect] || 0).toFixed(3);
  }
  if (curAcc && charts_Kinect.acc && charts_Kinect.acc.data.datasets[0].data.length > currentIndex_Kinect) {
    curAcc.innerText = (charts_Kinect.acc.data.datasets[0].data[currentIndex_Kinect] || 0).toFixed(3);
  }
  
  // 更新3D骨骼
  updateThreeSkeleton_Kinect(currentIndex_Kinect);
  
  // 更新图表
  Object.values(charts_Kinect).forEach(c => { if(c) c.update('none'); });
}

// 计算 Kinect 物理数据
function calculatePhysics_Kinect() {
  const jointSelect = document.getElementById('jointSelect_Kinect');
  const jointId = jointSelect ? jointSelect.value : 20;
  if (!frames_Kinect || frames_Kinect.length === 0) return;
  
  const labels = frames_Kinect.map(f => f.frame_id || 0);
  const speeds = [0], accs = [0], pX = [], pY = [], pZ = [];
  const sX = frames_Kinect[0][`joint_${jointId}_x`] || 0;
  const sY = frames_Kinect[0][`joint_${jointId}_y`] || 0;
  const sZ = frames_Kinect[0][`joint_${jointId}_z`] || 0;

  for (let i = 1; i < frames_Kinect.length; i++) {
    const f1 = frames_Kinect[i-1], f2 = frames_Kinect[i];
    const dt = ((f2.timestamp_usec || 0) - (f1.timestamp_usec || 0)) / 1000000;
    const x1 = f1[`joint_${jointId}_x`] || 0;
    const y1 = f1[`joint_${jointId}_y`] || 0;
    const z1 = f1[`joint_${jointId}_z`] || 0;
    const x2 = f2[`joint_${jointId}_x`] || 0;
    const y2 = f2[`joint_${jointId}_y`] || 0;
    const z2 = f2[`joint_${jointId}_z`] || 0;

    const dist = Math.sqrt(Math.pow(x2-x1, 2) + Math.pow(y2-y1, 2) + Math.pow(z2-z1, 2)) / 1000;
    const v = dt > 0 ? dist / dt : 0;
    speeds.push(v);
    accs.push(dt > 0 ? Math.abs(v - (speeds[i-1] || 0)) / dt : 0);
    
    pX.push((-x2 - (-sX)) / 1000);
    pY.push((-y2 - (-sY)) / 1000);
    pZ.push((z2 - sZ) / 1000);
  }

  if (charts_Kinect.speed) {
    charts_Kinect.speed.data.labels = labels;
    charts_Kinect.speed.data.datasets[0].data = speeds;
  }
  if (charts_Kinect.acc) {
    charts_Kinect.acc.data.labels = labels;
    charts_Kinect.acc.data.datasets[0].data = accs;
  }
  if (charts_Kinect.pos) {
    charts_Kinect.pos.data.labels = labels;
    charts_Kinect.pos.data.datasets[0].data = pX;
    charts_Kinect.pos.data.datasets[1].data = pY;
    charts_Kinect.pos.data.datasets[2].data = pZ;
  }
}

// 更新 Kinect 3D 骨骼
function updateThreeSkeleton_Kinect(idx) {
  const frame = frames_Kinect[idx];
  if (!frame) return;

  const jointSelect = document.getElementById('jointSelect_Kinect');
  const targetId = jointSelect ? parseInt(jointSelect.value) : 20;
  
  let minY = Infinity;
  for (let i = 0; i < 32; i++) {
    const y = frame[`joint_${i}_y`];
    if (y !== undefined && -y < minY) minY = -y;
  }
  const yOffset = -minY;

  for (let i = 0; i < 32; i++) {
    const x = frame[`joint_${i}_x`];
    const y = frame[`joint_${i}_y`];
    const z = frame[`joint_${i}_z`];
    if (x !== undefined) {
      const posX = -x;
      const posY = -y + yOffset;
      const posZ = z;

      jointMeshes_Kinect[i].position.set(posX, posY, posZ);
      jointMeshes_Kinect[i].visible = true;
      
      if (i === targetId) {
        highLightBall_Kinect.position.set(posX, posY, posZ);
        highLightBall_Kinect.visible = true;
      }
    } else {
      jointMeshes_Kinect[i].visible = false;
    }
  }

  connections_Kinect.forEach((conn, i) => {
    const j1 = jointMeshes_Kinect[conn[0]], j2 = jointMeshes_Kinect[conn[1]];
    if (j1 && j2 && j1.visible && j2.visible) {
      const posAttr = boneLines_Kinect[i].geometry.attributes.position;
      posAttr.array[0] = j1.position.x;
      posAttr.array[1] = j1.position.y;
      posAttr.array[2] = j1.position.z;
      posAttr.array[3] = j2.position.x;
      posAttr.array[4] = j2.position.y;
      posAttr.array[5] = j2.position.z;
      posAttr.needsUpdate = true;
      boneLines_Kinect[i].visible = true;
    } else {
      if (boneLines_Kinect[i]) boneLines_Kinect[i].visible = false;
    }
  });

  if (jointMeshes_Kinect[0].visible && controls_Kinect) {
    controls_Kinect.target.lerp(jointMeshes_Kinect[0].position, 0.2);
  }
}

// 重置 Kinect 相机
function resetCam_Kinect() {
  if (!camera_Kinect || !controls_Kinect) return;
  if (jointMeshes_Kinect[0] && jointMeshes_Kinect[0].visible) {
    const p = jointMeshes_Kinect[0].position;
    camera_Kinect.position.set(p.x, p.y + 1200, p.z + 3500);
    controls_Kinect.target.copy(p);
  } else {
    camera_Kinect.position.set(0, 1200, 3500);
    controls_Kinect.target.set(0, 0, 0);
  }
  controls_Kinect.update();
}

// Load CSV data for Kinect
function loadCSVData_Kinect(csvText) {
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return;
  
  const headers = lines[0].split(',').map(h => h.trim());
  frames_Kinect = [];
  
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',');
    const frame = {};
    headers.forEach((header, idx) => {
      const val = values[idx] ? values[idx].trim() : '';
      frame[header] = isNaN(parseFloat(val)) ? val : parseFloat(val);
    });
    frames_Kinect.push(frame);
  }
  
  if (frames_Kinect.length > 0) {
    currentIndex_Kinect = 0;
    calculatePhysics_Kinect();
    updateUI_Kinect();
  }
}

// Check all data loaded and update max frames
function checkAllDataLoaded() {
  const footFrames = pressureData.length;
  const imuFrames = motionData_IMU.length;
  const kinectFrames = frames_Kinect.length;
  
  if (footFrames > 0 || imuFrames > 0 || kinectFrames > 0) {
    maxFrames = Math.max(footFrames, imuFrames, kinectFrames);
    const progressSlider = document.getElementById('unified-progress');
    if (progressSlider) {
      progressSlider.max = maxFrames - 1;
    }
    console.log(`All data loaded. Max frames: ${maxFrames}`);
  }
}
