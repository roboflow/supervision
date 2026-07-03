<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'

import type { SourcePoint } from '@/api/speed'

const props = defineProps<{
  frameUrl: string
  videoWidth: number
  videoHeight: number
}>()

const emit = defineEmits<{
  update: [points: SourcePoint[]]
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const points = ref<SourcePoint[]>([])
const image = new Image()

const pointLabels = ['远端左', '远端右', '近端右', '近端左']

function drawFrame() {
  const canvas = canvasRef.value
  if (!canvas || !image.complete) {
    return
  }

  const context = canvas.getContext('2d')
  if (!context) {
    return
  }

  const width = canvas.clientWidth
  const height = Math.round((props.videoHeight / props.videoWidth) * width)
  canvas.width = width
  canvas.height = height

  context.drawImage(image, 0, 0, width, height)

  context.strokeStyle = 'rgba(245, 197, 24, 0.35)'
  context.lineWidth = 1
  context.strokeRect(12, 12, width - 24, height - 24)

  const corners: [number, number][] = [
    [12, 12],
    [width - 12, 12],
    [width - 12, height - 12],
    [12, height - 12],
  ]
  const bracket = 18
  context.strokeStyle = '#f5c518'
  context.lineWidth = 2
  for (const [x, y] of corners) {
    const dx = x < width / 2 ? 1 : -1
    const dy = y < height / 2 ? 1 : -1
    context.beginPath()
    context.moveTo(x, y + dy * bracket)
    context.lineTo(x, y)
    context.lineTo(x + dx * bracket, y)
    context.stroke()
  }

  if (points.value.length > 1) {
    context.beginPath()
    const scaled = points.value.map(toDisplay)
    context.moveTo(scaled[0]!.x, scaled[0]!.y)
    for (let index = 1; index < scaled.length; index += 1) {
      context.lineTo(scaled[index]!.x, scaled[index]!.y)
    }
    if (points.value.length === 4) {
      context.closePath()
      context.fillStyle = 'rgba(255, 107, 44, 0.12)'
      context.fill()
    }
    context.strokeStyle = '#ff6b2c'
    context.lineWidth = 2
    context.stroke()
  }

  points.value.forEach((point, index) => {
    const display = toDisplay(point)
    context.beginPath()
    context.fillStyle = '#ff6b2c'
    context.arc(display.x, display.y, 5, 0, Math.PI * 2)
    context.fill()

    context.font = '600 11px "JetBrains Mono", monospace'
    context.fillStyle = '#f5c518'
    context.fillText(`P${index + 1}`, display.x + 8, display.y - 8)
  })
}

function toDisplay(point: SourcePoint) {
  const canvas = canvasRef.value!
  const scaleX = canvas.clientWidth / props.videoWidth
  const scaleY = canvas.height / props.videoHeight
  return {
    x: point.x * scaleX,
    y: point.y * scaleY,
  }
}

function toNative(event: MouseEvent): SourcePoint {
  const canvas = canvasRef.value!
  const rect = canvas.getBoundingClientRect()
  const scaleX = props.videoWidth / rect.width
  const scaleY = props.videoHeight / rect.height
  return {
    x: Math.round((event.clientX - rect.left) * scaleX),
    y: Math.round((event.clientY - rect.top) * scaleY),
  }
}

function onCanvasClick(event: MouseEvent) {
  if (points.value.length >= 4) {
    return
  }
  points.value = [...points.value, toNative(event)]
  emit('update', points.value)
  drawFrame()
}

function resetPoints() {
  points.value = []
  emit('update', [])
  drawFrame()
}

watch(
  () => props.frameUrl,
  (url) => {
    points.value = []
    emit('update', [])
    image.onload = () => drawFrame()
    image.src = url
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  image.onload = null
})

defineExpose({ resetPoints })
</script>

<template>
  <div class="calibration-shell">
    <div class="hud-bar">
      <span class="rec">REC</span>
      <span class="feed-label">CALIBRATION FEED</span>
      <span class="step">
        {{
          points.length < 4
            ? `标记 ${pointLabels[points.length]}`
            : '标定完成'
        }}
      </span>
    </div>
    <canvas
      ref="canvasRef"
      class="calibration-canvas"
      :class="{ complete: points.length >= 4 }"
      @click="onCanvasClick"
    />
    <div class="point-guide">
      <span v-for="(label, index) in pointLabels" :key="label" :class="{ done: index < points.length }">
        {{ label }}
      </span>
    </div>
    <button type="button" class="reset-btn" @click="resetPoints">重新标定</button>
  </div>
</template>

<style scoped>
.calibration-shell {
  display: grid;
  gap: 0.75rem;
}

.hud-bar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--speed-mist);
}

.rec {
  color: #ff4d4d;
  animation: pulse 1.4s ease-in-out infinite;
}

.feed-label {
  color: var(--speed-marking);
}

.step {
  margin-left: auto;
  color: var(--speed-text);
}

.calibration-canvas {
  width: 100%;
  border: 1px solid rgba(245, 197, 24, 0.25);
  cursor: crosshair;
  display: block;
  background: #000;
}

.calibration-canvas.complete {
  cursor: default;
}

.point-guide {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.5rem;
}

.point-guide span {
  padding: 0.35rem 0.5rem;
  border: 1px solid rgba(139, 148, 158, 0.35);
  font-size: 0.75rem;
  text-align: center;
  color: var(--speed-mist);
}

.point-guide span.done {
  border-color: rgba(255, 107, 44, 0.6);
  color: var(--speed-accent);
  background: rgba(255, 107, 44, 0.08);
}

.reset-btn {
  justify-self: start;
  padding: 0.45rem 0.85rem;
  border: 1px solid rgba(245, 197, 24, 0.45);
  background: transparent;
  color: var(--speed-marking);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  cursor: pointer;
}

.reset-btn:hover {
  background: rgba(245, 197, 24, 0.08);
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.45;
  }
}

@media (prefers-reduced-motion: reduce) {
  .rec {
    animation: none;
  }
}
</style>
