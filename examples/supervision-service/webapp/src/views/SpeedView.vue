<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue'

import { estimateSpeed } from '@/api/speed'
import type { SourcePoint } from '@/api/speed'
import CalibrationCanvas from '@/components/CalibrationCanvas.vue'
import ProcessingState from '@/components/ProcessingState.vue'
import ResultPanel from '@/components/ResultPanel.vue'
import VideoDropzone from '@/components/VideoDropzone.vue'

type Phase = 'upload' | 'calibrate' | 'processing' | 'result'

const phase = ref<Phase>('upload')
const selectedFile = ref<File | null>(null)
const frameUrl = ref('')
const videoWidth = ref(0)
const videoHeight = ref(0)
const sourcePoints = ref<SourcePoint[]>([])
const targetWidth = ref(25)
const targetHeight = ref(250)
const confidenceThreshold = ref(0.3)
const iouThreshold = ref(0.7)
const errorMessage = ref('')
const resultVideoUrl = ref('')
const progress = ref(0)
const currentFrame = ref(0)
const totalFrames = ref(0)

function revokeFrameUrl() {
  if (frameUrl.value) {
    URL.revokeObjectURL(frameUrl.value)
    frameUrl.value = ''
  }
}

function captureFirstFrame(file: File): Promise<void> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    const url = URL.createObjectURL(file)
    video.preload = 'metadata'
    video.src = url
    video.muted = true
    video.playsInline = true

    video.onloadeddata = () => {
      videoWidth.value = video.videoWidth
      videoHeight.value = video.videoHeight
      video.currentTime = 0
    }

    video.onseeked = () => {
      const canvas = document.createElement('canvas')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const context = canvas.getContext('2d')
      if (!context) {
        URL.revokeObjectURL(url)
        reject(new Error('无法读取视频帧。'))
        return
      }
      context.drawImage(video, 0, 0)
      canvas.toBlob((blob) => {
        URL.revokeObjectURL(url)
        if (!blob) {
          reject(new Error('无法生成预览帧。'))
          return
        }
        revokeFrameUrl()
        frameUrl.value = URL.createObjectURL(blob)
        resolve()
      }, 'image/jpeg', 0.92)
    }

    video.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('视频无法播放，请更换格式。'))
    }
  })
}

async function onSelect(file: File) {
  errorMessage.value = ''
  selectedFile.value = file
  sourcePoints.value = []
  phase.value = 'upload'

  try {
    await captureFirstFrame(file)
    phase.value = 'calibrate'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '读取视频失败。'
    selectedFile.value = null
  }
}

function onPointsUpdate(points: SourcePoint[]) {
  sourcePoints.value = points
}

async function runEstimation() {
  if (!selectedFile.value) {
    errorMessage.value = '请先上传视频。'
    return
  }
  if (sourcePoints.value.length !== 4) {
    errorMessage.value = '请在画面上标记四个路面角点。'
    return
  }

  phase.value = 'processing'
  errorMessage.value = ''
  progress.value = 0
  currentFrame.value = 0
  totalFrames.value = 0
  resultVideoUrl.value = ''

  try {
    const result = await estimateSpeed({
      file: selectedFile.value,
      sourcePoints: sourcePoints.value,
      targetWidth: targetWidth.value,
      targetHeight: targetHeight.value,
      confidenceThreshold: confidenceThreshold.value,
      iouThreshold: iouThreshold.value,
      onProgress: (nextProgress, frame, total) => {
        progress.value = nextProgress
        currentFrame.value = frame
        totalFrames.value = total
      },
    })
    resultVideoUrl.value = result.resultUrl
    phase.value = 'result'
  } catch (error) {
    phase.value = 'calibrate'
    errorMessage.value = error instanceof Error ? error.message : '测速处理失败。'
  }
}

function startOver() {
  revokeFrameUrl()
  selectedFile.value = null
  sourcePoints.value = []
  resultVideoUrl.value = ''
  phase.value = 'upload'
  errorMessage.value = ''
}

onBeforeUnmount(() => {
  revokeFrameUrl()
})
</script>

<template>
  <div class="workspace page">
    <header class="page-header">
      <div>
        <p class="eyebrow">Speed Module</p>
        <h1>路面标定测速</h1>
        <p class="lede">标定路面四边形，将透视画面中的位移换算为 km/h。</p>
      </div>
    </header>

    <div class="grid">
      <section class="feed">
        <VideoDropzone v-if="phase === 'upload'" @select="onSelect" />

        <CalibrationCanvas
          v-else-if="frameUrl && phase !== 'result'"
          :frame-url="frameUrl"
          :video-width="videoWidth"
          :video-height="videoHeight"
          @update="onPointsUpdate"
        />

        <ProcessingState
          v-if="phase === 'processing'"
          message="正在检测、跟踪并计算速度…"
          :progress="progress"
          :current-frame="currentFrame"
          :total-frames="totalFrames"
        />

        <ResultPanel
          v-if="phase === 'result' && resultVideoUrl"
          :url="resultVideoUrl"
          download-name="speed_result.mp4"
          title="测速结果"
        />
      </section>

      <aside class="controls">
        <div class="block">
          <h2>标定参数</h2>
          <p class="hint">四边形对应真实路面的宽与长（米）</p>
          <label class="field">
            <span>路面宽度 (m)</span>
            <input v-model.number="targetWidth" type="number" min="1" step="1" />
          </label>
          <label class="field">
            <span>路面长度 (m)</span>
            <input v-model.number="targetHeight" type="number" min="1" step="1" />
          </label>
        </div>

        <div class="block">
          <h2>检测阈值</h2>
          <label class="field">
            <span>置信度 {{ confidenceThreshold.toFixed(2) }}</span>
            <input v-model.number="confidenceThreshold" type="range" min="0" max="1" step="0.05" />
          </label>
          <label class="field">
            <span>IOU {{ iouThreshold.toFixed(2) }}</span>
            <input v-model.number="iouThreshold" type="range" min="0" max="1" step="0.05" />
          </label>
        </div>

        <dl class="stats">
          <div>
            <dt>分辨率</dt>
            <dd>{{ videoWidth && videoHeight ? `${videoWidth}×${videoHeight}` : '—' }}</dd>
          </div>
          <div>
            <dt>标定点</dt>
            <dd>{{ sourcePoints.length }} / 4</dd>
          </div>
        </dl>

        <div class="actions">
          <button
            type="button"
            class="primary"
            :disabled="phase === 'processing' || sourcePoints.length !== 4 || !selectedFile"
            @click="runEstimation"
          >
            {{ phase === 'processing' ? '处理中…' : '开始测速' }}
          </button>
          <button type="button" class="ghost" @click="startOver">换一段视频</button>
        </div>

        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.page {
  padding: 2rem clamp(1rem, 4vw, 2.5rem) 3rem;
}

.page-header {
  margin-bottom: 1.5rem;
}

.eyebrow {
  margin: 0 0 0.5rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--marking);
}

.page-header h1 {
  margin: 0 0 0.5rem;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: clamp(1.8rem, 3vw, 2.4rem);
}

.lede {
  margin: 0;
  color: var(--text-muted);
}

.grid {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(260px, 0.9fr);
  gap: 1rem;
  align-items: start;
}

.feed,
.controls {
  border: 1px solid var(--line);
  background: var(--bg-panel);
  padding: 1rem;
}

.controls {
  display: grid;
  gap: 1rem;
}

.block h2 {
  margin: 0 0 0.35rem;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1rem;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.hint {
  margin: 0 0 0.75rem;
  color: var(--text-muted);
  font-size: 0.82rem;
}

.field {
  display: grid;
  gap: 0.35rem;
  margin-bottom: 0.75rem;
}

.field span {
  font-size: 0.82rem;
  color: var(--text-muted);
}

.field input[type='number'] {
  padding: 0.55rem 0.65rem;
  border: 1px solid rgba(139, 148, 158, 0.45);
  background: var(--bg-base);
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
}

.stats {
  margin: 0;
  display: grid;
  gap: 0.55rem;
}

.stats div {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: 0.45rem;
  border-bottom: 1px solid rgba(139, 148, 158, 0.2);
}

.stats dt {
  font-size: 0.78rem;
  color: var(--text-muted);
}

.stats dd {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: var(--marking);
}

.actions {
  display: grid;
  gap: 0.65rem;
}

.primary,
.ghost {
  padding: 0.75rem 1rem;
  border: 0;
  cursor: pointer;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.primary {
  background: var(--accent);
  color: var(--bg-base);
}

.primary:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.ghost {
  background: transparent;
  border: 1px solid rgba(139, 148, 158, 0.45);
  color: var(--text-muted);
}

.error {
  margin: 0;
  color: var(--danger);
  font-size: 0.875rem;
}

@media (max-width: 900px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
