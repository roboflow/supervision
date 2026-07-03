<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue'

import { trackVideo } from '@/api/video'
import ProcessingState from '@/components/ProcessingState.vue'
import ResultPanel from '@/components/ResultPanel.vue'
import VideoDropzone from '@/components/VideoDropzone.vue'
import { useVideoFile } from '@/composables/useVideoFile'

const { file, previewUrl, fileMeta, setFile, clear, revoke } = useVideoFile()

const confidenceThreshold = ref(0.3)
const iouThreshold = ref(0.7)
const isProcessing = ref(false)
const errorMessage = ref('')
const resultVideoUrl = ref('')
const progress = ref(0)
const currentFrame = ref(0)
const totalFrames = ref(0)

function onSelect(selected: File) {
  errorMessage.value = ''
  resultVideoUrl.value = ''
  setFile(selected)
}

async function submit() {
  if (!file.value) {
    errorMessage.value = '请先上传视频。'
    return
  }

  isProcessing.value = true
  errorMessage.value = ''
  progress.value = 0
  currentFrame.value = 0
  totalFrames.value = 0
  resultVideoUrl.value = ''

  try {
    const result = await trackVideo({
      file: file.value,
      confidenceThreshold: confidenceThreshold.value,
      iouThreshold: iouThreshold.value,
      onProgress: (nextProgress, frame, total) => {
        progress.value = nextProgress
        currentFrame.value = frame
        totalFrames.value = total
      },
    })
    resultVideoUrl.value = result.resultUrl
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '处理失败。'
  } finally {
    isProcessing.value = false
  }
}

function resetAll() {
  clear()
  errorMessage.value = ''
  resultVideoUrl.value = ''
}

onBeforeUnmount(() => {
  revoke()
})
</script>

<template>
  <div class="workspace page">
    <header class="page-header">
      <div>
        <p class="eyebrow">Track Module</p>
        <h1>检测与跟踪</h1>
        <p class="lede">YOLO 检测 + ByteTrack 多目标跟踪，输出带 ID 标注的视频。</p>
      </div>
    </header>

    <div class="grid">
      <section class="feed">
        <VideoDropzone v-if="!file" @select="onSelect" />

        <template v-else>
          <div class="meta">
            <div>
              <strong>{{ fileMeta?.name }}</strong>
              <span>{{ fileMeta?.sizeMb }} MB</span>
            </div>
            <button type="button" class="ghost" @click="resetAll">更换视频</button>
          </div>
          <video v-if="previewUrl" class="preview" :src="previewUrl" controls />
        </template>

        <ProcessingState
          v-if="isProcessing"
          message="正在检测并跟踪目标…"
          :progress="progress"
          :current-frame="currentFrame"
          :total-frames="totalFrames"
        />
        <ResultPanel
          v-if="resultVideoUrl"
          :url="resultVideoUrl"
          download-name="tracked_result.mp4"
        />
      </section>

      <aside class="controls">
        <div class="block">
          <h2>检测参数</h2>
          <label class="field">
            <span>置信度 {{ confidenceThreshold.toFixed(2) }}</span>
            <input v-model.number="confidenceThreshold" type="range" min="0" max="1" step="0.05" />
          </label>
          <label class="field">
            <span>IOU {{ iouThreshold.toFixed(2) }}</span>
            <input v-model.number="iouThreshold" type="range" min="0" max="1" step="0.05" />
          </label>
        </div>

        <button
          type="button"
          class="primary"
          :disabled="isProcessing || !file"
          @click="submit"
        >
          {{ isProcessing ? '处理中…' : '开始处理' }}
        </button>

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

.meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.75rem;
}

.meta strong {
  display: block;
  font-size: 0.92rem;
}

.meta span {
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
}

.preview {
  width: 100%;
  background: #000;
  border: 1px solid var(--line);
}

.controls {
  display: grid;
  gap: 1rem;
}

.block h2 {
  margin: 0 0 0.75rem;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1rem;
  letter-spacing: 0.07em;
  text-transform: uppercase;
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
