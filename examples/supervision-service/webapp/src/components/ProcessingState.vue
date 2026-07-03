<script setup lang="ts">
defineProps<{
  message?: string
  progress?: number
  currentFrame?: number
  totalFrames?: number
}>()
</script>

<template>
  <div class="processing">
    <div class="processing-head">
      <span class="spinner" aria-hidden="true" />
      <p>{{ message ?? '正在处理视频，请稍候…' }}</p>
      <span v-if="progress !== undefined" class="percent">{{ progress }}%</span>
    </div>

    <div v-if="progress !== undefined" class="progress-track" role="progressbar" :aria-valuenow="progress" aria-valuemin="0" aria-valuemax="100">
      <div class="progress-fill" :style="{ width: `${progress}%` }" />
    </div>

    <p v-if="totalFrames" class="frames">
      帧进度 {{ currentFrame ?? 0 }} / {{ totalFrames }}
    </p>
  </div>
</template>

<style scoped>
.processing {
  display: grid;
  gap: 0.65rem;
  padding: 0.85rem 1rem;
  border-left: 3px solid var(--accent);
  background: rgba(255, 107, 44, 0.08);
}

.processing-head {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.spinner {
  width: 1rem;
  height: 1rem;
  border: 2px solid rgba(255, 107, 44, 0.25);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  flex-shrink: 0;
}

.processing-head p {
  margin: 0;
  flex: 1;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
}

.percent {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: var(--marking);
}

.progress-track {
  height: 0.45rem;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(245, 197, 24, 0.2);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), var(--marking));
  transition: width 0.25s ease;
}

.frames {
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--text-muted);
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@media (prefers-reduced-motion: reduce) {
  .spinner {
    animation: none;
  }

  .progress-fill {
    transition: none;
  }
}
</style>
