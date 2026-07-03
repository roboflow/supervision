<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  url: string
  downloadName?: string
  title?: string
}>()

const playbackError = ref(false)

function onVideoError() {
  playbackError.value = true
}
</script>

<template>
  <section class="result-panel">
    <header>
      <h2>{{ title ?? '处理结果' }}</h2>
    </header>
    <video :src="url" controls playsinline @error="onVideoError" />
    <p v-if="playbackError" class="warn">
      浏览器无法直接播放该视频，请点击下方链接下载后用本地播放器查看。
    </p>
    <a :href="url" :download="downloadName ?? 'result.mp4'">下载视频</a>
  </section>
</template>

<style scoped>
.result-panel {
  display: grid;
  gap: 0.75rem;
}

.result-panel h2 {
  margin: 0;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.1rem;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.result-panel video {
  width: 100%;
  background: #000;
  border: 1px solid var(--line);
}

.warn {
  margin: 0;
  color: var(--marking);
  font-size: 0.82rem;
}

.result-panel a {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: var(--marking);
  text-decoration: none;
}

.result-panel a:hover {
  text-decoration: underline;
}
</style>
