<script setup lang="ts">
import { RouterLink } from 'vue-router'

const features = [
  {
    to: '/track',
    title: '检测与跟踪',
    summary: '上传视频，自动识别目标并分配跟踪 ID，输出带标注的结果视频。',
    tag: 'YOLO + ByteTrack',
  },
  {
    to: '/speed',
    title: '路面标定测速',
    summary: '在监控画面上标定路面区域，将像素位移换算为 km/h 速度读数。',
    tag: 'Perspective + Speed',
  },
  {
    to: '/history',
    title: '历史记录',
    summary: '查看上传与解析任务记录，预览并下载原始视频和处理结果。',
    tag: 'SQLite Records',
  },
]

const steps = [
  '选择功能模块',
  '上传监控视频',
  '配置参数（测速需标定四点）',
  '在线预览并下载结果',
]
</script>

<template>
  <div class="dashboard page">
    <header class="hero">
      <p class="eyebrow">Highway Vision Console</p>
      <h1>Supervision 视频分析控制台</h1>
      <p class="lede">
        基于 supervision 的 Web 端工作台。上传视频、在线处理、预览结果，无需命令行。
      </p>
    </header>

    <section class="feature-grid">
      <RouterLink
        v-for="feature in features"
        :key="feature.to"
        :to="feature.to"
        class="feature-card"
      >
        <span class="tag">{{ feature.tag }}</span>
        <h2>{{ feature.title }}</h2>
        <p>{{ feature.summary }}</p>
        <span class="cta">进入模块 →</span>
      </RouterLink>
    </section>

    <section class="workflow">
      <h2>使用流程</h2>
      <ol>
        <li v-for="(step, index) in steps" :key="step">
          <span class="index">{{ index + 1 }}</span>
          <span>{{ step }}</span>
        </li>
      </ol>
    </section>
  </div>
</template>

<style scoped>
.page {
  padding: 2rem clamp(1rem, 4vw, 2.5rem) 3rem;
}

.hero {
  max-width: 44rem;
  margin-bottom: 2rem;
}

.eyebrow {
  margin: 0 0 0.75rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--marking);
}

.hero h1 {
  margin: 0 0 0.75rem;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: clamp(2rem, 4vw, 3rem);
  line-height: 0.95;
}

.lede {
  margin: 0;
  color: var(--text-muted);
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.feature-card {
  display: grid;
  gap: 0.65rem;
  padding: 1.25rem;
  border: 1px solid var(--line);
  background: var(--bg-panel);
  text-decoration: none;
  color: inherit;
}

.feature-card:hover {
  border-color: rgba(255, 107, 44, 0.45);
  background: var(--bg-elevated);
}

.tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--marking);
}

.feature-card h2 {
  margin: 0;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.35rem;
  letter-spacing: 0.04em;
}

.feature-card p {
  margin: 0;
  color: var(--text-muted);
  font-size: 0.92rem;
}

.cta {
  margin-top: 0.25rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--accent);
}

.workflow {
  padding: 1.25rem;
  border: 1px solid var(--line);
  background: var(--bg-panel);
}

.workflow h2 {
  margin: 0 0 1rem;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.1rem;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.workflow ol {
  margin: 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 0.65rem;
}

.workflow li {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.index {
  width: 1.5rem;
  height: 1.5rem;
  display: grid;
  place-items: center;
  border: 1px solid var(--line);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--marking);
}
</style>
