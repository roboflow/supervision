<script setup lang="ts">
import { onMounted, ref } from 'vue'

import {
  fetchJobRecords,
  fetchUploadRecords,
  jobResultUrl,
  uploadFileUrl,
} from '@/api/records'
import type { ProcessingJobRecord, UploadRecord } from '@/api/records'

type Tab = 'jobs' | 'uploads'

const activeTab = ref<Tab>('jobs')
const uploads = ref<UploadRecord[]>([])
const jobs = ref<ProcessingJobRecord[]>([])
const loading = ref(false)
const errorMessage = ref('')
const previewUrl = ref('')

const jobTypeLabels: Record<string, string> = {
  track: '检测跟踪',
  speed: '速度估算',
}

const statusLabels: Record<string, string> = {
  completed: '完成',
  failed: '失败',
  processing: '处理中',
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function formatTime(value: string | null): string {
  if (!value) {
    return '—'
  }
  return value.replace('T', ' ')
}

async function loadRecords() {
  loading.value = true
  errorMessage.value = ''
  try {
    const [uploadData, jobData] = await Promise.all([
      fetchUploadRecords(),
      fetchJobRecords(),
    ])
    uploads.value = uploadData.items
    jobs.value = jobData.items
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '加载历史记录失败。'
  } finally {
    loading.value = false
  }
}

function showPreview(url: string) {
  previewUrl.value = url
}

function closePreview() {
  previewUrl.value = ''
}

onMounted(() => {
  void loadRecords()
})
</script>

<template>
  <div class="history page">
    <header class="page-header">
      <div>
        <p class="eyebrow">Records</p>
        <h1>历史记录</h1>
        <p class="lede">查看已上传的视频与解析任务，支持下载原始文件和处理结果。</p>
      </div>
      <button type="button" class="refresh" :disabled="loading" @click="loadRecords">
        {{ loading ? '刷新中…' : '刷新' }}
      </button>
    </header>

    <div class="tabs">
      <button
        type="button"
        class="tab"
        :class="{ active: activeTab === 'jobs' }"
        @click="activeTab = 'jobs'"
      >
        解析记录 ({{ jobs.length }})
      </button>
      <button
        type="button"
        class="tab"
        :class="{ active: activeTab === 'uploads' }"
        @click="activeTab = 'uploads'"
      >
        上传记录 ({{ uploads.length }})
      </button>
    </div>

    <p v-if="errorMessage" class="error">{{ errorMessage }}</p>

    <section v-if="activeTab === 'jobs'" class="panel">
      <p v-if="!loading && jobs.length === 0" class="empty">暂无解析记录，处理视频后会出现在这里。</p>

      <article v-for="job in jobs" :key="job.id" class="record-card">
        <div class="record-main">
          <div class="record-title">
            <span class="type">{{ jobTypeLabels[job.job_type] ?? job.job_type }}</span>
            <span class="status" :class="job.status">{{ statusLabels[job.status] ?? job.status }}</span>
          </div>
          <p class="filename">{{ job.original_filename ?? '未知文件' }}</p>
          <dl class="meta">
            <div>
              <dt>任务 ID</dt>
              <dd>{{ job.id.slice(0, 8) }}…</dd>
            </div>
            <div>
              <dt>创建时间</dt>
              <dd>{{ formatTime(job.created_at) }}</dd>
            </div>
            <div>
              <dt>完成时间</dt>
              <dd>{{ formatTime(job.completed_at) }}</dd>
            </div>
          </dl>
          <p v-if="job.error_message" class="job-error">{{ job.error_message }}</p>
          <div v-if="job.status === 'processing' && job.total_frames > 0" class="job-progress">
            <div class="progress-track">
              <div class="progress-fill" :style="{ width: `${job.progress}%` }" />
            </div>
            <span class="progress-label">
              {{ job.current_frame }} / {{ job.total_frames }} · {{ job.progress }}%
            </span>
          </div>
        </div>

        <div v-if="job.status === 'completed'" class="record-actions">
          <button type="button" class="ghost" @click="showPreview(jobResultUrl(job.id))">
            预览
          </button>
          <a :href="jobResultUrl(job.id)" download>下载结果</a>
        </div>
      </article>
    </section>

    <section v-else class="panel">
      <p v-if="!loading && uploads.length === 0" class="empty">暂无上传记录。</p>

      <article v-for="upload in uploads" :key="upload.id" class="record-card">
        <div class="record-main">
          <p class="filename">{{ upload.original_filename }}</p>
          <dl class="meta">
            <div>
              <dt>大小</dt>
              <dd>{{ formatSize(upload.file_size) }}</dd>
            </div>
            <div>
              <dt>上传时间</dt>
              <dd>{{ formatTime(upload.created_at) }}</dd>
            </div>
            <div>
              <dt>存储文件</dt>
              <dd>{{ upload.stored_filename }}</dd>
            </div>
          </dl>
        </div>

        <div class="record-actions">
          <button type="button" class="ghost" @click="showPreview(uploadFileUrl(upload.id))">
            预览
          </button>
          <a :href="uploadFileUrl(upload.id)" :download="upload.original_filename">下载原文件</a>
        </div>
      </article>
    </section>

    <div v-if="previewUrl" class="preview-modal" @click.self="closePreview">
      <div class="preview-dialog">
        <header>
          <h2>视频预览</h2>
          <button type="button" class="close" @click="closePreview">关闭</button>
        </header>
        <video :src="previewUrl" controls autoplay />
      </div>
    </div>
  </div>
</template>

<style scoped>
.page {
  padding: 2rem clamp(1rem, 4vw, 2.5rem) 3rem;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
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

.refresh {
  padding: 0.55rem 0.9rem;
  border: 1px solid var(--line);
  background: transparent;
  color: var(--marking);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  cursor: pointer;
}

.refresh:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.tab {
  padding: 0.55rem 0.9rem;
  border: 1px solid var(--line);
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.875rem;
}

.tab.active {
  border-color: rgba(255, 107, 44, 0.45);
  background: rgba(255, 107, 44, 0.08);
  color: var(--text);
}

.panel {
  display: grid;
  gap: 0.75rem;
}

.empty {
  margin: 0;
  padding: 1.25rem;
  border: 1px dashed var(--line);
  color: var(--text-muted);
  text-align: center;
}

.record-card {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding: 1rem;
  border: 1px solid var(--line);
  background: var(--bg-panel);
}

.record-main {
  min-width: 0;
}

.record-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}

.type {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.95rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.status {
  padding: 0.15rem 0.45rem;
  border-radius: 999px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
}

.status.completed {
  color: var(--success);
  background: rgba(61, 214, 140, 0.12);
}

.status.failed {
  color: var(--danger);
  background: rgba(255, 138, 128, 0.12);
}

.status.processing {
  color: var(--marking);
  background: rgba(245, 197, 24, 0.12);
}

.filename {
  margin: 0 0 0.65rem;
  font-weight: 600;
  word-break: break-all;
}

.meta {
  margin: 0;
  display: grid;
  gap: 0.35rem;
}

.meta div {
  display: flex;
  gap: 0.75rem;
  font-size: 0.82rem;
}

.meta dt {
  min-width: 4.5rem;
  color: var(--text-muted);
}

.meta dd {
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
  color: var(--text);
}

.job-error {
  margin: 0.65rem 0 0;
  color: var(--danger);
  font-size: 0.82rem;
}

.job-progress {
  margin-top: 0.65rem;
  display: grid;
  gap: 0.35rem;
}

.progress-track {
  height: 0.35rem;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(245, 197, 24, 0.2);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), var(--marking));
}

.progress-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--text-muted);
}

.record-actions {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  align-items: flex-end;
  flex-shrink: 0;
}

.record-actions a,
.ghost {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  text-decoration: none;
  color: var(--marking);
}

.ghost {
  padding: 0.35rem 0.65rem;
  border: 1px solid rgba(139, 148, 158, 0.45);
  background: transparent;
  cursor: pointer;
}

.error {
  color: var(--danger);
  margin-bottom: 1rem;
}

.preview-modal {
  position: fixed;
  inset: 0;
  z-index: 100;
  display: grid;
  place-items: center;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.72);
}

.preview-dialog {
  width: min(960px, 100%);
  border: 1px solid var(--line);
  background: var(--bg-panel);
  padding: 1rem;
}

.preview-dialog header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.preview-dialog h2 {
  margin: 0;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.1rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.close {
  border: 0;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
}

.preview-dialog video {
  width: 100%;
  background: #000;
}

@media (max-width: 720px) {
  .record-card {
    flex-direction: column;
  }

  .record-actions {
    flex-direction: row;
    align-items: center;
  }
}
</style>
