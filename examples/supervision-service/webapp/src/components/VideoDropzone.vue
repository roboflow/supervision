<script setup lang="ts">
const emit = defineEmits<{
  select: [file: File]
}>()

function onChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) {
    emit('select', file)
  }
  input.value = ''
}

function onDrop(event: DragEvent) {
  event.preventDefault()
  const file = event.dataTransfer?.files?.[0]
  if (file && file.type.startsWith('video/')) {
    emit('select', file)
  }
}

function onDragOver(event: DragEvent) {
  event.preventDefault()
}
</script>

<template>
  <label
    class="dropzone"
    @dragover="onDragOver"
    @drop="onDrop"
  >
    <input type="file" accept="video/*" hidden @change="onChange" />
    <span class="icon">▶</span>
    <span class="title">拖入或选择视频</span>
    <span class="hint">mp4 / mov / avi</span>
  </label>
</template>

<style scoped>
.dropzone {
  display: grid;
  place-items: center;
  gap: 0.35rem;
  min-height: 220px;
  padding: 2rem;
  border: 1px dashed rgba(245, 197, 24, 0.45);
  background: rgba(0, 0, 0, 0.2);
  cursor: pointer;
  text-align: center;
}

.dropzone:hover {
  background: rgba(245, 197, 24, 0.04);
}

.icon {
  width: 2.5rem;
  height: 2.5rem;
  display: grid;
  place-items: center;
  border: 1px solid var(--line);
  border-radius: 50%;
  color: var(--marking);
  font-size: 0.9rem;
}

.title {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.35rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.hint {
  color: var(--text-muted);
  font-size: 0.82rem;
}
</style>
