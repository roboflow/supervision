<script setup lang="ts">
defineProps<{
  status: 'loading' | 'online' | 'offline'
}>()

const labels = {
  loading: '检查中',
  online: '在线',
  offline: '离线',
} as const
</script>

<template>
  <span class="api-status" :class="status">API {{ labels[status] }}</span>
</template>

<style scoped>
.api-status {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.65rem;
  border-radius: 999px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
}

.api-status::before {
  content: '';
  width: 0.45rem;
  height: 0.45rem;
  border-radius: 50%;
  background: currentColor;
}

.api-status.loading {
  color: var(--text-muted);
  background: rgba(139, 148, 158, 0.12);
}

.api-status.online {
  color: var(--success);
  background: rgba(61, 214, 140, 0.12);
}

.api-status.offline {
  color: var(--danger);
  background: rgba(255, 138, 128, 0.12);
}
</style>
