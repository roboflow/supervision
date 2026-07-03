import { onMounted, onUnmounted, ref } from 'vue'

import { fetchHealth } from '@/api/video'

export function useApiHealth(pollMs = 15000) {
  const status = ref<'loading' | 'online' | 'offline'>('loading')
  let timer: ReturnType<typeof setInterval> | undefined

  async function refresh() {
    status.value = 'loading'
    try {
      const health = await fetchHealth()
      status.value = health.status === 'ok' ? 'online' : 'offline'
    } catch {
      status.value = 'offline'
    }
  }

  onMounted(() => {
    void refresh()
    timer = setInterval(() => void refresh(), pollMs)
  })

  onUnmounted(() => {
    if (timer) {
      clearInterval(timer)
    }
  })

  return { status, refresh }
}
