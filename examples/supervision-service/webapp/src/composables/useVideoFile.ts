import { computed, ref } from 'vue'

export function useVideoFile() {
  const file = ref<File | null>(null)
  const previewUrl = ref('')

  const fileMeta = computed(() => {
    if (!file.value) {
      return null
    }
    const sizeMb = (file.value.size / (1024 * 1024)).toFixed(2)
    return {
      name: file.value.name,
      sizeMb,
      type: file.value.type || 'video/*',
    }
  })

  function setFile(next: File | null) {
    if (previewUrl.value) {
      URL.revokeObjectURL(previewUrl.value)
      previewUrl.value = ''
    }
    file.value = next
    if (next) {
      previewUrl.value = URL.createObjectURL(next)
    }
  }

  function clear() {
    setFile(null)
  }

  function revoke() {
    if (previewUrl.value) {
      URL.revokeObjectURL(previewUrl.value)
      previewUrl.value = ''
    }
  }

  return { file, previewUrl, fileMeta, setFile, clear, revoke }
}
