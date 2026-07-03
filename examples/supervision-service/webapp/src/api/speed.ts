import { jobResultUrl, pollJobUntilComplete } from '@/api/jobs'

export interface SourcePoint {
  x: number
  y: number
}

export interface SpeedEstimateOptions {
  file: File
  sourcePoints: SourcePoint[]
  targetWidth: number
  targetHeight: number
  confidenceThreshold?: number
  iouThreshold?: number
  onProgress?: (progress: number, currentFrame: number, totalFrames: number) => void
  signal?: AbortSignal
}

export interface SpeedEstimateResult {
  jobId: string
  resultUrl: string
}

export async function estimateSpeed({
  file,
  sourcePoints,
  targetWidth,
  targetHeight,
  confidenceThreshold = 0.3,
  iouThreshold = 0.7,
  onProgress,
  signal,
}: SpeedEstimateOptions): Promise<SpeedEstimateResult> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('source_points', JSON.stringify(sourcePoints))
  formData.append('target_width', String(targetWidth))
  formData.append('target_height', String(targetHeight))
  formData.append('confidence_threshold', String(confidenceThreshold))
  formData.append('iou_threshold', String(iouThreshold))

  const response = await fetch('/api/v1/videos/speed-estimate', {
    method: 'POST',
    body: formData,
    signal,
  })

  if (!response.ok) {
    let detail = `Request failed: ${response.status}`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) {
        detail = payload.detail
      }
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(detail)
  }

  const created = (await response.json()) as { job_id: string }

  await pollJobUntilComplete(
    created.job_id,
    (job) => {
      onProgress?.(job.progress, job.current_frame, job.total_frames)
    },
    signal,
  )

  return {
    jobId: created.job_id,
    resultUrl: jobResultUrl(created.job_id),
  }
}
