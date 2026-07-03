import type { ProcessingJobRecord } from '@/api/records'

export type { ProcessingJobRecord }

export interface JobCreatedResponse {
  job_id: string
  upload_id: string
  status: string
}

export function jobResultUrl(jobId: string): string {
  return `/api/v1/records/jobs/${jobId}/file`
}

export async function fetchJobRecord(jobId: string): Promise<ProcessingJobRecord> {
  const response = await fetch(`/api/v1/records/jobs/${jobId}`)
  if (!response.ok) {
    throw new Error(`Failed to load job: ${response.status}`)
  }
  return response.json() as Promise<ProcessingJobRecord>
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

export async function pollJobUntilComplete(
  jobId: string,
  onProgress: (job: ProcessingJobRecord) => void,
  signal?: AbortSignal,
): Promise<ProcessingJobRecord> {
  while (true) {
    if (signal?.aborted) {
      throw new Error('任务已取消。')
    }

    const job = await fetchJobRecord(jobId)
    onProgress(job)

    if (job.status === 'completed') {
      return job
    }
    if (job.status === 'failed') {
      throw new Error(job.error_message ?? '处理失败。')
    }

    await sleep(600)
  }
}
