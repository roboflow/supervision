export interface UploadRecord {
  id: string
  original_filename: string
  stored_filename: string
  file_path: string
  file_size: number
  content_type: string | null
  created_at: string
}

export interface ProcessingJobRecord {
  id: string
  upload_id: string
  job_type: string
  status: string
  output_path: string | null
  parameters: Record<string, unknown>
  error_message: string | null
  progress: number
  current_frame: number
  total_frames: number
  created_at: string
  completed_at: string | null
  original_filename: string | null
}

export interface UploadListResponse {
  items: UploadRecord[]
  total: number
}

export interface ProcessingJobListResponse {
  items: ProcessingJobRecord[]
  total: number
}

export async function fetchUploadRecords(limit = 50): Promise<UploadListResponse> {
  const response = await fetch(`/api/v1/records/uploads?limit=${limit}`)
  if (!response.ok) {
    throw new Error(`Failed to load upload records: ${response.status}`)
  }
  return response.json() as Promise<UploadListResponse>
}

export async function fetchJobRecords(limit = 50): Promise<ProcessingJobListResponse> {
  const response = await fetch(`/api/v1/records/jobs?limit=${limit}`)
  if (!response.ok) {
    throw new Error(`Failed to load job records: ${response.status}`)
  }
  return response.json() as Promise<ProcessingJobListResponse>
}

export function uploadFileUrl(uploadId: string): string {
  return `/api/v1/records/uploads/${uploadId}/file`
}

export function jobResultUrl(jobId: string): string {
  return `/api/v1/records/jobs/${jobId}/file`
}
