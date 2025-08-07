---
comments: true
---

# Video Utils

<div class="md-typeset">
    <h2><a href="#supervision.video.Video">Video</a></h2>
</div>

:::supervision.video.Video

<div class="md-typeset">
    <h2><a href="#supervision.utils.video.VideoInfo">VideoInfo</a></h2>
</div>

:::supervision.utils.video.VideoInfo

<div class="admonition warning">
<p class="admonition-title">Deprecated</p>
<p><code>VideoSink</code> is deprecated. Use <code>sv.Video(...).sink(...)</code> instead.</p>
</div>

:::supervision.utils.video.VideoSink

<div class="md-typeset">
    <h2><a href="#supervision.utils.video.FPSMonitor">FPSMonitor</a></h2>
</div>

:::supervision.utils.video.FPSMonitor

<div class="admonition warning">
<p class="admonition-title">Deprecated</p>
<p><code>get_video_frames_generator</code> is deprecated. Iterate over <code>sv.Video(source)</code> or use <code>sv.Video(source).frames(...)</code>.</p>
</div>

:::supervision.utils.video.get_video_frames_generator

<div class="admonition warning">
<p class="admonition-title">Deprecated</p>
<p><code>process_video</code> is deprecated. Use <code>sv.Video(source).save(target, callback=...)</code>.</p>
</div>

:::supervision.utils.video.process_video
