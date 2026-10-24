# Claude Code Project Instructions

<!-- Imports AGENTS.md, which contains agent roles, behavioral rules, and coding constraints for this project. -->

@AGENTS.md

## GitHub Actions

- Keep public CI on standard GitHub-hosted runners.
- Cancel superseded PR runs with a workflow-specific concurrency group. Serialize documentation publishing without cancelling active pushes.
- Filter link checks to Markdown and their configuration. Filter GitHub configuration tests to their scripts, fixtures, documentation, and dependency inputs. Keep required checks triggered for every PR so path filters cannot block merging.
- Cover every supported Python version on Linux, and the oldest and newest versions on macOS and Windows. Preserve OpenCV backend and isolated wheel coverage on every OS.
- Set a timeout on every runner job, including reusable workflows and result guardians.
- Cache dependencies in test workflows. Disable caching when building or publishing releases so publishing jobs cannot restore caches from test runs. Retain distribution artifacts for seven days.
- Pin actions to full commit SHAs with release-version comments. Keep the GitHub Actions Dependabot ecosystem enabled.
- Default workflow permissions to read-only and elevate individual jobs only when needed. Disable checkout credential persistence unless that checkout must push.
- Pass event values through environment variables and quote shell expansions. Never interpolate branch names, dispatch inputs, or release metadata into shell code.
- Keep the conflict labeler API-only. A `pull_request_target` job must never check out or execute PR code.
