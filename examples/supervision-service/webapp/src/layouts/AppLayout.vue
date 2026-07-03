<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'

import ApiStatusBadge from '@/components/ApiStatusBadge.vue'
import { useApiHealth } from '@/composables/useApiHealth'

const { status } = useApiHealth()

const navItems = [
  { to: '/', label: '概览', desc: 'Dashboard' },
  { to: '/track', label: '检测跟踪', desc: 'Track' },
  { to: '/speed', label: '速度估算', desc: 'Speed' },
  { to: '/history', label: '历史记录', desc: 'Records' },
]
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <RouterLink to="/" class="brand">
        <span class="brand-mark">SV</span>
        <span class="brand-text">
          <strong>Supervision</strong>
          <small>Vision Console</small>
        </span>
      </RouterLink>

      <nav class="nav">
        <RouterLink
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          class="nav-link"
        >
          <span class="nav-label">{{ item.label }}</span>
          <span class="nav-desc">{{ item.desc }}</span>
        </RouterLink>
      </nav>

      <footer class="sidebar-footer">
        <ApiStatusBadge :status="status" />
        <a href="/docs" target="_blank" rel="noreferrer">API 文档</a>
      </footer>
    </aside>

    <main class="main">
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.app-shell {
  display: grid;
  grid-template-columns: var(--sidebar-width) minmax(0, 1fr);
  min-height: 100vh;
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  padding: 1.25rem 1rem;
  border-right: 1px solid var(--line);
  background: var(--bg-panel);
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  text-decoration: none;
  color: var(--text);
}

.brand-mark {
  width: 2.25rem;
  height: 2.25rem;
  display: grid;
  place-items: center;
  border: 1px solid var(--line);
  color: var(--marking);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  font-weight: 600;
}

.brand-text {
  display: grid;
  line-height: 1.15;
}

.brand-text strong {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.05rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.brand-text small {
  color: var(--text-muted);
  font-size: 0.72rem;
}

.nav {
  display: grid;
  gap: 0.35rem;
}

.nav-link {
  display: grid;
  gap: 0.1rem;
  padding: 0.65rem 0.75rem;
  border: 1px solid transparent;
  text-decoration: none;
  color: var(--text-muted);
}

.nav-link:hover {
  border-color: var(--line);
  color: var(--text);
}

.nav-link.router-link-exact-active {
  border-color: rgba(255, 107, 44, 0.45);
  background: rgba(255, 107, 44, 0.08);
  color: var(--text);
}

.nav-label {
  font-weight: 600;
  font-size: 0.92rem;
}

.nav-desc {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.sidebar-footer {
  margin-top: auto;
  display: grid;
  gap: 0.75rem;
}

.sidebar-footer a {
  color: var(--text-muted);
  font-size: 0.82rem;
  text-decoration: none;
}

.sidebar-footer a:hover {
  color: var(--marking);
}

.main {
  min-width: 0;
  background:
    radial-gradient(circle at 10% 0%, rgba(255, 107, 44, 0.07), transparent 35%),
    radial-gradient(circle at 90% 10%, rgba(245, 197, 24, 0.05), transparent 30%),
    var(--bg-base);
}

@media (max-width: 860px) {
  .app-shell {
    grid-template-columns: 1fr;
  }

  .sidebar {
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }

  .nav {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }
}
</style>
