import { createRouter, createWebHistory } from 'vue-router'

import AppLayout from '@/layouts/AppLayout.vue'
import DashboardView from '@/views/DashboardView.vue'
import HistoryView from '@/views/HistoryView.vue'
import SpeedView from '@/views/SpeedView.vue'
import TrackView from '@/views/TrackView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      component: AppLayout,
      children: [
        { path: '', name: 'dashboard', component: DashboardView },
        { path: 'track', name: 'track', component: TrackView },
        { path: 'speed', name: 'speed', component: SpeedView },
        { path: 'history', name: 'history', component: HistoryView },
      ],
    },
  ],
})

export default router
