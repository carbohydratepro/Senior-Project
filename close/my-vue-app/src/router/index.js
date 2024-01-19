import { createRouter, createWebHistory } from 'vue-router';
import WordCloud from '../components/WordCloud.vue';
import ResultView from '../components/ResultView.vue';
import CosinSimilarity from '../components/CosinSimilarity.vue';

const routes = [
  {
    path: '/',
    name: 'WordCloud',
    component: WordCloud,
  },
  {
    path: '/result',
    name: 'Result',
    component: ResultView,
    // props: true
  },
  {
    path: '/similarity',
    name: 'Similarity',
    component: CosinSimilarity,
  }
  
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
});

export default router;
