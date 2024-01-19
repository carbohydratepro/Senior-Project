<template>
  <div id="app">
    <router-view></router-view>
  </div>
</template>

<script>
import router from './router';

export default {
  name: 'App',
  setup() {
    return {};
  },
  router
};
</script>

<style>
/* ここにスタイルを追加 */
</style>
