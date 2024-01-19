<template>
  <h1>progress</h1>
  <div v-if="processing">
    処理中です... しばらくお待ちください。
    <progress :value="progress" max="100"></progress>
  </div>
</template>

<script>
export default {
  props: {
    processing: Boolean,
    progress: Number
  }
};
</script>
