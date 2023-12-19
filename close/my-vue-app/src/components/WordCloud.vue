<template>
  <div class="wordcloud">
    <vue-word-cloud :words="words" :color="wordColor" font-family="Roboto">
      <template v-slot="{text, weight}">
        <div :title="weight" style="cursor: pointer;" @click="handleWordClick(text, weight)">
          {{ text }}
        </div>
      </template>
    </vue-word-cloud>
  </div>
</template>

<script>
import VueWordCloud from 'vuewordcloud';
import axios from 'axios';

export default {
  components: {
    'vue-word-cloud': VueWordCloud
  },
  created() {
    axios.get('http://localhost:5000/get_words')
      .then(response => {
        console.log(response.data);  // 受信データを出力
        this.words = response.data.map(item => [item.text, item.weight]);
      })
      .catch(error => {
        console.error('Error fetching words:', error);
      });
  },
  methods: {
    wordColor() {
      const hue = Math.random() * (240 - 120) + 120;
      const saturation = 100;
      const lightness = 50;
      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    },
    handleWordClick(text, weight) {
      console.log(text, weight);  // クリックされた単語と重みを出力
      axios.post('http://localhost:5000/word_clicked', { word: text })
      .then(response => {
        console.log('Server response:', response.data);
      })
      .catch(error => {
        console.error('Error on word click:', error);
      });
    }
  },
  data() {
    return {
      words: [],
    };
  },
};
</script>

<style scoped>
  .wordcloud {
    height: 480px;
    width: 640px;
  }
</style>
