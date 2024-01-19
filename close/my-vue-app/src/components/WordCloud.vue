<template>
  <div>
    <!-- 新しいワードを追加する入力フィールド -->
    <input v-model="newWord" @keyup.enter="addWord" placeholder="新しいワードを追加">
    <button @click="addWord">追加</button>

    <!-- 選択された単語を表示する領域 -->
    <div class="selected-words">
      <span v-for="(word, index) in selectedWords" :key="index" class="selected-word" @click="removeWord(word)">
        {{ word }}
      </span>
    </div>

  
    <div>
      <!-- 生成ボタン -->
      <button @click="sendWordsToPython" class="send-button">生成</button>
      <ProgressView :processing="processing" :progress="progress" />
    </div>
    <div class="wordcloud">
      <vue-word-cloud :words="words" :color="wordColor" :size="[10, 30]" font-family="Roboto">
        <template v-slot="{text, weight}">
          <div :title="weight" style="cursor: pointer;" @click="handleWordClick(text, weight)">
            {{ text }}
          </div>
        </template>
      </vue-word-cloud>
    </div>
  </div>
</template>

<script>
import VueWordCloud from 'vuewordcloud';
import axios from 'axios';
import ProgressView from './ProgressView.vue';

export default {
  components: {
    'vue-word-cloud': VueWordCloud,
    ProgressView
  },
  data() {
    return {
      words: [],
      selectedWords: [], // 選択された単語を格納する配列
      newWord: '' // 新しいワードの入力値
    };
  },
  created() {
    axios.get('http://localhost:5000/get_words')
      .then(response => {
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
    handleWordClick(text) {
      const index = this.selectedWords.indexOf(text);
      if (index === -1) {
        this.selectedWords.push(text);
      } else {
        this.selectedWords.splice(index, 1);
      }
      // その他の処理...
    },
    addWord() {
      if (this.newWord && !this.selectedWords.includes(this.newWord)) {
        // 新しい単語を選択された単語リストに追加
        this.selectedWords.push(this.newWord);
        this.newWord = ''; // 入力フィールドをクリア
      }
    },
    removeWord(word) {
      const index = this.selectedWords.indexOf(word);
      if (index !== -1) {
        this.selectedWords.splice(index, 1);
      }
    },
    sendWordsToPython() {
      axios.post('http://localhost:5000/send_words', { words: this.selectedWords })
        .then(response => {
          this.$router.push({
            name: 'Result',
            query: {
              command: response.data.command,
              error: response.data.error
            }
          });
        })
        .catch(error => {
          console.error('Error sending words:', error);
        });
    }
  }
};
</script>

<style scoped>
  .controls {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
  }

  .input-field {
    flex-grow: 1;
    margin-right: 10px;
    padding: 10px;
    font-size: 16px;
  }

  .selected-words {
    flex-grow: 2;
    padding: 10px;
    background-color: #f0f0f0;
    margin-right: 10px;
    overflow: auto;
  }

  .selected-word {
    margin-right: 10px;
    padding: 5px;
    background-color: #e0e0e0;
    border-radius: 5px;
  }

  .send-button {
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
  }

  .wordcloud {
    height: 100vh;  /* ビューポートの高さの100% */
    width: 100vw;   /* ビューポートの幅の100% */
  }
</style>