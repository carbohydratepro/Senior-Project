// ResultView.vue
<template>
    <div class="container">
      <div v-if="errorMessage" class="error-message">
        {{ errorMessage }}
      </div>
      <div class="result">
        <h1>以下のプロンプトをChatGPTに張り付け、得られた論文のタイトルと概要を入力してください</h1>
        <div class="preformatted-text">{{ command }}</div>
      </div>
      <div class="form-container">
        <form @submit.prevent="sendDataToPython">
          <div class="form-group">
            <label for="title">タイトル:</label>
            <input id="title" v-model="form.title" type="text">
          </div>
          <div class="form-group">
            <label for="summary">概要:</label>
            <textarea id="summary" v-model="form.summary"></textarea>
          </div>
          <button type="submit" class="submit-button">送信</button>
        </form>
      </div>
    </div>
  </template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
        error: '', // エラーメッセージの初期値は空文字
        command: '',
        form: {
            title: '',
            summary: ''
        }
    };
  },
  created() {
    this.command = this.$route.query.command
  },
  methods: {
    sendDataToPython() {
        axios.post('http://localhost:5000/from_gpt_response', this.form)
        .then(response => {
            console.log('Server response:', response.data);
            this.$router.push({
            name: 'Similarity',
            query: {
               sourceTitle: response.data.title,
               sourceSummary: response.data.summary,
              }
          });
        })
        .catch(error => {
            console.error('Error sending data:', error);
        });
    }
    }
};
</script>


<style scoped>
.error-message {
  color: #d32f2f; /* エラーの色 */
  background-color: #ffcdd2;
  padding: 10px;
  margin-bottom: 20px;
  border-radius: 4px;
}

.preformatted-text {
  white-space: pre-line; /* 改行を認識する */
}
.container {
  margin: 0 auto; /* 上下のマージンを0にし、左右のマージンを自動で調整 */
  padding: 20px;  /* 内側の余白を20pxに設定 */
  max-width: 800px; /* コンテナの最大幅を指定 */
  color: #555;
  background-color: #f8f8f8;
  padding: 20px;
  font-family: 'Arial', sans-serif;
  border-radius: 10px;
}

.result h1 {
  color: #f3a683;
}

.form-container {
  background-color: #d5f4fa;
  padding: 20px;
  border-radius: 5px;
  margin-top: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 5px;
  color: #6a89cc;
}

.form-group input,
.form-group textarea {
  width: 100%;
  padding: 10px;
  border: 1px solid #dcdde1;
  border-radius: 4px;
  background-color: #f0f0f0;
}

.form-group textarea {
  height: 100px;
  resize: vertical;
}

.submit-button {
  background-color: #78e08f;
  color: #fff;
  padding: 10px 15px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.submit-button:hover {
  background-color: #60a3bc;
}
</style>