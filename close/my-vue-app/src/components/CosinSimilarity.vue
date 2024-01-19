<template>
    <div class="cosine-similarity">
      <h2>比較元のタイトルと概要</h2>
      <p>タイトル: {{ sourceTitle }}</p>
      <p>概要: </p>
      <div class="preformatted-text">{{ sourceSummary }}</div>
      <p>研究室の上位10個との一致度: {{ sim_ave[0] }}</p>
      <p>研究室の上位10%との一致度: {{ sim_ave[1] }}</p>
      <p>研究室の上位25%との一致度: {{ sim_ave[2] }}</p>
      <p>研究室の上位50%との一致度: {{ sim_ave[3] }}</p>
      <p>研究室のすべてとの一致度: {{ sim_ave[4] }}</p>
  
      <h2>コサイン類似度の比較結果</h2>
      <ul>
        <li v-for="(result, index) in simResult" :key="index">
          <div>タイトル: {{ result.title }}</div>
          <div>年: {{ result.year }}</div>
          <!-- <div>学部学科: {{ result.department }}</div> -->
          <div>類似度: {{ result.cosineSimilarity }}</div>
          <button @click="toggleDetail(index)">詳細表示</button>
          <p v-if="result.showDetail">概要: {{ result.summary }}</p>
        </li>
      </ul>
    </div>
</template>
  
  
<script>
import axios from 'axios';

export default {
  data() {
    return {
      sourceTitle: '',
      sourceSummary: '',
      sim_ave: [],
      simResult: []
    };
  },
  created() {
    axios.post('http://localhost:5000/response_data', { summary: this.$route.query.sourceSummary })
      .then(response => {
        this.simResult = response.data.sim_result;
        this.sim_ave = response.data.sim_ave;
      })
      .catch(error => {
        console.error('Error fetching words:', error);
      });
    this.sourceTitle = this.$route.query.sourceTitle;
    this.sourceSummary = this.$route.query.sourceSummary;
  },
  methods: {
    toggleDetail(index) {
      this.simResult[index].showDetail = !this.simResult[index].showDetail;
    }
  },
};
</script>


<style scoped>
.cosine-similarity {
  font-family: 'Arial', sans-serif;
  background-color: #fff0f6; /* パステルピンク */
  color: #555; /* ソフトな文字色 */
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  max-width: 600px;
  margin: 20px auto;
}

.cosine-similarity h2 {
  color: #a3d8f4; /* パステルブルー */
}

.cosine-similarity ul {
  list-style-type: none;
  padding: 0;
}

.cosine-similarity li {
  background-color: #e3f2fd; /* ライトブルー */
  margin-bottom: 10px;
  padding: 10px;
  border-radius: 5px;
}

.cosine-similarity button {
  background-color: #e3f2fd;
  border: none;
  padding: 5px 10px;
  border-radius: 5px;
  cursor: pointer;
}

.cosine-similarity button:hover {
  background-color: #bbdefb;
}
</style>
