const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true
})

// vue.config.js

module.exports = {
  devServer: {
    proxy: {
      '/get_words': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/word_clicked': {
        target: 'http://localhost:5000',
        changeOrigin: true
      }
    }
  }
};
