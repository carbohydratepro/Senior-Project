from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/get_words', methods=['GET'])
def get_words():
    words = [
        {"text": "りんご", "weight": 26},
        {"text": "みかん", "weight": 19},
        # 他のデータ...
    ]
    return jsonify(words)


@app.route('/word_clicked', methods=['POST'])
def word_clicked():
    print(request.data)  # 生のリクエストデータを表示
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request, JSON required"}), 400

    data = request.get_json()
    word = data.get('word')

    if not word:
        return jsonify({"status": "error", "message": "No word provided"}), 400

    # クリックされた単語に基づいて処理を行います
    print(f"クリックされたワード: {word}")
    # その後の処理...
    return jsonify({"status": "success", "word": word})

@app.route('/')
def index():
    return render_template('wordcloud.html')

if __name__ == '__main__':
    app.run(debug=True)
