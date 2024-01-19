from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS
import time
import json

# 自作
from wordcloud import count_words_in_db
from create_command import create_command
from cosin_similarity import calc_sim

app = Flask(__name__)
CORS(app)

@app.route('/get_words', methods=['GET'])
def get_words():
    # words = [
    #     {"text": "りんご", "weight": 26},
    #     {"text": "みかん", "weight": 19},
    #     # 他のデータ...
    # ]
    db_path = './close/db/tuboroxn.db'
    num_words = 100
    words = count_words_in_db(db_path, num_words)
    # print(words)
    return jsonify(words)


@app.route('/send_words', methods=['GET', 'POST'])
def send_words():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request, JSON required"}), 400

    data = request.get_json()
    words = data.get('words')
    # print(words)

    if not words:
        return jsonify({"status": "error", "message": "No word provided"}), 400
    
    elif len(words) < 2:
        return jsonify({"status": "error", "message": "choose more words"}), 400
        

    # for word in words:
    #     time.sleep(1)

    command, error = create_command(words)
    # return Response(generate(), mimetype='text/event-stream')
    return jsonify({"status": "success", "command": command, "error": error})


@app.route('/from_gpt_response', methods=['POST'])
def receive_data():
    data = request.json
    title = data.get('title')
    summary = data.get('summary')

    print(title, summary)

    return jsonify({"status": "success", "title": title, "summary": summary })

@app.route('/response_data', methods=['GET', 'POST'])
def similarity():
    data = request.json
    summary = data.get('summary')

    sim_result, sim_averages = calc_sim(summary)
    sim_ave = []
    for sim_average in sim_averages:
        sim_ave.append(str(round(sim_average*100, 2))+"%")

    return jsonify({
        "status": "success",
        "sim_ave": sim_ave,
        "sim_result": sim_result
        })

@app.route('/')
def index():
    return render_template('wordcloud.html')

if __name__ == '__main__':
    app.run(debug=True)
