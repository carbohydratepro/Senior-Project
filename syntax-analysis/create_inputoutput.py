import re
import torch
import nltk
from transformers import BertTokenizer, BertForSequenceClassification
from janome.tokenizer import Tokenizer


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def preprocess_problem_statement(problem_statement, language="en"):
    # 文の分割
    if language == "ja":
        t = Tokenizer()

        sentences = [token.surface for token in t.tokenize(problem_statement)]

    elif language == "en":
        # 初回のみ必要
        # nltk.download('punkt')
        sentences = nltk.sent_tokenize(problem_statement)

    # 入力と出力の例を特定するための正規表現
    input_pattern = re.compile(r"Input:?\s*(?:Example:?)?")
    output_pattern = re.compile(r"Output:?\s*(?:Example:?)?")

    inputs, outputs = [], []

    # 各文に対して、入力または出力の例を抽出
    for sentence in sentences:
        if input_pattern.search(sentence):
            inputs.append(sentence)
        elif output_pattern.search(sentence):
            outputs.append(sentence)

    return inputs, outputs


def generate_examples(inputs, outputs): #入出力の例を生成
    examples = []

    for i in range(len(inputs)):
        input_example = tokenizer(inputs[i], return_tensors="pt")
        output_example = tokenizer(outputs[i], return_tensors="pt")

        # BERTモデルで入力および出力の例を生成
        input_example_prediction = model(**input_example)[0].argmax(dim=1)
        output_example_prediction = model(**output_example)[0].argmax(dim=1)

        examples.append((input_example_prediction, output_example_prediction))

    return examples


def main():

    result = preprocess_problem_statement(problem_txt)
    print(result)

if __name__ == "__main__":
    main()