import matplotlib.pyplot as plt
from create_bert_model import eval
from datasets import read_data

def calc():
    data_num_def = [100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000, 12000, 15000]
    accuracy_rate = []
    for i, data_num in enumerate(data_num_def):
        test_data_num = int(data_num * 0.2)
        # データセットの読み込み
        datasets = read_data(data_num)
        rate = eval(datasets, test_data_num)
        accuracy_rate.append(rate)
        print("processing status... ", i+1, "/", len(data_num_def))


    return data_num_def, accuracy_rate

    


def plot(data_num, accuracy_rate):

    # グラフを作成
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('data_num')
    ax1.set_ylabel('accuracy')
    ax1.plot(data_num, accuracy_rate, color=color)

    fig.tight_layout()

    # グラフを保存
    plt.savefig(f".\\run-test\\figure\\evaluate_bert_model.png")

if __name__ == "__main__":
    data_num, accuracy_rate = calc()
    plot(data_num, accuracy_rate)
