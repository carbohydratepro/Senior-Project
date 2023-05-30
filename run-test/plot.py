import matplotlib.pyplot as plt
import re

# 入力文字列
input_string = """
Epoch: 0 Loss: 5.524284362792969 Accuracy: 0.9233984375
Epoch: 1 Loss: 0.8242585062980652 Accuracy: 0.99244140625
Epoch: 2 Loss: 0.22770734131336212 Accuracy: 0.99244140625
Epoch: 3 Loss: 0.15066586434841156 Accuracy: 0.99244140625
Epoch: 4 Loss: 0.1498841643333435 Accuracy: 0.99244140625
Epoch: 5 Loss: 0.12495490908622742 Accuracy: 0.99244140625
Epoch: 6 Loss: 0.09421853721141815 Accuracy: 0.99244140625
Epoch: 7 Loss: 0.07726828753948212 Accuracy: 0.99244140625
Epoch: 8 Loss: 0.17037612199783325 Accuracy: 0.99244140625
Epoch: 9 Loss: 0.08920824527740479 Accuracy: 0.99244140625
"""

# 正規表現でデータを抽出
loss_pattern = r"Loss: (\d+\.\d+)"
accuracy_pattern = r"Accuracy: (\d+\.\d+)"

losses = [float(match) for match in re.findall(loss_pattern, input_string)]
accuracies = [float(match) for match in re.findall(accuracy_pattern, input_string)]
epochs = list(range(len(losses)))

# グラフを作成
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epochs, losses, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(epochs, accuracies, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()

# グラフを保存
plt.savefig(f".\\run-test\\figure\\figure2.png")
