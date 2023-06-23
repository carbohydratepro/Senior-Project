import pandas as pd
import numpy as np

# ランダムなデータを生成
np.random.seed(10)  # 再現性のためにseedを設定
data = {'Category 1': np.random.normal(100, 10, 200),
        'Category 2': np.random.normal(90, 20, 200),
        'Category 3': np.random.normal(80, 30, 200),
        'Category 4': np.random.normal(70, 40, 200)}

df = pd.DataFrame(data)

import matplotlib.pyplot as plt

# 箱ひげ図を作成
plt.boxplot([df['Category 1'], df['Category 2'], df['Category 3'], df['Category 4']], labels=['Category 1', 'Category 2', 'Category 3', 'Category 4'])

plt.title('Boxplot of Categories')  # タイトルを設定
plt.ylabel('Value')  # y軸のラベルを設定
plt.show()  # 箱ひげ図を表示
