import pandas as pd

# 读取CSV文件
df = pd.read_csv('image_features.csv', dtype=str)

filtered_df = df[df.iloc[:, 0].str.startswith('19515')]
print(filtered_df)
