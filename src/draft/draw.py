import matplotlib.pyplot as plt

# 数据
models = ['GPT4Rec', 'InstructRec', 'KAR', 'KGBM']
inference_times = [250, 240, 220, 200]

# 绘制饼状图
plt.figure(figsize=(8, 8))
plt.pie(inference_times, labels=models, autopct='%1.1f%%', startangle=140)
plt.title('Inference Time Distribution of Models on Taobao Dataset')
plt.axis('equal')  # 使饼图为圆形
plt.show()
