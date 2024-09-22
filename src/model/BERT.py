from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


# 加载预训练的BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 定义输入文本
text = "Hello, how are you?"

# 对输入文本进行分词并转换为BERT的输入格式
inputs = tokenizer(text, return_tensors='pt')

print(inputs)
# 输出包含 input_ids 和 attention_mask


# 获取 BERT 模型的输出
outputs = model(**inputs)

# last_hidden_state: 每个 token 的输出
last_hidden_state = outputs.last_hidden_state

# pooler_output: [CLS] token 的表示
pooler_output = outputs.pooler_output

print(last_hidden_state.shape)  # [batch_size, sequence_length, hidden_size]
print(pooler_output.shape)      # [batch_size, hidden_size]


# 使用 [CLS] token 的表示作为句子嵌入
cls_embedding = last_hidden_state[:, 0, :]  # 选择第一个token (CLS token)
print(cls_embedding.shape)




class BertClassifier(nn.Module):
    def __init__(self, bert):
        super(BertClassifier, self).__init__()
        self.bert = bert
        # 分类器线性层，将BERT的输出（768维）映射到2个分类
        self.classifier = nn.Linear(bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        # 获取BERT的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] token 的表示进行分类
        cls_output = outputs.pooler_output
        logits = self.classifier(cls_output)
        return logits




# 初始化模型和优化器
model = BertClassifier(model)
optimizer = Adam(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss()

# 假设我们有一个简单的训练数据
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
labels = torch.tensor([1]).unsqueeze(0)  # 假设有一个标签

# 前向传播，计算损失
model.train()
optimizer.zero_grad()

logits = model(input_ids, attention_mask)
loss = loss_fn(logits, labels)

# 反向传播和优化
loss.backward()
optimizer.step()

print(f'Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    predictions = torch.argmax(logits, dim=-1)
    print(f'Predicted label: {predictions.item()}')
