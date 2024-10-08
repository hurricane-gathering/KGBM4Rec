from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Hello, how are you?"

inputs = tokenizer(text, return_tensors='pt')

print(inputs)


outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state

pooler_output = outputs.pooler_output

print(last_hidden_state.shape)  # [batch_size, sequence_length, hidden_size]
print(pooler_output.shape)      # [batch_size, hidden_size]


cls_embedding = last_hidden_state[:, 0, :]  
print(cls_embedding.shape)




class BertClassifier(nn.Module):
    def __init__(self, bert):
        super(BertClassifier, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
       
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
      
        cls_output = outputs.pooler_output
        logits = self.classifier(cls_output)
        return logits




model = BertClassifier(model)
optimizer = Adam(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss()


input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
labels = torch.tensor([1]).unsqueeze(0)  


model.train()
optimizer.zero_grad()

logits = model(input_ids, attention_mask)
loss = loss_fn(logits, labels)

loss.backward()
optimizer.step()

print(f'Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    predictions = torch.argmax(logits, dim=-1)
    print(f'Predicted label: {predictions.item()}')
