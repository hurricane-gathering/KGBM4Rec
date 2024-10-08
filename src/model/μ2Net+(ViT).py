import torch
import torch.nn as nn
import torch.nn.functional as F

class µ2NetPlus(nn.Module):
    def __init__(self, vit, u2net):
        super(µ2NetPlus, self).__init__()
        self.vit = vit  
        self.u2net = u2net  
        
        self.vit_to_image = nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=1)  

    def forward(self, image):
        vit_inputs = feature_extractor(images=image, return_tensors="pt")
        vit_outputs = self.vit(**vit_inputs).last_hidden_state 
        
        batch_size, num_patches, hidden_size = vit_outputs.shape
        vit_feature_map = vit_outputs.permute(0, 2, 1)  
        
        vit_feature_map = vit_feature_map.view(batch_size, hidden_size, 14, 14)  
        vit_feature_map = self.vit_to_image(vit_feature_map)  
        
        
        u2net_inputs = torch.cat([image, vit_feature_map], dim=1)  
        
        
        u2net_outputs = self.u2net(u2net_inputs)
        return u2net_outputs

# 使用示例
vit_model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
u2net_model = U2NET()  

model = µ2NetPlus(vit_model, u2net_model)


image = Image.open('test_image.jpg')
output = model(image)
