import torch
from torch import nn

class MLP_Block(nn.Module):
    def __init__(self, num_features, expansion, dropout):
        super(MLP_Block, self).__init__()
        num_hidden = int(expansion*num_features)
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.Gelu = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.Gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x 

class Token_Mixer(nn.Module):
    def __init__(self, num_patches, num_channels, expansion, dropout):
        super(Token_Mixer, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels)
        self.mlp_block = MLP_Block(num_patches, expansion, dropout)
    
    def forward(self, x):
        initial = x
        x = self.layer_norm(x)
        x = torch.transpose(x, 1, 2)
        x = self.mlp_block(x)
        x = torch.transpose(x, 1, 2)
        output = initial + x
        return output

class Channel_Mixer(nn.Module):
    def __init__(self, num_channels, expansion, dropout):
        super(Channel_Mixer, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels)
        self.mlp_block = MLP_Block(num_channels, expansion, dropout)
    
    def forward(self, x):
        initial = x
        x = self.layer_norm(x)
        x = self.mlp_block(x)
        output = initial + x
        return output

class Mixer_Layer(nn.Module):
    def __init__(self, num_patches, num_channels, expansion_token, expansion_channel,dropout):
        super(Mixer_Layer, self).__init__()

        self.token_mixer = Token_Mixer( num_patches, num_channels, expansion_token, dropout)
        self.channel_mixer = Channel_Mixer(num_channels, expansion_channel, dropout)

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x

class MLP_Mixer(nn.Module):
    def __init__(self, image_shape : tuple, 
                 patch_size: int,
                 num_classes, 
                 num_mixers, 
                 num_features, 
                 expansion_token=4,
                 expansion_channel=0.5, 
                 dropout=0.5):
        
        super(MLP_Mixer, self).__init__()
        
        if len(image_shape)==2:
            in_channel = 1 
        elif len(image_shape):
            in_channel = image_shape[2]

        assert image_shape[0] % patch_size == 0
        num_patches = (image_shape[0]//patch_size)**2

        #this conv layer is only for breaking the image into patches of latent dim size
        self.patch_breaker = nn.Conv2d(in_channel, num_features, kernel_size=patch_size, stride=patch_size)

        layers=[]
        for _ in range(num_mixers):
            layers.append(Mixer_Layer(num_patches, 
                                    num_features,
                                    expansion_token,
                                    expansion_channel, 
                                    dropout))
            
        self.mixer_layers = nn.Sequential(*layers)

        self.final_fc = nn.Linear(num_features , num_classes)

    def forward(self, x):
        patches = self.patch_breaker(x)
        batch_size, num_features, h, w = patches.shape
        patches = patches.permute(0,2,3,1)
        patches = patches.view(batch_size, -1, num_features)

        patches = self.mixer_layers(patches)
                    
        outputs = torch.mean(patches, dim=1)
        outputs = self.final_fc(outputs)

        return outputs


if __name__=="__main__":
    model = MLP_Mixer(image_shape=(224,224,3), 
                  patch_size = 16,
                  num_classes=10, 
                  num_mixers=8, 
                  num_features=512)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img = torch.randn((1,3, 224,224),dtype = torch.float32, device = device)

    preds = model(img)
    print(preds)

