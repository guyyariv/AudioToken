import torch.nn as nn
from modules.FGA.atten import Atten

class FGAEmbedder(nn.Module):
    def __init__(self, input_size=768*3, output_size=768):
        super(FGAEmbedder, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.gelu = nn.GELU()
        self.fga = Atten(util_e=[output_size], pairwise_flag=False)

    def forward(self, audio_embs):
        audio_embs = self.fc1(audio_embs)
        audio_embs = self.gelu(audio_embs)
        audio_embs = self.fc2(audio_embs)
        attend = self.fga([audio_embs])[0]
        return attend
