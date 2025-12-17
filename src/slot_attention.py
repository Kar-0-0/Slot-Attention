import torch 
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(128, 128, 5, stride=2, padding=2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).transpose(-2, -1) # (B, num_inputs, feature_dim)

        return x


class SlotAttention(nn.Module):
    def __init__(self, feature_dim, attn_dim, n_slots, n_heads, t_iters):
        super().__init__() 
        self.n_slots = n_slots
        self.n_heads = n_heads
        self.head_size = attn_dim // self.n_heads
        self.t_iters = t_iters
        self.proj = nn.Linear(feature_dim, attn_dim)
        self.ln = nn.LayerNorm(attn_dim)
        self.q = nn.Linear(attn_dim, attn_dim)
        self.kv = nn.Linear(attn_dim, attn_dim*2)
        self.gru = nn.GRU(attn_dim, attn_dim)
        

    def forward(self, x):
        x = self.ln(self.proj(x))
        B, n_inp, attn_dim = x.shape
        slots = torch.randn((B, self.n_slots, attn_dim), device=x.device)
    
        for _ in range(self.t_iters):
            slot_in = slots
            q = self.q(slot_in) # (B, n_slots, attn_dim)
            kv = self.kv(x) # (B, n_inp, attn_dim
            k, v = kv.split(attn_dim, dim=2) # (B, n_inp, attn_dim)

            q = q.view(B, self.n_slots, self.n_heads, self.head_size).transpose(1, 2) # (B, nh, ns, hs)
            k = k.view(B, n_inp, self.n_heads, self.head_size).transpose(1, 2) # (B, nh, ni, hs)
            v = v.view(B, n_inp, self.n_heads, self.head_size).transpose(1, 2) # (B, nh, ni, hs)

            scores = (q @ k.transpose(-2, -1)) * 1/sqrt(k.size(-1)) # (B, nh, ns, ni)
            scores = F.softmax(scores, dim=-2)

            updates = scores @ v # (B, nh, ns, hs)
            updates = updates.transpose(1, 2).contiguous().view(B, self.n_slots, attn_dim)

            slots_reshaped = slots.view(B*self.n_slots, 1, attn_dim)
            updates_reshaped = updates.view(B*self.n_slots, 1, attn_dim)

            _, h_0 = self.gru(updates_reshaped, slots_reshaped)
            slots = h_0.squeeze(0).view(B, self.n_slots, attn_dim)  

        return slots
    

class Decoder(nn.Module):
    def __init__(self, attn_dim, height, width):
        super().__init__()
        self.rgb_proj = nn.Linear(attn_dim, height*width*3)
        self.mask_proj = nn.Linear(attn_dim, height*width)
        self.height = height
        self.width = width
    
    def forward(self, x):
        B, n_slots, _ = x.shape
        rgb_map = self.rgb_proj(x)
        rgb_map = rgb_map.view(B, n_slots, self.height, self.width, 3)
        mask = self.mask_proj(x)
        mask = mask.view(B, n_slots, self.height, self.width, 1)
        mask_conf = F.softmax(mask, dim=1)
        slot_imgs = mask_conf * rgb_map
        img = slot_imgs.sum(dim=1)
        
        return img
            