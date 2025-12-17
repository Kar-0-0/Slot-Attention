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

            slots_reshaped = slots.view(1, B*self.n_slots, attn_dim)
            updates_reshaped = updates.view(1, B*self.n_slots, attn_dim)

            _, h_0 = self.gru(updates_reshaped, slots_reshaped)
            slots = h_0.squeeze(0).view(B, self.n_slots, attn_dim)  

        return slots
    

class Decoder(nn.Module):
    def __init__(self, attn_dim, height, width):
        super().__init__()
        self.height = height
        self.width = width
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
        x_grid, y_grid = torch.meshgrid(xs, ys, indexing="ij")
        xy_grid = torch.stack([x_grid, y_grid], dim=-1) # (H, W, 2)
        self.register_buffer("xy_grid", xy_grid)
        self.conv_tower = nn.Sequential(
            nn.Conv2d(attn_dim + 2, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2), 
            nn.ReLU(),
            nn.Conv2d(64, 4, 3, padding=1)  # (B, H, W, 4)
        )

    def forward(self, x):
        B, n_slots, D = x.shape
        x = x[:, :, :, None, None]
        x = x.expand((B, n_slots, D, self.height, self.width))
        xy_grid = self.xy_grid.permute(2, 0, 1)[None, None, :, :, :]
        feats = torch.cat([x, xy_grid], dim=2) # (B, n_slots, D+2, H, W)
        x = feats.view(B*n_slots, D+2, self.height, self.width)
        full_x = self.conv_tower(x)
        full_x = full_x.view(B, n_slots, 4, self.height, self.width)
        full_x = full_x.permute(0, 1, 3, 4, 2)
        rgb = full_x[:, :, :, :, 0:3] # (B, n_slots, H, W, 3)
        mask = full_x[:, :, :, :, 3:4] # (B, n_slots, H, W, 1)
        mask = F.softmax(mask, dim=1)
        rgb = mask * rgb
        rgb = rgb.sum(dim=1)

        return rgb


class ReconModel(nn.Module):
    def __init__(
            self,
            feature_dim,
            attn_dim, 
            n_slots, 
            n_heads, 
            t_iters,
            height, 
            width
    ):
        super().__init__()
        self.encoder = Encoder()
        self.slot_attention = SlotAttention(
            feature_dim,
            attn_dim,
            n_slots,
            n_heads,
            t_iters
        )
        self.decoder = Decoder(
            attn_dim,
            height,
            width
        )

    def forward(self, x):
        encoding = self.encoder(x)
        slots = self.slot_attention(encoding)
        img = self.decoder(slots)

        return img