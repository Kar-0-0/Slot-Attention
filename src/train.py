if __name__ == '__main__':
    import torch 
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from multi_object_datasets_torch import MultiDSprites
    import matplotlib.pyplot as plt
    from slot_attention import ReconModel


    def visualize_reconstruction(real_imgs, recon_imgs, epoch, n=4):
        real_imgs = real_imgs.detach().cpu()
        recon_imgs = recon_imgs.detach().cpu()
        
        n = min(n, real_imgs.size(0))
        fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
        
        for i in range(n):
            # Real image
            real_img = real_imgs[i]
            if real_img.shape[0] == 3:  # [C,H,W]
                real_img = real_img.permute(1,2,0)
            axes[0, i].imshow(torch.clamp(real_img,0,1))
            axes[0, i].axis("off")
            if i == n//2:
                axes[0, i].set_title("Real", fontsize=16)
            
            # Reconstruction
            recon_img = recon_imgs[i]
            if recon_img.shape[0] != 3:  # ensure [C,H,W]
                recon_img = recon_img.permute(2,0,1)
            axes[1, i].imshow(torch.clamp(recon_img.permute(1,2,0),0,1))
            axes[1, i].axis("off")
            if i == n//2:
                axes[1, i].set_title("Reconstruction", fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f"recon_{epoch}.png")


    def visualize_slots(real_imgs, slots, decoder, epoch, n=3):
        real_imgs = real_imgs.detach().cpu()
        n = min(n, real_imgs.size(0))
        B, n_slots, _ = slots.shape
        
        # Full reconstruction
        recon_imgs = decoder(slots).detach().cpu()
        if recon_imgs.shape[1] != 3:  # ensure [B,C,H,W]
            recon_imgs = recon_imgs.permute(0, 3, 1, 2)
        
        # Generate per-slot reconstructions
        slot_recons = []
        for slot_idx in range(n_slots):
            slot_only = torch.zeros_like(slots)
            slot_only[:, slot_idx, :] = slots[:, slot_idx, :]
            slot_img = decoder(slot_only).detach().cpu()
            if slot_img.shape[1] != 3:
                slot_img = slot_img.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
            slot_recons.append(slot_img)
        slot_recons = torch.stack(slot_recons, dim=0)  # [n_slots, B, C, H, W]
        
        # Plotting
        fig, axes = plt.subplots(n_slots + 2, n, figsize=(3*n, 3*(n_slots+2)))
        
        for i in range(n):
            # Real image
            axes[0, i].imshow(real_imgs[i].permute(1,2,0))
            axes[0, i].axis("off")
            if i == n // 2:
                axes[0, i].set_title("Real", fontsize=16)
            
            # Full reconstruction
            axes[1, i].imshow(recon_imgs[i].permute(1,2,0))
            axes[1, i].axis("off")
            if i == n // 2:
                axes[1, i].set_title("Reconstruction", fontsize=16)
            
            # Individual slot contributions
            for s in range(n_slots):
                axes[2+s, i].imshow(slot_recons[s, i].permute(1,2,0))
                axes[2+s, i].axis("off")
                if i == n // 2:
                    axes[2+s, i].set_title(f"Slot {s}", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"slot_vis_{epoch}.png")


    # Define transforms
    img_transform = T.Compose([
        T.ConvertImageDtype(torch.float32)
    ])

    # Create training dataset
    train_dataset = MultiDSprites(
        root="~/datasets/multidsprites",
        split="Train",
        version="colored_on_colored",
        transforms={'image': img_transform},
        download=True,
        convert=True
    )

    # DataLoader with batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )


    # Hyperparameters
    feature_dim = 128
    attn_dim = 64
    n_slots = 5
    n_heads = 4
    t_iters = 3
    height = 64
    width = 64
    epochs = 50
    lr = 1e-4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = ReconModel(
        feature_dim=feature_dim,
        attn_dim=attn_dim,
        n_slots=n_slots,
        n_heads=n_heads,
        t_iters=t_iters,
        height=height,
        width=width
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        losses = []
        for i, batch in enumerate(train_loader):
            images = batch["image"]
            images = images.to(device)
            recon = model(images)
            recon = recon.permute(0, 3, 1, 2)  # ensure [B,C,H,W]
            loss = F.mse_loss(images, recon)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 500 == 0:
                e = i + 1
                print(f"{e}/{len(train_loader)}")
        
        e = epoch + 1
        print(f"Epoch {e}---------\nLoss: {sum(losses)/len(losses)}\n---------")
        # if epoch == 0 or (epoch + 1) % 5 == 0:
        with torch.no_grad():
            images = images[:5]  # [5,3,H,W]
            recon = recon[:5]
            visualize_reconstruction(images, recon, epoch=e, n=5)

            encoding = model.encoder(images)
            slots = model.slot_attention(encoding)
            visualize_slots(images, slots, model.decoder, epoch=e, n=3)
