import argparse
import os

import geoopt
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset


class ICDGraphDataset(Dataset):
    def __init__(self, edge_index, num_nodes, num_neg=20):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_neg = num_neg
        self.num_edges = edge_index.shape[1]

    def __len__(self):
        return self.num_edges

    def __getitem__(self, idx):
        src = self.edge_index[0, idx]
        dst = self.edge_index[1, idx]
        neg = torch.randint(0, self.num_nodes, (self.num_neg,))
        return src, dst, neg


class HyperbolicICDModel(torch.nn.Module):
    def __init__(self, num_nodes, embed_dim=768, c=1.0):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=c, learnable=True)
        init_embeddings = torch.randn(num_nodes, embed_dim) * 1e-3
        self.embeddings = geoopt.ManifoldParameter(
            self.ball.projx(init_embeddings),
            manifold=self.ball,
        )

    def forward(self, src, dst, neg):
        src_embed = self.embeddings[src]
        dst_embed = self.embeddings[dst]
        neg_embed = self.embeddings[neg]

        pos_dist = self.ball.dist(src_embed, dst_embed)
        src_expand = src_embed.unsqueeze(1).expand_as(neg_embed)
        neg_dist = self.ball.dist(
            src_expand.reshape(-1, src_expand.shape[-1]),
            neg_embed.reshape(-1, neg_embed.shape[-1]),
        ).reshape(src.shape[0], -1)

        logits = torch.cat([-pos_dist.unsqueeze(1), -neg_dist], dim=1)
        labels = torch.zeros(src.shape[0], dtype=torch.long, device=src.device)
        return torch.nn.functional.cross_entropy(logits, labels)


def train_hyperbolic(
    data_path,
    save_dir,
    embed_dim=768,
    num_neg=20,
    epochs=500,
    lr=1e-3,
    batch_size=2048,
    num_workers=4,
    device="cuda",
):
    data = torch.load(data_path)
    edge_index = data["edge_index"]
    code_to_id = data["code_to_id"]
    num_nodes = len(code_to_id)
    num_edges = edge_index.shape[1]

    print(f"ICD graph: {num_nodes} nodes, {num_edges} edges")
    print(f"Training hyperbolic embeddings: dim={embed_dim}, neg={num_neg}, lr={lr}, epochs={epochs}")

    dataset = ICDGraphDataset(edge_index, num_nodes, num_neg=num_neg)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = HyperbolicICDModel(num_nodes, embed_dim=embed_dim).to(device)
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr, stabilize=10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for src, dst, neg in dataloader:
            src = src.to(device)
            dst = dst.to(device)
            neg = neg.to(device)

            optimizer.zero_grad()
            loss = model(src, dst, neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0 or epoch == 1:
            c_val = torch.nn.functional.softplus(model.ball.isp_c).item() if hasattr(model.ball, "isp_c") else 1.0
            norms = model.ball.norm(model.embeddings.data).detach()
            print(
                f"Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.6f} | "
                f"LR: {current_lr:.6f} | c: {c_val:.4f} | "
                f"Norm: [{norms.min():.3f}, {norms.mean():.3f}, {norms.max():.3f}]"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "embeddings": model.embeddings.data.cpu(),
                    "ball.isp_c": model.ball.isp_c.data.cpu() if hasattr(model.ball, "isp_c") else torch.tensor(0.5414),
                },
                os.path.join(save_dir, "icd_hyperbolic_best.pth"),
            )

        if epoch % 100 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "embeddings": model.embeddings.data.cpu(),
                    "ball.isp_c": model.ball.isp_c.data.cpu() if hasattr(model.ball, "isp_c") else torch.tensor(0.5414),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss_history": loss_history,
                },
                os.path.join(save_dir, f"icd_hyperbolic_epoch{epoch}.pth"),
            )

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Hyperbolic ICD Embedding Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=150)
    plt.close()

    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Saved outputs to: {save_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Poincare-ball ICD graph embeddings.")
    parser.add_argument("--data_path", default="icd_graph/icd_graph_data.pt")
    parser.add_argument("--save_dir", default="icd_graph/checkpoints")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--num_neg", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_hyperbolic(
        data_path=args.data_path,
        save_dir=args.save_dir,
        embed_dim=args.embed_dim,
        num_neg=args.num_neg,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
