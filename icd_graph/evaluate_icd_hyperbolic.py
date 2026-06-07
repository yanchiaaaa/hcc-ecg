import argparse
import os

import geoopt
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr


def evaluate_icd_embeddings(checkpoint_path, data_path, output_dir=None, run_tsne=False):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Graph data not found: {data_path}")

    data = torch.load(data_path, map_location="cpu")
    edge_index = data["edge_index"]
    code_to_id = data["code_to_id"]
    id_to_code = {v: k for k, v in code_to_id.items()}

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    c = torch.nn.functional.softplus(checkpoint["ball.isp_c"]).item() if "ball.isp_c" in checkpoint else 1.0
    embeddings = checkpoint["embeddings"].float()
    num_nodes, dim = embeddings.shape
    ball = geoopt.PoincareBall(c=c)

    print("=" * 60)
    print("ICD hyperbolic embedding evaluation")
    print(f"Curvature: {c:.4f} | Dim: {dim} | Nodes: {num_nodes}")

    with torch.no_grad():
        origin = torch.zeros((1, dim))
        norms = ball.dist(embeddings, origin).view(-1).numpy()

    degrees = torch.zeros(num_nodes)
    edges_cpu = edge_index.cpu()
    for i in range(edges_cpu.shape[1]):
        degrees[edges_cpu[0, i]] += 1

    corr, _ = spearmanr(norms, -degrees.numpy())
    print(f"Norm vs negative out-degree Spearman correlation: {corr:.4f}")
    print(f"Mean hyperbolic norm: {norms.mean():.4f}")

    sample_size = min(1000, edge_index.shape[1])
    sample_indices = torch.randperm(edge_index.shape[1])[:sample_size]
    parent_ids = edge_index[0, sample_indices]
    child_ids = edge_index[1, sample_indices]

    with torch.no_grad():
        pos_dists = ball.dist(embeddings[parent_ids], embeddings[child_ids]).mean().item()
        r1 = torch.randint(0, num_nodes, (sample_size,))
        r2 = torch.randint(0, num_nodes, (sample_size,))
        neg_dists = ball.dist(embeddings[r1], embeddings[r2]).mean().item()

    ratio = neg_dists / (pos_dists + 1e-7)
    print(f"Random-pair distance / edge-pair distance: {ratio:.2f}x")

    test_codes = ["I50.9", "E11.9", "N18.9", "I10"]
    print("Nearest-neighbor sanity check:")
    for code in test_codes:
        target_id = None
        for fmt in [code, code.replace(".", ""), code.upper(), code.upper().replace(".", "")]:
            if fmt in code_to_id:
                target_id = code_to_id[fmt]
                break

        if target_id is None:
            print(f"  {code}: not found")
            continue

        with torch.no_grad():
            dists = ball.dist(embeddings[target_id].unsqueeze(0), embeddings).view(-1)
            _, ids = torch.topk(dists, k=min(6, num_nodes), largest=False)
            neighbors = [id_to_code[int(node_id.item())] for node_id in ids.flatten()]
        print(f"  {id_to_code[target_id]}: {' -> '.join(neighbors)}")

    if run_tsne:
        if output_dir is None:
            output_dir = os.path.dirname(checkpoint_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", n_jobs=-1, verbose=1)
        low_dim = tsne.fit_transform(embeddings.numpy())

        plt.figure(figsize=(12, 10))
        sc = plt.scatter(low_dim[:, 0], low_dim[:, 1], c=norms, cmap="viridis", s=1, alpha=0.4)
        plt.colorbar(sc, label="Hyperbolic norm")
        plt.title(f"ICD Poincare embeddings TSNE (c={c:.4f})")
        save_path = os.path.join(output_dir, "icd_eval_tsne.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved TSNE plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained hyperbolic ICD graph embeddings.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_path", default="icd_graph/icd_graph_data.pt")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_tsne", action="store_true")
    args = parser.parse_args()

    evaluate_icd_embeddings(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        run_tsne=args.run_tsne,
    )


if __name__ == "__main__":
    main()
