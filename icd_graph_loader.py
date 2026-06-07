

import torch
import torch.nn.functional as F
import geoopt
import logging
from typing import Optional, Tuple, Dict


class ICDGraphEmbeddingLoader:

    def __init__(
        self,
        graph_data_path: str,
        embeddings_path: str,
        special_tokens: Optional[list] = None,
        special_token_seed: int = 42,
        logger: Optional[logging.Logger] = None,
    ):
        
        self.graph_data_path = graph_data_path
        self.embeddings_path = embeddings_path
        self.special_tokens = special_tokens or ['NORM']
        self.special_token_seed = special_token_seed
        
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.ERROR)

        self.icd_embeddings: Optional[torch.Tensor] = None
        self.code_to_id: Optional[Dict[str, int]] = None
        self.id_to_code: Optional[Dict[int, str]] = None
        self.curvature: Optional[float] = None


    def load(self) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, int]]]:
        
        try:
            self._load_graph_structure()
            hyperbolic_embeddings, curvature = self._load_hyperbolic_embeddings()
            self.icd_embeddings = self._convert_to_euclidean(hyperbolic_embeddings, curvature)
            self._inject_special_tokens()
            return self.icd_embeddings, self.code_to_id
        except Exception as e:
            self.logger.error(f"Failed to load ICD graph embeddings: {e}")
            import traceback
            traceback.print_exc()
            self.logger.error("Training will continue without ICD graph embeddings")
            return None, None


    def _load_graph_structure(self):
        self.logger.info("Loading ICD graph structure...")
        graph_data = torch.load(self.graph_data_path, weights_only=False)
        self.code_to_id = graph_data['code_to_id']
        self.id_to_code = graph_data['id_to_code']
        self.logger.info(f"ICD graph loaded: {len(self.code_to_id)} nodes")


    def _load_hyperbolic_embeddings(self) -> Tuple[torch.Tensor, float]:
        self.logger.info("Loading hyperbolic embeddings...")
        map_location = None if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(self.embeddings_path, map_location=map_location, weights_only=False)

        if isinstance(checkpoint, dict) and 'embeddings' in checkpoint:
            embeddings = checkpoint['embeddings']
            curvature = self._extract_curvature(checkpoint)
        elif isinstance(checkpoint, torch.Tensor):
            embeddings = checkpoint
            curvature = 1.0
            self.logger.warning(f"Old format checkpoint (raw tensor), using default c={curvature}")
        else:
            raise KeyError(
                f"Cannot find 'embeddings' key in checkpoint. "
                f"Available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
            )

        self.logger.info(f"Hyperbolic embeddings loaded: shape={embeddings.shape}")
        self.logger.info(f"  Norm range: [{embeddings.norm(dim=-1).min():.4f}, {embeddings.norm(dim=-1).max():.4f}]")
        return embeddings, curvature

    def _extract_curvature(self, checkpoint: dict) -> float:
        if 'ball.isp_c' in checkpoint:
            isp_c = checkpoint['ball.isp_c']
            actual_c = F.softplus(isp_c).item()
            self.logger.info(f"Curvature: isp_c={isp_c.item():.6f} -> c={actual_c:.6f}")
            return actual_c
        else:
            self.logger.warning("Curvature parameter 'ball.isp_c' not found, using default c=1.0")
            return 1.0


    def _convert_to_euclidean(self, hyperbolic_embeddings: torch.Tensor, curvature: float) -> torch.Tensor:
        ball = geoopt.PoincareBall(c=curvature)
        euclidean_embeddings = ball.logmap0(hyperbolic_embeddings)
        self.curvature = curvature

        self.logger.info(f"Converted to Euclidean space via LogMap0 (c={curvature:.6f})")
        self.logger.info(
            f"  Euclidean norm range: "
            f"[{euclidean_embeddings.norm(dim=-1).min():.4f}, "
            f"{euclidean_embeddings.norm(dim=-1).max():.4f}]"
        )
        return euclidean_embeddings


    def _inject_special_tokens(self):
        for token in self.special_tokens:
            if token in self.code_to_id:
                self.logger.info(f"'{token}' already exists in graph (id={self.code_to_id[token]})")
                continue

            self.logger.info(f"Injecting '{token}' into embeddings...")

            new_id = len(self.code_to_id)
            self.code_to_id[token] = new_id
            self.id_to_code[new_id] = token

            with torch.no_grad():
                mean_vec = self.icd_embeddings.mean(dim=0)
                std_vec = self.icd_embeddings.std(dim=0)

                g = torch.Generator(device=mean_vec.device)
                g.manual_seed(self.special_token_seed)

                token_embedding = torch.normal(
                    mean=mean_vec, std=std_vec, generator=g
                ).unsqueeze(0)

                self.icd_embeddings = torch.cat(
                    [self.icd_embeddings, token_embedding], dim=0
                )

            self.logger.info(
                f"'{token}' injected: id={new_id}, "
                f"new shape={self.icd_embeddings.shape}"
            )
