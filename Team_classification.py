from dataclasses import dataclass
from typing import List, Iterable, Any, Generator, Dict

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"


@dataclass
class TeamClusteringState:
    """
    Just holds the models and config – no methods, no 'self'.
    """
    vision_backbone: SiglipVisionModel
    processor: AutoProcessor
    reducer: umap.UMAP
    kmeans: KMeans
    device: str
    batch_size: int

def batched(iterable: Iterable[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """Yield lists of length <= batch_size from an iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def init_team_clustering(device: str = "cpu", batch_size: int = 32) -> TeamClusteringState:
    """
    Load SigLIP and prepare an empty TeamClusteringState.
    UMAP and KMeans will be fitted later.
    """
    backbone = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_NAME).to(device)
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)

    # Unfitted at first – we'll fit them when we see data
    reducer = umap.UMAP(n_components=3)
    kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0)

    state = TeamClusteringState(
        vision_backbone=backbone,
        processor=processor,
        reducer=reducer,
        kmeans=kmeans,
        device=device,
        batch_size=batch_size,
    )
    return state

def compute_siglip_embeddings(crops: List[np.ndarray], state: TeamClusteringState,) -> np.ndarray:
    """
    Turn a list of BGR crops (np.ndarray) into a matrix of SigLIP embeddings.
    """
    if len(crops) == 0:
        return np.empty((0, 0), dtype=np.float32)

    # convert OpenCV images -> PIL for the processor
    pil_list = [sv.cv2_to_pillow(img) for img in crops]

    all_chunks = []
    state.vision_backbone.eval()

    with torch.no_grad():
        for batch_images in tqdm(batched(pil_list, state.batch_size), desc="SigLIP embeddings"):
            inputs = state.processor(images=batch_images, return_tensors="pt").to(state.device)
            outputs = state.vision_backbone(**inputs)
            # average over patches → one vector per image
            emb = outputs.last_hidden_state.mean(dim=1)  # (B, D)
            all_chunks.append(emb.cpu().numpy())

    return np.concatenate(all_chunks, axis=0)

def train_team_clusters(
    crops: List[np.ndarray],
    state: TeamClusteringState,
) -> TeamClusteringState:
    """
    Fit UMAP and KMeans on a set of player crops.
    Returns the *same* state object, but with fitted reducer + kmeans.
    """
    feats = compute_siglip_embeddings(crops, state)
    # Learn low-dimensional manifold
    projected = state.reducer.fit_transform(feats)
    # Learn 2-team clustering
    state.kmeans.fit(projected)
    return state

def assign_teams_to_crops(
crops: List[np.ndarray],
state: TeamClusteringState,
) -> np.ndarray:
    """
    For each crop, return a label 0 or 1 indicating the team cluster.
    """
    if len(crops) == 0:
        return np.array([], dtype=int)

    feats = compute_siglip_embeddings(crops, state)
    projected = state.reducer.transform(feats)
    labels = state.kmeans.predict(projected)
    return labels.astype(int)

