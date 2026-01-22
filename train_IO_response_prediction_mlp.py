"""
ASPIRE Patient Outcome Prediction Script

Trains a multimodal MLP for patient-level outcome prediction by:
1. Loading pre-computed MUSK embeddings
2. Aggregating patch-level gene signature expression predictions using 4 feature families
3. Combining features for binary response classification

"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "batch_size": 16,
    "num_epochs": 30,          
    "hidden_dim_1": 32,          
    "hidden_dim_2": 16,         
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,        
    "patience": 30,              
}

# Attention-ranked bins for feature computation
ATTENTION_BINS = [(1, 25), (26, 50), (51, 75), (76, 100)]

# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_musk_embeddings(embedding_dir, patient_id):
    """
    Load pre-computed MUSK embeddings for a patient.
    
    Expected format:
        embedding_dir/{patient_id}/{patient_id}_{x}_{y}_embedding.pt
        Each .pt file contains {'patch_id': str, 'image_embedding': np.array(1024,)}
    
    Returns:
        embeddings_dict: dict mapping (x, y) coordinates to 1024-dim embeddings
    """
    patient_dir = os.path.join(embedding_dir, patient_id)
    if not os.path.isdir(patient_dir):
        return None
    
    pt_files = [f for f in os.listdir(patient_dir) if f.endswith('_embedding.pt')]
    if not pt_files:
        return None
    
    embeddings_dict = {}
    for pt_file in pt_files:
        # Parse x, y from filename: {patient_id}_{x}_{y}_embedding.pt
        parts = pt_file.replace('_embedding.pt', '').split('_')
        try:
            x = int(parts[-2])
            y = int(parts[-1])
            data = torch.load(os.path.join(patient_dir, pt_file), map_location='cpu', weights_only=False)
            if isinstance(data, dict) and 'image_embedding' in data:
                emb = data['image_embedding']
            else:
                emb = data
            emb = np.asarray(emb)
            if emb.ndim == 2 and emb.shape[0] == 1:
                emb = emb[0]
            embeddings_dict[(x, y)] = emb
        except (ValueError, IndexError):
            continue
    
    return embeddings_dict if embeddings_dict else None


def load_gene_predictions(gene_pred_dir, patient_id):
    """
    Load patch-level gene signature expression predictions.
    """
    pred_path = os.path.join(gene_pred_dir, f"{patient_id}_predictions.csv")
    if not os.path.exists(pred_path):
        return None
    
    return pd.read_csv(pred_path)


# ============================================================================
# Coordinate Normalization (Re-center to Highest Attention Patch)
# ============================================================================

def recenter_coordinates_to_centroid(gene_predictions):
    """
    Re-center all patch coordinates relative to the "Centroid" patch.
    
    The Centroid patch is defined as the patch with the HIGHEST ATTENTION SCORE.
    This transformation ensures spatial features capture relative topology of the 
    immune response (e.g., dispersion from the tumor core) rather than absolute 
    slide positions.
    
    Returns:
        x_centered: np.ndarray of x coordinates centered to centroid
        y_centered: np.ndarray of y coordinates centered to centroid
        centroid_idx: index of the centroid (highest attention) patch
    """
    has_xy = "x" in gene_predictions.columns and "y" in gene_predictions.columns
    has_attn = "attention_weight" in gene_predictions.columns
    
    if not has_xy:
        return None, None, None
    
    x = gene_predictions["x"].values.astype(np.float64)
    y = gene_predictions["y"].values.astype(np.float64)
    
    if has_attn:
        attn = gene_predictions["attention_weight"].values
        # Centroid = patch with HIGHEST attention score
        centroid_idx = np.argmax(attn)
    else:
        # Fallback: use geometric center
        centroid_idx = 0  # or could use patch closest to mean position
    
    # Get centroid coordinates
    centroid_x = x[centroid_idx]
    centroid_y = y[centroid_idx]
    
    # Re-center all coordinates relative to centroid
    x_centered = x - centroid_x
    y_centered = y - centroid_y
    
    return x_centered, y_centered, centroid_idx


def _compute_pairwise_distances(x, y):
    """Compute pairwise Euclidean distances between patches using scipy pdist."""
    if x is None or y is None or len(x) < 2:
        return None
    coords = np.column_stack([x, y])
    try:
        distances = squareform(pdist(coords))
        return distances
    except:
        return None


# ============================================================================
# Flattened Features (Top-N Raw Values)
# ============================================================================

def flattened_top_n_features(gene_predictions, x_centered, y_centered, top_n=50):
    """
    Flattened Features: Raw patch-level values for top-N attention patches.
    
    Creates a fixed-length feature vector by flattening the (adjusted_x, adjusted_y, 
    immune_score, attention_weight) values for the top-N attention-ranked patches.
    Pads with zeros if fewer than top_n patches available.
    
    Args:
        gene_predictions: DataFrame with patch data
        x_centered: centered x coordinates
        y_centered: centered y coordinates  
        top_n: number of top attention patches to use (default: 50)
    
    Returns:
        np.ndarray of shape (top_n * 4,) containing flattened features
    """
    has_xy = x_centered is not None and y_centered is not None
    has_attn = "attention_weight" in gene_predictions.columns
    has_immune = "immune_score" in gene_predictions.columns
    
    n_patches = len(gene_predictions)
    
    if not has_xy or not has_attn or not has_immune:
        return np.zeros(top_n * 4, dtype=np.float32)
    
    # Get attention weights and sort indices
    attn = gene_predictions["attention_weight"].values
    immune_scores = gene_predictions["immune_score"].values
    sorted_indices = np.argsort(attn)[::-1]  # Highest attention first
    
    # Select top_n patches
    patches_to_use = min(top_n, n_patches)
    top_indices = sorted_indices[:patches_to_use]
    
    # Build feature matrix: [adjusted_x, adjusted_y, immune_score, attention_weight]
    feature_matrix = np.column_stack([
        x_centered[top_indices],
        y_centered[top_indices],
        immune_scores[top_indices],
        attn[top_indices]
    ])
    
    # Pad with zeros if needed, this is for compatibility of larger top_n options, and with top_n = 50 this is never really used
    if patches_to_use < top_n:
        padding = np.zeros((top_n - patches_to_use, 4))
        feature_matrix = np.vstack([feature_matrix, padding])
    
    return feature_matrix.flatten().astype(np.float32)


def _get_attention_bin_indices(attn, bin_range, total_patches):
    """
    Get patch indices falling within a specific attention rank bin.
    
    Args:
        attn: attention weights array
        bin_range: tuple (start_percentile, end_percentile), e.g. (1, 25) for top 25%
        total_patches: total number of patches
    
    Returns:
        indices of patches in this attention bin
    """
    sorted_indices = np.argsort(attn)[::-1]  # descending order (highest attention first)
    
    # Convert percentile to indices
    start_pct, end_pct = bin_range
    start_idx = int((start_pct - 1) * total_patches / 100)
    end_idx = int(end_pct * total_patches / 100)
    
    # Clamp indices
    start_idx = max(0, start_idx)
    end_idx = min(total_patches, end_idx)
    
    return sorted_indices[start_idx:end_idx]


# ============================================================================
# Feature Family 1: Spatial Tissue Architecture
# ============================================================================

def spatial_tissue_architecture_features(gene_predictions, x_centered, y_centered):
    """
    Feature Family 1: Spatial Tissue Architecture
    
    Quantifies the geometric spread of the selected regions by computing:
    - Global coordinate statistics: mean, variance, min, max, median of adjusted positions
    - Pairwise Euclidean distances (using scipy pdist)
    - Nearest-neighbor clustering coefficients
    - Multi-radius neighbor counting for local spatial density
    - Edge effect/boundary analysis features
    
    Distinguishes between diffuse distribution expression patterns and 
    spatially aggregated expression clusters.
    
    Returns:
        np.ndarray of spatial architecture features
    """
    features = []
    
    has_xy = x_centered is not None and y_centered is not None
    has_attn = "attention_weight" in gene_predictions.columns
    
    immune_scores = gene_predictions["immune_score"].values if "immune_score" in gene_predictions.columns else np.zeros(len(gene_predictions))
    
    if has_attn:
        attn = gene_predictions["attention_weight"].values
    else:
        attn = np.ones(len(gene_predictions)) / len(gene_predictions)
    
    n_patches = len(gene_predictions)
    
    if has_xy and n_patches > 0:
        # === Global coordinate statistics (on centered coordinates) ===
        features.extend([
            np.mean(x_centered), np.mean(y_centered),      # Mean
            np.var(x_centered), np.var(y_centered),        # Variance
            np.min(x_centered), np.min(y_centered),        # Min
            np.max(x_centered), np.max(y_centered),        # Max
            np.median(x_centered), np.median(y_centered),  # Median
            np.std(x_centered), np.std(y_centered),        # Std
            np.ptp(x_centered), np.ptp(y_centered),        # Range (peak-to-peak)
        ])
        
        # === Pairwise Euclidean distances (using scipy pdist) ===
        if n_patches > 1:
            coords = np.column_stack([x_centered, y_centered])
            try:
                spatial_distances = pdist(coords)
                dist_matrix = squareform(spatial_distances)
                
                features.extend([
                    np.mean(spatial_distances),      # Mean pairwise distance
                    np.std(spatial_distances),       # Std pairwise distance
                    np.min(spatial_distances),       # Min pairwise distance
                    np.max(spatial_distances),       # Max pairwise distance
                    np.median(spatial_distances),    # Median pairwise distance
                ])
                
                # === Nearest-neighbor analysis ===
                if n_patches > 2:
                    np.fill_diagonal(dist_matrix, np.inf)
                    nn_distances = np.min(dist_matrix, axis=1)
                    
                    # Note: min(nn_distances) == min(spatial_distances), so omit
                    features.extend([
                        np.mean(nn_distances),        # Mean NN distance
                        np.std(nn_distances),         # Std NN distance
                        np.max(nn_distances),         # Max NN distance
                        np.median(nn_distances),      # Median NN distance
                        np.std(nn_distances) / (np.mean(nn_distances) + 1e-8),  # Regularity of spacing
                        np.sum(nn_distances < np.median(spatial_distances)) / len(nn_distances),  # Clustering proxy
                    ])
                    
                    # Immune similarity of nearest neighbors
                    nn_immune_diff = np.mean([
                        abs(immune_scores[i] - immune_scores[np.argmin(dist_matrix[i])]) 
                        for i in range(n_patches)
                    ])
                    features.append(nn_immune_diff)
                else:
                    features.extend([0] * 7)  # Reduced by 1 (removed min NN)
                
                # === Multi-radius neighbor counting (using percentile-based radii) ===
                radii_percentiles = [25, 50, 75]
                radii = [np.percentile(spatial_distances, p) for p in radii_percentiles]
                
                for radius in radii:
                    neighbor_counts = np.sum(dist_matrix < radius, axis=1)
                    features.extend([
                        np.mean(neighbor_counts),
                        np.std(neighbor_counts),
                        np.max(neighbor_counts),
                    ])
                
                # Immune score in high-density regions
                if n_patches > 3:
                    median_density = np.median(neighbor_counts)
                    high_density_mask = neighbor_counts > median_density
                    if np.sum(high_density_mask) > 0:
                        features.extend([
                            np.mean(immune_scores[high_density_mask]),
                            np.std(immune_scores[high_density_mask]) if np.sum(high_density_mask) > 1 else 0,
                            np.sum(high_density_mask) / n_patches,
                        ])
                    else:
                        features.extend([0, 0, 0])
                    
                    # Attention in high-density regions
                    norm_attention = attn / (attn.sum() + 1e-8)
                    if np.sum(high_density_mask) > 0:
                        features.extend([
                            np.mean(attn[high_density_mask]),
                            np.std(attn[high_density_mask]) if np.sum(high_density_mask) > 1 else 0,
                            np.sum(norm_attention[high_density_mask]),
                        ])
                    else:
                        features.extend([0, 0, 0])
                else:
                    features.extend([0] * 6)
                    
            except Exception:
                features.extend([0] * 28)  # All distance-based features
        else:
            features.extend([0] * 28)
        
        # === Edge effect / boundary analysis ===
        immune_median = np.median(immune_scores) if len(immune_scores) > 0 else 0
        edge_features = []
        for coord_arr, coord_name in [(x_centered, 'x'), (y_centered, 'y')]:
            min_mask = coord_arr == np.min(coord_arr)
            max_mask = coord_arr == np.max(coord_arr)
            edge_features.append(np.mean(immune_scores[min_mask]) if np.sum(min_mask) > 0 else immune_median)
            edge_features.append(np.mean(immune_scores[max_mask]) if np.sum(max_mask) > 0 else immune_median)
        features.extend(edge_features)
        
    else:
        features.extend([0] * 46)  # No spatial coordinates available
    
    # === Compute features within attention-ranked bins ===
    if has_attn and has_xy and n_patches >= 4:
        for bin_start, bin_end in ATTENTION_BINS:
            bin_indices = _get_attention_bin_indices(attn, (bin_start, bin_end), n_patches)
            
            if len(bin_indices) > 0:
                bin_x = x_centered[bin_indices]
                bin_y = y_centered[bin_indices]
                bin_immune = immune_scores[bin_indices]
                bin_attn = attn[bin_indices]
                
                # Spatial spread and statistics within bin
                # Note: bin immune means duplicate quartile means in attention_weighted, so omit
                features.extend([
                    np.mean(bin_x), np.mean(bin_y),
                    np.std(bin_x) if len(bin_x) > 1 else 0,
                    np.std(bin_y) if len(bin_y) > 1 else 0,
                    np.ptp(bin_x), np.ptp(bin_y),
                    np.std(bin_immune) if len(bin_immune) > 1 else 0,  # Keep std, not mean
                    np.mean(bin_attn),
                    np.std(bin_attn) if len(bin_attn) > 1 else 0,
                ])
                
                # Pairwise distances within bin
                if len(bin_indices) > 1:
                    bin_coords = np.column_stack([bin_x, bin_y])
                    try:
                        bin_distances = pdist(bin_coords)
                        features.extend([np.mean(bin_distances), np.std(bin_distances)])
                    except:
                        features.extend([0, 0])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0] * 11)  # Reduced from 12
    else:
        features.extend([0] * 44)  # 4 bins × 11 features
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# Feature Family 2: Attention-Weighted Summaries  
# ============================================================================

def attention_weighted_summaries_features(gene_predictions, x_centered, y_centered):
    """
    Feature Family 2: Attention-Weighted Summaries
    
    Computes:
    - Averages of immune activity weighted by attention scores
    - Shannon entropy of attention distribution (spatial concentration of tumor-relevant morphology)
    - Dispersion of high-attention hotspots (with pdist-based distance calculations)
    - Signal progression of immune signature across attention-ranked patches
    - Attention clustering and hotspot metrics
    
    Characterizes concordance between histological saliency and molecular phenotype.
    
    Returns:
        np.ndarray of attention-weighted summary features
    """
    has_xy = x_centered is not None and y_centered is not None
    has_attn = "attention_weight" in gene_predictions.columns
    
    immune_scores = gene_predictions["immune_score"].values if "immune_score" in gene_predictions.columns else np.zeros(len(gene_predictions))
    
    if has_attn:
        attn = gene_predictions["attention_weight"].values
        attn_norm = attn / (attn.sum() + 1e-8)  # Normalize to probability distribution
    else:
        attn = np.ones(len(gene_predictions)) / len(gene_predictions)
        attn_norm = attn
    
    features = []
    n_patches = len(gene_predictions)
    
    # === Attention-weighted averages of immune activity ===
    weighted_immune_mean = np.average(immune_scores, weights=attn_norm)
    unweighted_mean = np.mean(immune_scores)
    features.append(weighted_immune_mean)
    
    # Weighted vs unweighted comparison (unweighted mean is in distribution_summaries)
    features.append(weighted_immune_mean - unweighted_mean)  # Difference only (not raw unweighted)
    
    # Weighted spatial statistics (unweighted x/y std are in spatial_architecture)
    if has_xy:
        features.extend([
            np.average(x_centered, weights=attn_norm),  # Weighted x mean
            np.average(y_centered, weights=attn_norm),  # Weighted y mean
        ])
    else:
        features.extend([0, 0])
    
    # === Shannon entropy of attention distribution ===
    # Quantifies spatial concentration of tumor-relevant morphology
    attn_entropy = -np.sum(attn_norm * np.log2(attn_norm + 1e-10))
    # Note: Using normalized entropy only (raw and normalized are perfectly correlated for fixed n_patches)
    max_entropy = np.log2(n_patches) if n_patches > 1 else 1
    features.append(attn_entropy / (max_entropy + 1e-8))  # Normalized entropy (0 to 1)
    
    # === Attention clustering & hotspots (from original code) ===
    if has_xy and n_patches > 1:
        # Multiple attention thresholds for hotspot analysis
        for percentile_thresh in [70, 80, 90]:
            thresh = np.percentile(attn, percentile_thresh)
            mask = attn >= thresh
            
            if np.sum(mask) > 1:
                masked_patches = np.column_stack([x_centered[mask], y_centered[mask]])
                try:
                    distances = pdist(masked_patches)
                    features.extend([
                        np.mean(distances) if len(distances) > 0 else 0,  # Mean hotspot spread
                        np.mean(immune_scores[mask]),                      # Mean immune in hotspot
                        np.sum(mask) / n_patches,                          # Fraction in hotspot
                    ])
                except:
                    features.extend([0, np.mean(immune_scores[mask]), np.sum(mask) / n_patches])
            else:
                features.extend([0, 0, 0])
        
        # High attention (80th percentile) cluster metrics
        high_attn_thresh = np.percentile(attn, 80)
        high_attn_mask = attn >= high_attn_thresh
        
        if np.sum(high_attn_mask) > 0:
            high_attn_x = x_centered[high_attn_mask]
            high_attn_y = y_centered[high_attn_mask]
            high_attn_immune = immune_scores[high_attn_mask]
            high_attn_weights = attn[high_attn_mask]
            
            features.extend([
                np.std(high_attn_immune) if len(high_attn_immune) > 1 else 0,  # Immune variability
                np.std(high_attn_x) if len(high_attn_x) > 1 else 0,            # X spread
                np.std(high_attn_y) if len(high_attn_y) > 1 else 0,            # Y spread
                np.ptp(high_attn_x),                                            # X range
                np.ptp(high_attn_y),                                            # Y range
                np.mean(high_attn_weights),                                     # Mean attention in hotspot
                np.std(high_attn_weights) if len(high_attn_weights) > 1 else 0, # Attention variability
            ])
        else:
            features.extend([0] * 7)
        
        # Hotspot centroid distance from origin
        n_hotspots = max(1, int(0.1 * n_patches))
        hotspot_indices = np.argsort(attn)[-n_hotspots:]
        hotspot_x = x_centered[hotspot_indices]
        hotspot_y = y_centered[hotspot_indices]
        
        hotspot_centroid_x = np.mean(hotspot_x)
        hotspot_centroid_y = np.mean(hotspot_y)
        hotspot_to_centroid_dist = np.sqrt(hotspot_centroid_x**2 + hotspot_centroid_y**2)
        
        weighted_x = np.average(x_centered, weights=attn_norm)
        weighted_y = np.average(y_centered, weights=attn_norm)
        weighted_centroid_dist = np.sqrt(weighted_x**2 + weighted_y**2)
        
        features.extend([hotspot_to_centroid_dist, weighted_centroid_dist])
    else:
        features.extend([0] * 18)  # All hotspot features
    
    # === Signal progression across attention-ranked patches ===
    if has_attn and n_patches >= 4:
        sorted_indices = np.argsort(attn)[::-1]  # Highest attention first
        sorted_immune = immune_scores[sorted_indices]
        
        # Split into thirds for progression analysis
        third = n_patches // 3
        if third > 0:
            early_immune = np.mean(sorted_immune[:third])
            mid_immune = np.mean(sorted_immune[third:2*third])
            late_immune = np.mean(sorted_immune[-third:])
            features.extend([early_immune, mid_immune, late_immune])
        else:
            features.extend([0, 0, 0])
        
        # Immune score progression by quartiles
        quartile_size = n_patches // 4
        if quartile_size > 0:
            q1_immune = np.mean(sorted_immune[:quartile_size])
            q2_immune = np.mean(sorted_immune[quartile_size:2*quartile_size])
            q3_immune = np.mean(sorted_immune[2*quartile_size:3*quartile_size])
            q4_immune = np.mean(sorted_immune[3*quartile_size:])
            
            features.extend([q1_immune, q2_immune, q3_immune, q4_immune])
            features.append(q1_immune - q4_immune)  # Top vs bottom difference
            
            # Variability in early vs late
            features.append(np.std(sorted_immune[:n_patches//2]))
            features.append(np.std(sorted_immune[n_patches//2:]))
        else:
            features.extend([0] * 7)
        
        # Correlation between attention and immune score
        attn_immune_corr = np.corrcoef(attn, immune_scores)[0, 1]
        features.append(attn_immune_corr if not np.isnan(attn_immune_corr) else 0)
        
        # Trend analysis: correlation with rank
        if n_patches > 1:
            rank_corr = np.corrcoef(np.arange(len(sorted_immune)), sorted_immune)[0, 1]
            features.append(rank_corr if not np.isnan(rank_corr) else 0)
        else:
            features.append(0)
    else:
        features.extend([0] * 14)
    
    # === Attention statistics ===
    # Note: np.max(attn) is same as flat_patch0_attn (top attention = max), so omit
    # Note: peak-to-mean ratio correlates perfectly with max, so omit
    features.extend([
        np.mean(attn),
        np.std(attn),
        np.median(attn),
        np.percentile(attn, 90),                 # 90th percentile
        np.sum(attn_norm ** 2),                  # Herfindahl index (concentration)
    ])
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# Feature Family 3: Distribution Summaries
# ============================================================================

def distribution_summaries_features(gene_predictions, x_centered, y_centered):
    """
    Feature Family 3: Distribution Summaries
    
    Characterizes the intensity profile of the immune signal using:
    - Percentile-based statistics: medians, quartiles, deciles, interquartile ranges
    - Higher-order statistical moments: skewness, kurtosis (using scipy.stats)
    - Intensity concentration metrics: max-to-median ratios, IQR ratios
    - Biological ratios and threshold-based features
    
    Quantifies inequality and dominance of immune response independent of absolute spatial location.
    
    Returns:
        np.ndarray of distribution summary features
    """
    has_xy = x_centered is not None and y_centered is not None
    has_attn = "attention_weight" in gene_predictions.columns
    
    immune_scores = gene_predictions["immune_score"].values if "immune_score" in gene_predictions.columns else np.zeros(len(gene_predictions))
    
    if has_attn:
        attn = gene_predictions["attention_weight"].values
        attn_norm = attn / (attn.sum() + 1e-8)
    else:
        attn = np.ones(len(gene_predictions)) / len(gene_predictions)
        attn_norm = attn
    
    n_patches = len(gene_predictions)
    features = []
    
    # === Basic statistics ===
    features.extend([
        np.mean(immune_scores),
        np.std(immune_scores),
        np.min(immune_scores),
        np.max(immune_scores),
    ])
    
    # === Percentile-based statistics ===
    q10 = np.percentile(immune_scores, 10)
    q25 = np.percentile(immune_scores, 25)
    q50 = np.percentile(immune_scores, 50)  # Median
    q75 = np.percentile(immune_scores, 75)
    q90 = np.percentile(immune_scores, 90)
    q95 = np.percentile(immune_scores, 95)
    q05 = np.percentile(immune_scores, 5)
    
    features.extend([q05, q10, q25, q50, q75, q90, q95])
    
    # Deciles (excluding 10, 50, 90 which are already in percentiles above)
    for p in [20, 30, 40, 60, 70, 80]:
        features.append(np.percentile(immune_scores, p))
    
    # === Interquartile ranges ===
    iqr = q75 - q25
    idr = q90 - q10  # Interdecile range
    full_range = np.max(immune_scores) - np.min(immune_scores)
    
    features.extend([iqr, idr, full_range])
    
    # Range ratios
    features.append((q90 - q10) / (iqr + 1e-8))  # IDR/IQR ratio
    features.append((q75 - q25) / (full_range + 1e-8))  # IQR/Range ratio
    features.append((q90 - q10) / (full_range + 1e-8))  # IDR/Range ratio
    
    # === Higher-order statistical moments (using scipy.stats) ===
    if len(immune_scores) > 2:
        skewness = stats.skew(immune_scores)
        kurtosis = stats.kurtosis(immune_scores)
    else:
        skewness = 0
        kurtosis = 0
    
    features.extend([skewness, kurtosis])
    
    # Attention statistics (higher-order)
    if has_attn and len(attn) > 2:
        features.extend([
            stats.skew(attn),
            stats.kurtosis(attn),
        ])
    else:
        features.extend([0, 0])
    
    # === Intensity concentration metrics ===
    mean_immune = np.mean(immune_scores)
    
    # Max-to-median and max-to-mean ratios
    features.extend([
        np.max(immune_scores) / (q50 + 1e-8),      # Max/Median
        np.max(immune_scores) / (mean_immune + 1e-8),  # Max/Mean
        q50 / (np.min(immune_scores) + 1e-8),     # Median/Min
    ])
    
    # IQR ratio (relative spread)
    features.append(iqr / (q50 + 1e-8))
    
    # Coefficient of variation for immune (attn CV correlates with attn_std, so omit)
    cv_immune = np.std(immune_scores) / (mean_immune + 1e-8)
    features.append(cv_immune)
    
    # === Gini coefficient (inequality measure) ===
    sorted_immune = np.sort(immune_scores)
    n = len(sorted_immune)
    if n > 1 and np.sum(sorted_immune) > 0:
        gini = np.sum(np.abs(sorted_immune[:, None] - sorted_immune)) / (2 * n**2 * np.mean(sorted_immune) + 1e-8)
    else:
        gini = 0
    features.append(gini)
    
    # === Biological ratios & threshold-based features (from original code) ===
    # Fraction in various ranges
    features.extend([
        np.sum(immune_scores > q50) / n_patches,   # Above median
        np.sum(immune_scores > q75) / n_patches,   # Above Q75
        np.sum(immune_scores < q25) / n_patches,   # Below Q25
        np.sum(immune_scores > q90) / n_patches,   # Above Q90 (high immune)
        np.sum(immune_scores < q10) / n_patches,   # Below Q10 (low immune)
    ])
    
    # Attention-weighted by immune zones
    features.extend([
        np.sum(attn_norm[immune_scores > q50]),   # Attention on high-immune
        np.sum(attn_norm[immune_scores < q50]),   # Attention on low-immune
        np.sum(attn_norm[immune_scores > q75]),   # Attention on very high immune
        np.sum(attn_norm[immune_scores < q25]),   # Attention on very low immune
    ])
    
    # High immune + high attention regions
    if has_attn:
        median_attn = np.median(attn)
        q75_attn = np.percentile(attn, 75)
        features.extend([
            np.sum((immune_scores > q50) & (attn > median_attn)) / n_patches,
            np.sum((immune_scores > q75) & (attn > q75_attn)) / n_patches,
        ])
    else:
        features.extend([0, 0])
    
    # Skewness indicators (simple)
    features.extend([
        (mean_immune - q50) / (np.std(immune_scores) + 1e-8),  # Mean-median skew
        (q75 - q50) / (q50 - q25 + 1e-8),                       # Quartile skew ratio
    ])
    
    # Attention-weighted immune in zones
    features.append(np.mean(immune_scores[attn > np.median(attn)]) / (q50 + 1e-8))
    
    # === Entropy-based features ===
    # Histogram entropy
    n_bins = 10
    hist, _ = np.histogram(immune_scores, bins=n_bins, density=True)
    hist_norm = hist / (hist.sum() + 1e-8)
    distribution_entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-10))
    features.append(distribution_entropy)
    
    # Immune score "entropy" (treating as probability-like)
    immune_positive = immune_scores - np.min(immune_scores) + 1e-8
    immune_prob = immune_positive / np.sum(immune_positive)
    immune_entropy = -np.sum(immune_prob * np.log(immune_prob + 1e-10))
    features.append(immune_entropy)
    
    # === Spatial distribution features (if available) ===
    if has_xy:
        # Spatial percentiles (excluding 50 which is median, already in spatial_architecture)
        for p in [25, 75]:
            features.append(np.percentile(x_centered, p))
            features.append(np.percentile(y_centered, p))
        
        # Distance from origin distribution
        distances_from_centroid = np.sqrt(x_centered**2 + y_centered**2)
        features.extend([
            np.mean(distances_from_centroid),
            np.std(distances_from_centroid),
            np.median(distances_from_centroid),
            np.percentile(distances_from_centroid, 90),
        ])
        
        # Center vs periphery immune comparison
        median_dist = np.median(distances_from_centroid)
        center_immune = np.mean(immune_scores[distances_from_centroid < median_dist])
        periphery_immune = np.mean(immune_scores[distances_from_centroid >= median_dist])
        
        features.extend([
            center_immune,
            periphery_immune,
            center_immune - periphery_immune,  # Center-periphery difference
        ])
        
        # Correlation between distance and immune score
        dist_immune_corr = np.corrcoef(distances_from_centroid, immune_scores)[0, 1]
        features.append(dist_immune_corr if not np.isnan(dist_immune_corr) else 0)
    else:
        features.extend([0] * 12)  # Placeholder for spatial features (reduced from 14)
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# Feature Family 4: Structural Dispersion
# ============================================================================

def structural_dispersion_features(gene_predictions, x_centered, y_centered):
    """
    Feature Family 4: Structural Dispersion
    
    Captures multi-scale heterogeneity by partitioning the coordinate space into:
    - Radial zones (concentric rings) using percentile-based boundaries
    - Quadrants (Cartesian sectors) using median split (as in original code)
    - Octants (angular sectors) using arctan2
    - Spatial gradients using np.gradient() as in original code
    - Progression & trajectory analysis (sorted by attention)
    
    Within each partition, calculates:
    - Density of immune-active patches
    - Intensity of immune signal
    - Directional biases in the infiltrate
    - Spatial gradient magnitudes (directional derivatives of immune score)
    
    Returns:
        np.ndarray of structural dispersion features
    """
    has_xy = x_centered is not None and y_centered is not None
    has_attn = "attention_weight" in gene_predictions.columns
    
    immune_scores = gene_predictions["immune_score"].values if "immune_score" in gene_predictions.columns else np.zeros(len(gene_predictions))
    
    if has_attn:
        attn = gene_predictions["attention_weight"].values
        attn_norm = attn / (attn.sum() + 1e-8)
    else:
        attn = np.ones(len(gene_predictions)) / len(gene_predictions)
        attn_norm = attn
    
    features = []
    n_patches = len(gene_predictions)
    
    if has_xy and n_patches > 0:
        # Compute distances from centroid (origin after re-centering)
        distances_from_centroid = np.sqrt(x_centered**2 + y_centered**2)
        
        # === Quadrant analysis using MEDIAN SPLIT (as in original code) ===
        x_mid = np.median(x_centered)
        y_mid = np.median(y_centered)
        
        quadrant_masks = [
            (x_centered <= x_mid) & (y_centered <= y_mid),  # Q1
            (x_centered > x_mid) & (y_centered <= y_mid),   # Q2
            (x_centered <= x_mid) & (y_centered > y_mid),   # Q3
            (x_centered > x_mid) & (y_centered > y_mid),    # Q4
        ]
        
        quad_means = []
        quad_stds = []
        quad_attns = []
        quad_counts = []
        
        for mask in quadrant_masks:
            q_count = np.sum(mask)
            quad_counts.append(q_count)
            if q_count > 0:
                quad_means.append(np.mean(immune_scores[mask]))
                quad_stds.append(np.std(immune_scores[mask]) if q_count > 1 else 0)
                quad_attns.append(np.mean(attn[mask]))
            else:
                quad_means.append(0)
                quad_stds.append(0)
                quad_attns.append(0)
        
        # Output only Q0 and Q1 fractions (Q2, Q3 are redundant due to median split)
        features.extend([
            quad_counts[0] / n_patches, quad_means[0], quad_attns[0],
            quad_counts[1] / n_patches, quad_means[1], quad_attns[1],
            quad_means[2], quad_attns[2],  # Only mean/attn for Q2, Q3 (frac is redundant)
            quad_means[3], quad_attns[3],
        ])
        
        # Quadrant statistics (from original code)
        features.extend([
            np.std(quad_means),                                # Quadrant immune variability
            np.max(quad_means) - np.min(quad_means),          # Quadrant immune range
            np.std(quad_attns),                                # Quadrant attention variability
            np.argmax(quad_means),                             # Dominant immune quadrant
            np.argmax(quad_attns),                             # Dominant attention quadrant
        ])
        
        # Quadrant correlation
        if np.std(quad_means) > 0 and np.std(quad_attns) > 0:
            quad_corr = np.corrcoef(quad_means, quad_attns)[0, 1]
            features.append(quad_corr if not np.isnan(quad_corr) else 0)
        else:
            features.append(0)
        
        # Spatial asymmetry (from original code)
        # Note: count_asymmetry ≈ count_range for 4 quadrants, keep only one
        features.extend([
            abs(quad_means[0] - quad_means[3]),                # Diagonal 1
            abs(quad_means[1] - quad_means[2]),                # Diagonal 2
            abs(np.mean(quad_means[:2]) - np.mean(quad_means[2:])),  # Top vs bottom
            abs(quad_means[0] + quad_means[2] - quad_means[1] - quad_means[3]),  # Cross
            np.std(quad_counts),                                # Count asymmetry (keep only this)
            np.mean(quad_stds),                                 # Mean intra-quadrant variability
            np.std(quad_stds),                                  # Variability of variabilities
        ])
        
        # === Octant analysis using arctan2 (as in original code) ===
        angles_deg = np.degrees(np.arctan2(y_centered - y_mid, x_centered - x_mid))
        angles_deg = (angles_deg + 360) % 360  # Normalize to [0, 360)
        
        octant_means = []
        for angle_start in range(0, 360, 45):
            angle_end = angle_start + 45
            mask = (angles_deg >= angle_start) & (angles_deg < angle_end)
            o_count = np.sum(mask)
            if o_count > 0:
                octant_means.append(np.mean(immune_scores[mask]))
                features.extend([o_count / n_patches, np.mean(immune_scores[mask])])
            else:
                octant_means.append(0)
                features.extend([0, 0])
        
        # Octant statistics
        features.extend([
            np.std(octant_means),
            np.max(octant_means) - np.min(octant_means),
            np.argmax(octant_means),  # Dominant octant
        ])
        
        # === Radial zones using percentile-based boundaries (as in original code) ===
        center_x = np.mean(x_centered)
        center_y = np.mean(y_centered)
        radial_dist = np.sqrt((x_centered - center_x)**2 + (y_centered - center_y)**2)
        
        r33 = np.percentile(radial_dist, 33) if n_patches > 2 else 0
        r66 = np.percentile(radial_dist, 66) if n_patches > 2 else np.max(radial_dist)
        
        zone_bounds = [(0, r33), (r33, r66), (r66, np.inf)]
        zone_means = []
        zone_attns = []
        
        for low, high in zone_bounds:
            mask = (radial_dist >= low) & (radial_dist < high)
            if np.sum(mask) > 0:
                zone_means.append(np.mean(immune_scores[mask]))
                zone_attns.append(np.mean(attn[mask]))
            else:
                zone_means.append(0)
                zone_attns.append(0)
        
        features.extend([
            zone_means[0], zone_means[1], zone_means[2],      # Inner, middle, outer means
            zone_means[2] - zone_means[0],                     # Radial gradient
            np.std(zone_means),                                # Zone variability
            zone_attns[0], zone_attns[2],                      # Inner vs outer attention
            zone_attns[0] - zone_attns[2],                     # Attention gradient
        ])
        
        # === Spatial gradients using np.gradient (as in original code) ===
        if n_patches > 2:
            # Note: np.gradient on 1D array returns d(immune)/d(index)
            # We compute gradient magnitude based on immune score variation
            immune_gradient = np.gradient(immune_scores)
            gradient_magnitude = np.abs(immune_gradient)
            
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.max(gradient_magnitude),
                np.min(gradient_magnitude),
                np.sum(gradient_magnitude > np.percentile(gradient_magnitude, 75)),  # High gradient count
            ])
            
            # Smoothness measures (roughness ≈ volatility, keep only one)
            if n_patches > 1:
                features.append(np.mean(np.diff(immune_scores)**2))  # Roughness
            else:
                features.append(0)
            
            # Correlations
            if np.std(x_centered) > 0:
                corr_x = np.corrcoef(x_centered, immune_scores)[0, 1]
                features.append(corr_x if not np.isnan(corr_x) else 0)
            else:
                features.append(0)
            
            if np.std(y_centered) > 0:
                corr_y = np.corrcoef(y_centered, immune_scores)[0, 1]
                features.append(corr_y if not np.isnan(corr_y) else 0)
            else:
                features.append(0)
            
            # Note: attn-immune correlation is already in attention_weighted_summaries
            
            if np.std(gradient_magnitude) > 0 and np.std(attn) > 0:
                corr_grad = np.corrcoef(gradient_magnitude, attn)[0, 1]
                features.append(corr_grad if not np.isnan(corr_grad) else 0)
            else:
                features.append(0)
        else:
            features.extend([0] * 10)  # Gradient features (reduced)
        
        # === Progression & Trajectory Analysis (from original code) ===
        if has_attn and n_patches > 2:
            # Sort by attention (importance ordering)
            sorted_indices = np.argsort(attn)[::-1]
            sorted_immune = immune_scores[sorted_indices]
            sorted_x = x_centered[sorted_indices]
            sorted_y = y_centered[sorted_indices]
            
            progressive_diffs = np.diff(sorted_immune)
            
            features.extend([
                np.mean(progressive_diffs),
                np.std(progressive_diffs),
                np.sum(progressive_diffs > 0) / len(progressive_diffs),  # Fraction increasing
                np.max(progressive_diffs),
                np.min(progressive_diffs),
            ])
            
            # Trajectory in space
            features.extend([
                np.mean(np.diff(sorted_x)),
                np.mean(np.diff(sorted_y)),
                np.std(np.diff(sorted_x)),
                np.std(np.diff(sorted_y)),
            ])
            
            # Monotonicity measures (frac_nondec ≈ frac_inc, frac_noninc ≈ 1-frac_inc, so keep only one)
            features.append(
                np.sum(np.abs(progressive_diffs) > np.std(progressive_diffs)) / len(progressive_diffs)  # Volatility
            )
            
            # Range dynamics
            half = n_patches // 2
            if half > 0:
                features.extend([
                    np.ptp(sorted_immune[:half]),    # Range in first half
                    np.ptp(sorted_immune[half:]),   # Range in second half
                ])
            else:
                features.extend([0, 0])
        else:
            features.extend([0] * 12)  # Progression features (reduced)
        
        # === Moran's I (spatial autocorrelation) ===
        if n_patches > 10:
            try:
                coords = np.column_stack([x_centered, y_centered])
                dist_matrix = squareform(pdist(coords))
                
                # Use k-nearest neighbors for weight matrix
                k = min(8, n_patches - 1)
                W = np.zeros((n_patches, n_patches))
                
                for i in range(n_patches):
                    dist_row = dist_matrix[i].copy()
                    dist_row[i] = np.inf
                    nearest = np.argsort(dist_row)[:k]
                    W[i, nearest] = 1
                
                # Row-normalize weights
                row_sums = W.sum(axis=1, keepdims=True)
                W = W / (row_sums + 1e-8)
                
                # Compute Moran's I
                z = immune_scores - np.mean(immune_scores)
                numerator = np.sum(W * np.outer(z, z))
                denominator = np.sum(z ** 2) + 1e-8
                morans_i = (n_patches / (W.sum() + 1e-8)) * (numerator / denominator)
                
                features.append(morans_i if not np.isnan(morans_i) else 0)
            except:
                features.append(0)
        else:
            features.append(0)
        
    else:
        # No spatial coordinates - return zeros
        features.extend([0] * 95)  # Placeholder for all structural features
    
    # Handle NaN values
    features = [0 if (isinstance(f, float) and np.isnan(f)) else f for f in features]
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# Patient-Level Feature Aggregation
# ============================================================================

def aggregate_patient_features(embeddings, gene_predictions, use_attention=True, top_n=50):
    """
    Aggregate patch-level features to patient-level using 4 feature families + flattened features.
    
    This function combines:
    - MUSK embedding statistics (mean pooled from TOP-N patches by attention)
    - Flattened top-N patch features (raw adjusted_x, adjusted_y, immune_score, attention_weight)
    - 4 Spatial Gene Expression Feature Families:
        1. Spatial Tissue Architecture (coordinate stats, pdist-based distances, clustering)
        2. Attention-Weighted Summaries (weighted means, entropy, hotspot dispersion)
        3. Distribution Summaries (percentiles, scipy.stats moments, biological ratios)
        4. Structural Dispersion (median-split quadrants, octants, np.gradient, progression)
    
    IMPORTANT: 
    - Embeddings are mean-pooled from TOP-N patches (selected by attention weight)
    - All spatial coordinates are re-centered relative to the "Centroid" patch 
      (patch with highest attention score) to capture relative topology.
    
    Args:
        embeddings: dict mapping (x, y) coordinates to 1024-dim embedding vectors
        gene_predictions: DataFrame with gene expression predictions (must have x, y, attention_weight)
        use_attention: Whether to use attention-weighted aggregation
        top_n: Number of top patches for flattened features AND embedding aggregation (default: 50)
    
    Returns:
        patient_features: np.ndarray of aggregated features
    """
    if embeddings is None or gene_predictions is None:
        return None
    
    # ===== Re-center coordinates to highest attention patch =====
    x_centered, y_centered, centroid_idx = recenter_coordinates_to_centroid(gene_predictions)
    
    # ===== MUSK Embedding: Mean pooling across TOP-N patches by attention =====
    # Select top-N patches by attention weight and mean pool their embeddings
    if "attention_weight" in gene_predictions.columns and use_attention:
        sorted_df = gene_predictions.sort_values("attention_weight", ascending=False)
        top_indices = sorted_df.head(top_n).index
    else:
        top_indices = gene_predictions.head(top_n).index
    
    # Gather embeddings for top patches
    top_embeddings = []
    for idx in top_indices:
        x = int(gene_predictions.loc[idx, 'x'])
        y = int(gene_predictions.loc[idx, 'y'])
        if (x, y) in embeddings:
            top_embeddings.append(embeddings[(x, y)])
    
    if len(top_embeddings) == 0:
        # Fallback: use any available embeddings if coordinate matching fails
        top_embeddings = list(embeddings.values())[:top_n]
    
    if len(top_embeddings) > 0:
        emb_mean = np.mean(np.stack(top_embeddings), axis=0)
    else:
        emb_mean = np.zeros(1024, dtype=np.float32)
    
    # ===== Flattened top-N patch features =====
    # Creates a fixed-length vector from top attention patches' raw values
    flattened_features = flattened_top_n_features(
        gene_predictions, x_centered, y_centered, top_n=top_n
    )
    
    # ===== 4 Spatial Gene Expression Feature Families =====
    
    # Family 1: Spatial Tissue Architecture
    spatial_arch_features = spatial_tissue_architecture_features(
        gene_predictions, x_centered, y_centered
    )
    
    # Family 2: Attention-Weighted Summaries
    attn_weighted_features = attention_weighted_summaries_features(
        gene_predictions, x_centered, y_centered
    )
    
    # Family 3: Distribution Summaries
    distribution_features = distribution_summaries_features(
        gene_predictions, x_centered, y_centered
    )
    
    # Family 4: Structural Dispersion
    dispersion_features = structural_dispersion_features(
        gene_predictions, x_centered, y_centered
    )
    
    # ===== Combine all features =====
    all_features = [
        emb_mean,               # MUSK embedding (1024-dim, mean pooled)
        flattened_features,     # Flattened top-N patch features
        spatial_arch_features,  # Family 1: Spatial Tissue Architecture
        attn_weighted_features, # Family 2: Attention-Weighted Summaries
        distribution_features,  # Family 3: Distribution Summaries
        dispersion_features,    # Family 4: Structural Dispersion
    ]
    
    patient_features = np.concatenate([f for f in all_features if len(f) > 0])
    
    return patient_features.astype(np.float32)

# ============================================================================
# Dataset
# ============================================================================

class OutcomeDataset(Dataset):
    """Dataset for patient outcome prediction using 4 feature families."""
    
    def __init__(self, csv_file, embedding_dir, gene_pred_dir,
                 outcome_col="outcome", patient_col="patient_id",
                 top_n=50, use_attention=True):
        """
        Args:
            csv_file: CSV with patient IDs and outcomes
            embedding_dir: Directory with pre-computed MUSK embeddings (.npz files)
            gene_pred_dir: Directory with patch-level gene predictions (.csv files)
            outcome_col: Column name for binary outcome
            patient_col: Column name for patient ID
            top_n: Number of top patches for flattened features (default: 50)
            use_attention: Whether to use attention-weighted aggregation
        """
        self.df = pd.read_csv(csv_file)
        self.embedding_dir = embedding_dir
        self.gene_pred_dir = gene_pred_dir
        self.outcome_col = outcome_col
        self.patient_col = patient_col
        self.top_n = top_n
        self.use_attention = use_attention
        
        self.patients = self.df[patient_col].unique().tolist()
        self.labels = {}
        
        for _, row in self.df.iterrows():
            pid = row[patient_col]
            self.labels[pid] = row[outcome_col]
        
        print(f"Outcome Dataset: {len(self.patients)} patients")
        print(f"  - Flattened features: top-{top_n} patches × 4 values = {top_n * 4} features")
        print(f"  - Attention weighting: {use_attention}")
        print(f"  - Coordinates re-centered to highest attention patch (Centroid)")
        print(f"  - 4 feature families: Spatial Architecture, Attention-Weighted, Distribution, Dispersion")

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        label = self.labels[patient_id]
        
        # Load MUSK embeddings
        embeddings = load_musk_embeddings(self.embedding_dir, patient_id)
        
        # Load gene signature predictions (augmented from patch-level model)
        gene_preds = load_gene_predictions(self.gene_pred_dir, patient_id)
        
        # Aggregate to patient-level features using 4 feature families
        features = aggregate_patient_features(
            embeddings, gene_preds, 
            use_attention=self.use_attention, 
            top_n=self.top_n
        )
        
        if features is None:
            return None
        
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "patient_id": patient_id
        }


def collate_fn(batch):
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    
    max_len = max(s["features"].shape[0] for s in batch)
    
    padded_features = []
    for s in batch:
        f = s["features"]
        if f.shape[0] < max_len:
            pad = torch.zeros(max_len - f.shape[0])
            f = torch.cat([f, pad])
        padded_features.append(f)
    
    return {
        "features": torch.stack(padded_features),
        "label": torch.stack([s["label"] for s in batch]),
        "patient_id": [s["patient_id"] for s in batch]
    }

# ============================================================================
# Model
# ============================================================================

class OutcomeMLP(nn.Module):
    """
    MLP for patient outcome prediction.
   
    """
    
    def __init__(self, in_dim, hidden_dim_1=32, hidden_dim_2=16):
        super().__init__()
        
        # Input normalization for stability with small network
        self.input_norm = nn.BatchNorm1d(in_dim)
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim_1),
            nn.ReLU(),
            
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim_2, 1)
        )

    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x).squeeze(-1)

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        if batch is None:
            continue
        
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        
        logits = model(features)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    all_logits, all_labels = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs > 0.5).astype(int)
    
    try:
        auc = roc_auc_score(all_labels, probs)
    except ValueError:
        auc = 0.5
    
    acc = accuracy_score(all_labels, preds)
    
    return {
        "loss": total_loss / max(len(loader), 1),
        "auc": auc,
        "accuracy": acc
    }

# ============================================================================
# Main
# ============================================================================

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config = DEFAULT_CONFIG.copy()
    # Datasets
    train_dataset = OutcomeDataset(
        args.train_csv, args.embedding_dir, args.gene_pred_dir,
        outcome_col=args.outcome_col, patient_col=args.patient_col,
        top_n=args.top_n, use_attention=args.use_attention
    )
    val_dataset = OutcomeDataset(
        args.val_csv, args.embedding_dir, args.gene_pred_dir,
        outcome_col=args.outcome_col, patient_col=args.patient_col,
        top_n=args.top_n, use_attention=args.use_attention
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # Determine input dimension from first batch
    sample_batch = next(iter(train_loader))
    if sample_batch is None:
        raise ValueError("No valid samples found in training data")
    in_dim = sample_batch["features"].shape[1]
    print(f"Input feature dimension: {in_dim}")
    
    model = OutcomeMLP(
        in_dim=in_dim,
        hidden_dim_1=config["hidden_dim_1"],
        hidden_dim_2=config["hidden_dim_2"]
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"],
                            weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
                                                       patience=5, factor=0.5)
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(config["num_epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["auc"])
        
        print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, "
              f"val_auc={val_metrics['auc']:.4f}, "
              f"val_acc={val_metrics['accuracy']:.4f}")
        
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "in_dim": in_dim,
                "config": config,
                "val_metrics": val_metrics
            }, args.save_path)
            print(f"  -> Saved best model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print("Early stopping triggered")
                break
    
    print(f"Training complete. Best validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASPIRE Outcome Prediction Training with 4 Feature Families")
    parser.add_argument("--train_csv", type=str, required=True,
                        help="CSV with patient IDs and outcomes for training")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="CSV with patient IDs and outcomes for validation")
    parser.add_argument("--embedding_dir", type=str, required=True,
                        help="Directory with pre-computed MUSK embeddings (.npz files)")
    parser.add_argument("--gene_pred_dir", type=str, required=True,
                        help="Directory with patch-level gene predictions (.csv files with x, y, immune_score, attention_weight)")
    parser.add_argument("--outcome_col", type=str, default="outcome",
                        help="Column name for binary outcome")
    parser.add_argument("--patient_col", type=str, default="patient_id",
                        help="Column name for patient ID")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Number of top patches for flattened features (default: 50)")
    parser.add_argument("--use_attention", action="store_true", default=True,
                        help="Use attention-weighted aggregation (default: True)")
    parser.add_argument("--no_attention", action="store_false", dest="use_attention",
                        help="Disable attention-weighted aggregation")
    parser.add_argument("--save_path", type=str, default="outcome_checkpoint.pt")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
