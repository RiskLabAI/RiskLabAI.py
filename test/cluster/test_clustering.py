"""
Tests for cluster/clustering.py
"""

import numpy as np
import pandas as pd
import pytest
from RiskLabAI.cluster.clustering import (
    covariance_to_correlation,
    random_block_correlation,
    cluster_k_means_base,
    cluster_k_means_top,
)

def test_covariance_to_correlation():
    """Test correlation matrix derivation."""
    cov = np.array([[2.0, 1.0], [1.0, 1.0]])
    # std = [sqrt(2), 1]
    # corr[0,1] = 1 / (sqrt(2) * 1) = 1/sqrt(2) = 0.7071
    corr = covariance_to_correlation(cov)
    
    expected = np.array([[1.0, 1 / np.sqrt(2)], [1 / np.sqrt(2), 1.0]])
    assert np.allclose(corr, expected)
    
    # Test numerical stability
    cov_unstable = np.array([[1.0, 1.000001], [1.000001, 1.0]])
    corr_unstable = covariance_to_correlation(cov_unstable)
    assert corr_unstable.max() <= 1.0
    assert corr_unstable.min() >= -1.0

@pytest.fixture
def block_corr_matrix():
    """Generate a known block correlation matrix for testing."""
    corr = random_block_correlation(
        n_columns=10, n_blocks=2, block_size_min=5, random_state=0
    )
    # With block_size_min=5, this forces two blocks of 5
    return corr

def test_random_block_correlation(block_corr_matrix):
    """Test the block correlation generator."""
    corr = block_corr_matrix
    assert corr.shape == (10, 10)
    
    # Check that intra-block correlation is high
    intra_block_1 = corr.iloc[0:5, 0:5].to_numpy()
    intra_block_2 = corr.iloc[5:10, 5:10].to_numpy()
    
    # Check that inter-block correlation is lower
    inter_block = corr.iloc[0:5, 5:10].to_numpy()
    
    # Avg corr inside block (minus diagonal)
    avg_intra_1 = (intra_block_1.sum() - 5) / (25 - 5)
    avg_intra_2 = (intra_block_2.sum() - 5) / (25 - 5)
    
    # Avg corr between blocks
    avg_inter = inter_block.mean()
    
    assert avg_intra_1 > avg_inter
    assert avg_intra_2 > avg_inter

def test_cluster_k_means_base(block_corr_matrix):
    """Test the base clustering logic."""
    corr = block_corr_matrix
    
    # We know there are 2 blocks, so set max_clusters low
    corr_sorted, clusters, silhouette = cluster_k_means_base(
        corr, max_clusters=2, iterations=10, random_state=0
    )
    
    # Should find 2 clusters
    assert len(clusters) == 2
    
    # Clusters should be the two blocks
    c0 = clusters[0]
    c1 = clusters[1]
    
    block1 = [0, 1, 2, 3, 4]
    block2 = [5, 6, 7, 8, 9]
    
    # Convert item names (which are ints) to list
    c0_names = sorted([int(c) for c in c0])
    c1_names = sorted([int(c) for c in c1])
    
    assert (c0_names == block1 and c1_names == block2) or \
           (c0_names == block2 and c1_names == block1)
           
    assert silhouette.shape == (10,)
    assert corr_sorted.shape == (10, 10)

def test_cluster_k_means_top(block_corr_matrix):
    """Test the top-level ONC algorithm."""
    corr = block_corr_matrix
    
    corr_sorted, clusters, silhouette = cluster_k_means_top(
        corr, max_clusters=10, iterations=10, random_state=0
    )
    
    # ONC should be stable and find the 2 clusters
    assert len(clusters) == 2
    
    block1 = [str(i) for i in range(5)]
    block2 = [str(i) for i in range(5, 10)]
    
    c0_names = sorted(clusters[0])
    c1_names = sorted(clusters[1])

    assert (c0_names == block1 and c1_names == block2) or \
           (c0_names == block2 and c1_names == block1)