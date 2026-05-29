"""
Tests for coordinate adjustment module.

These tests verify:
1. Strand normalization
2. Hardcoded SpliceAI adjustments (np.roll)
3. Custom adjustment application
4. Caching functions
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import json

from agentic_spliceai.splice_engine.base_layer.utils import (
    normalize_strand,
    adjust_scores_hardcoded,
    apply_custom_adjustments,
    get_adjustment_cache_path,
    load_cached_adjustments,
    save_adjustments_to_cache,
)


class TestStrandNormalization:
    """Test strand notation normalization."""
    
    def test_plus_strand_variants(self):
        """Test all accepted plus strand notations."""
        assert normalize_strand('+') == '+'
        assert normalize_strand('1') == '+'
        assert normalize_strand(1) == '+'
        assert normalize_strand('plus') == '+'
        assert normalize_strand('forward') == '+'
    
    def test_minus_strand_variants(self):
        """Test all accepted minus strand notations."""
        assert normalize_strand('-') == '-'
        assert normalize_strand('-1') == '-'
        assert normalize_strand(-1) == '-'
        assert normalize_strand('minus') == '-'
        assert normalize_strand('reverse') == '-'
    
    def test_invalid_strand(self):
        """Test that invalid strands raise ValueError."""
        with pytest.raises(ValueError):
            normalize_strand('invalid')


class TestHardcodedAdjustments:
    """Test SpliceAI hardcoded adjustments."""
    
    def test_donor_plus_strand(self):
        """Test donor + strand adjustment (+2bp)."""
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.0])
        adjusted = adjust_scores_hardcoded(scores, '+', 'donor')
        
        # Peak should shift from index 2 to index 4 (roll +2)
        assert adjusted[4] == 0.9
        # First 2 positions should be zeroed
        assert adjusted[0] == 0.0
        assert adjusted[1] == 0.0
    
    def test_donor_minus_strand(self):
        """Test donor - strand adjustment (+1bp)."""
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.0])
        adjusted = adjust_scores_hardcoded(scores, '-', 'donor')
        
        # Peak should shift from index 2 to index 3 (roll +1)
        assert adjusted[3] == 0.9
        # First position should be zeroed
        assert adjusted[0] == 0.0
    
    def test_acceptor_plus_strand(self):
        """Test acceptor + strand (no adjustment)."""
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.0])
        adjusted = adjust_scores_hardcoded(scores, '+', 'acceptor')
        
        # Should be unchanged
        assert np.array_equal(adjusted, scores)
    
    def test_acceptor_minus_strand(self):
        """Test acceptor - strand adjustment (-1bp)."""
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.0])
        adjusted = adjust_scores_hardcoded(scores, '-', 'acceptor')
        
        # Peak should shift from index 2 to index 1 (roll -1)
        assert adjusted[1] == 0.9
        # Last position should be zeroed
        assert adjusted[-1] == 0.0
    
    def test_neither_prob_handling(self):
        """Test that neither probabilities get set to 1.0 at edges."""
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.0])
        adjusted = adjust_scores_hardcoded(scores, '+', 'donor', is_neither_prob=True)
        
        # First 2 positions should be 1.0 (not 0.0)
        assert adjusted[0] == 1.0
        assert adjusted[1] == 1.0


class TestCustomAdjustments:
    """Test custom adjustment dictionary application."""
    
    def test_custom_donor_adjustment(self):
        """Test applying custom donor adjustment."""
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.0])
        adjustment_dict = {
            'donor': {'plus': 3, 'minus': 0},
            'acceptor': {'plus': 0, 'minus': 0}
        }
        
        adjusted = apply_custom_adjustments(scores, '+', 'donor', adjustment_dict)
        
        # Peak should shift by +3
        assert adjusted[5] == 0.9
        assert adjusted[0] == 0.0
        assert adjusted[1] == 0.0
        assert adjusted[2] == 0.0
    
    def test_zero_adjustment(self):
        """Test that zero adjustment returns unchanged scores."""
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.0])
        adjustment_dict = {
            'donor': {'plus': 0, 'minus': 0},
            'acceptor': {'plus': 0, 'minus': 0}
        }
        
        adjusted = apply_custom_adjustments(scores, '+', 'donor', adjustment_dict)
        
        assert np.array_equal(adjusted, scores)
    
    def test_none_adjustment_dict(self):
        """Test that None adjustment dict returns unchanged scores."""
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.0])
        
        adjusted = apply_custom_adjustments(scores, '+', 'donor', None)
        
        assert np.array_equal(adjusted, scores)


class TestCaching:
    """Test caching functions."""
    
    def test_save_and_load(self):
        """Test saving and loading adjustments from cache."""
        adjustments = {
            'donor': {'plus': 2, 'minus': 1},
            'acceptor': {'plus': 0, 'minus': -1}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_adjustments.json"
            
            # Save
            save_adjustments_to_cache(adjustments, cache_path, verbose=False)
            assert cache_path.exists()
            
            # Load
            loaded = load_cached_adjustments(cache_path, verbose=False)
            assert loaded == adjustments
    
    def test_load_nonexistent(self):
        """Test loading from nonexistent file returns None."""
        cache_path = Path("/tmp/nonexistent_cache_file_12345.json")
        
        loaded = load_cached_adjustments(cache_path, verbose=False)
        assert loaded is None
    
    def test_cache_path_construction(self):
        """Test that cache path is constructed correctly."""
        from agentic_spliceai.splice_engine.resources import get_model_resources
        
        # Test with known model
        cache_path = get_adjustment_cache_path('openspliceai', 'GRCh38', 'mane')
        
        assert 'GRCh38' in str(cache_path)
        assert 'mane' in str(cache_path)
        assert 'openspliceai_coordinate_adjustments.json' in str(cache_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
