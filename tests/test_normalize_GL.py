import pytest
import numpy as np
import simGL


def test_shape_preserved(GL):
    assert simGL.normalize_GL(GL).shape == GL.shape

def test_min_per_individual_is_zero(GL):
    GL_norm = simGL.normalize_GL(GL)
    assert (GL_norm.min(axis=2) == 0.0).all()

def test_relative_differences_preserved(GL):
    # Normalization shifts values but must not change differences between genotypes
    GL_norm = simGL.normalize_GL(GL)
    np.testing.assert_allclose(
        GL[:, :, 1:] - GL[:, :, :-1],
        GL_norm[:, :, 1:] - GL_norm[:, :, :-1],
    )

def test_all_nonnegative(GL):
    assert (simGL.normalize_GL(GL) >= 0).all()
