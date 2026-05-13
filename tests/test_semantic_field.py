from __future__ import annotations

import numpy as np
import pytest

from tsf.semantic_field import (
    RestrictedSemanticInputLayer,
    SemanticFieldSpec,
    construct_semantic_field,
    validate_feature_order,
)


def test_semantic_field_matches_weighted_sum() -> None:
    directions = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [-1.0, 1.0],
        ],
        dtype=np.float64,
    )
    x = np.asarray([[0.5, 2.0, -1.0]], dtype=np.float64)

    field = construct_semantic_field(x, directions)

    np.testing.assert_allclose(field, np.asarray([[1.5, 3.0]], dtype=np.float64))


def test_semantic_field_spec_uses_direct_embeddings() -> None:
    embeddings = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)

    spec = SemanticFieldSpec.from_embeddings(
        variable_names=["gas", "air"],
        descriptions=["fuel-side flow", "oxidizer-side flow"],
        embeddings=embeddings,
        embedding_model="test-embedding",
        normalize=True,
    )

    np.testing.assert_allclose(spec.embeddings, embeddings)
    np.testing.assert_allclose(
        spec.directions,
        np.asarray([[0.6, 0.8], [0.0, 1.0]], dtype=np.float64),
    )
    assert spec.semantic_matrix.shape == (2, 2)


def test_restricted_input_layer_preserves_shape() -> None:
    spec = SemanticFieldSpec.from_embeddings(
        variable_names=["a", "b", "c"],
        descriptions=["first variable", "second variable", "third variable"],
        embeddings=np.ones((3, 5), dtype=np.float64),
        embedding_model="test-embedding",
    )
    layer = RestrictedSemanticInputLayer.initialize(n_variables=3, semantic_dim=5, seed=11)
    x = np.ones((2, 4, 3), dtype=np.float64)

    output = layer.transform(x, spec.directions)

    assert output.shape == x.shape


def test_semantic_field_artifact_round_trip(tmp_path) -> None:
    embeddings = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    spec = SemanticFieldSpec.from_embeddings(
        variable_names=["gas", "air"],
        descriptions=["fuel-side flow", "oxidizer-side flow"],
        embeddings=embeddings,
        embedding_model="test-embedding",
    )

    spec.save_artifact(tmp_path)
    loaded = SemanticFieldSpec.load_artifact(tmp_path, feature_names=["gas", "air"])

    assert loaded.variable_names == spec.variable_names
    assert loaded.descriptions == spec.descriptions
    np.testing.assert_allclose(loaded.embeddings, spec.embeddings)
    np.testing.assert_allclose(loaded.directions, spec.directions)
    assert not (tmp_path / "projection.npy").exists()


def test_feature_order_must_match_semantic_variables() -> None:
    validate_feature_order(feature_names=["gas", "air"], semantic_variable_names=["gas", "air"])

    with pytest.raises(ValueError, match="feature_names must match"):
        validate_feature_order(feature_names=["air", "gas"], semantic_variable_names=["gas", "air"])
