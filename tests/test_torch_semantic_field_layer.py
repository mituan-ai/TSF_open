from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch method-layer tests require the train extra")

from tsf.methods.semantic_field_layer import SemanticFieldForwardOutput, SemanticFieldLayer
from tsf.semantic_field import SemanticFieldSpec, construct_semantic_field


def test_torch_semantic_field_matches_numpy() -> None:
    directions = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [-1.0, 1.0],
        ],
        dtype=np.float64,
    )
    x_np = np.asarray([[[0.5, 2.0, -1.0], [1.0, 0.0, 0.25]]], dtype=np.float64)
    expected = construct_semantic_field(x_np, directions)
    layer = SemanticFieldLayer(directions=directions, dtype=torch.float64)

    semantic_field = layer.compute_semantic_field(torch.as_tensor(x_np, dtype=torch.float64))

    np.testing.assert_allclose(semantic_field.detach().numpy(), expected)


def test_torch_semantic_directions_are_buffer_not_parameter() -> None:
    spec = SemanticFieldSpec.from_embeddings(
        variable_names=["a", "b"],
        descriptions=["first variable", "second variable"],
        embeddings=np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.float64),
        embedding_model="test-embedding",
    )
    layer = SemanticFieldLayer.from_spec(spec, feature_names=["a", "b"])

    assert "directions" in dict(layer.named_buffers())
    assert "directions" not in dict(layer.named_parameters())
    assert "semantic_projection.weight" in dict(layer.named_parameters())
    assert "numeric_scale" in dict(layer.named_parameters())


def test_torch_diagonal_numeric_path_preserves_default_shape() -> None:
    directions = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    layer = SemanticFieldLayer(directions=directions, dtype=torch.float64)
    x = torch.ones((2, 3, 2), dtype=torch.float64)

    output = layer(x, return_components=True)

    assert isinstance(output, SemanticFieldForwardOutput)
    assert output.transformed.shape == x.shape
    assert output.semantic_field.shape == (2, 3, 2)
    parameter_shapes = {name: tuple(parameter.shape) for name, parameter in layer.named_parameters()}
    assert parameter_shapes["numeric_scale"] == (2,)
    assert parameter_shapes["semantic_projection.weight"] == (2, 2)
    assert all(
        shape != (2, 2) or name == "semantic_projection.weight"
        for name, shape in parameter_shapes.items()
    )


def test_torch_output_dim_uses_restricted_numeric_residual() -> None:
    directions = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    layer = SemanticFieldLayer(directions=directions, output_dim=4, dtype=torch.float64)
    x = torch.ones((2, 3, 2), dtype=torch.float64)

    output = layer(x, return_components=True)

    assert isinstance(output, SemanticFieldForwardOutput)
    assert output.transformed.shape == (2, 3, 4)
    assert output.numeric_residual.shape == (2, 3, 4)
    assert "numeric_adapter.weight" not in dict(layer.named_parameters())
    assert dict(layer.named_parameters())["numeric_scale"].shape == (2,)


def test_torch_feature_order_mismatch_raises() -> None:
    spec = SemanticFieldSpec.from_embeddings(
        variable_names=["gas", "air"],
        descriptions=["fuel-side flow", "oxidizer-side flow"],
        embeddings=np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.float64),
        embedding_model="test-embedding",
    )

    with pytest.raises(ValueError, match="feature_names must match"):
        SemanticFieldLayer.from_spec(spec, feature_names=["air", "gas"])
