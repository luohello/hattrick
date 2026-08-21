"""Numerically stable priority-aware gradient projection utilities.

Objectives are ordered from highest to lowest priority. Each lower-priority
gradient is projected onto the orthogonal complement of the subspace protected
by the preceding non-zero gradients before accumulation.
"""

import math

import torch


EPS = 1e-12


def _is_nonzero(gradient, tolerance=EPS):
    return bool(torch.linalg.vector_norm(gradient).detach() > tolerance)


def _orthogonal_component(gradient, protected_gradients):
    protected = [g for g in protected_gradients if _is_nonzero(g)]
    if not protected or not _is_nonzero(gradient):
        return gradient

    matrix = torch.stack(protected, dim=1).to(dtype=torch.float32)
    candidate = gradient.to(dtype=torch.float32)
    # Only a few objectives are protected, so reduced SVD is inexpensive and
    # remains stable when gradients are zero or linearly dependent.
    u, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0:
        return gradient
    threshold = torch.finfo(matrix.dtype).eps * max(matrix.shape) * singular_values.max()
    basis = u[:, singular_values > threshold]
    if basis.numel() == 0:
        return gradient
    projected = candidate - basis @ (basis.T @ candidate)
    return projected.to(dtype=gradient.dtype)


def project_two_grads(flattened_grads1, flattened_grads2):
    return _orthogonal_component(flattened_grads2, [flattened_grads1])


def project_three_grads(flattened_grads1, flattened_grads2, flattened_grads3):
    return _orthogonal_component(flattened_grads3, [flattened_grads1, flattened_grads2])


def project_four_grads(flattened_grads1, flattened_grads2, flattened_grads3, flattened_grads4):
    return _orthogonal_component(
        flattened_grads4, [flattened_grads1, flattened_grads2, flattened_grads3]
    )


def _flatten_autograd(model, loss, retain_graph):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
        materialize_grads=False,
    )
    flat = []
    shapes = []
    for parameter, gradient in zip(parameters, gradients):
        shapes.append(parameter.shape)
        flat.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
        )
    flattened = torch.cat(flat)
    norm = torch.linalg.vector_norm(flattened)
    if torch.isfinite(norm) and norm > 25.0:
        flattened = flattened * (25.0 / norm)
    return flattened, shapes


def project_priority_gradients(model, losses):
    """Return one flattened gradient for an ordered sequence of objectives."""
    if not losses:
        raise ValueError("At least one priority objective is required")

    raw_gradients = []
    shapes = None
    for index, loss in enumerate(losses):
        gradient, current_shapes = _flatten_autograd(
            model, loss, retain_graph=index < len(losses) - 1
        )
        if shapes is None:
            shapes = current_shapes
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Non-finite gradient in priority objective {index + 1}")
        raw_gradients.append(gradient)

    protected = []
    projected = []
    for gradient in raw_gradients:
        component = _orthogonal_component(gradient, protected)
        if _is_nonzero(component):
            protected.append(component)
            projected.append(component)

    final_gradient = (
        torch.stack(projected).sum(dim=0)
        if projected
        else torch.zeros_like(raw_gradients[0])
    )
    return final_gradient, shapes, raw_gradients


def assign_gradients_and_step(model, final_grads, optimizer, shapes):
    if not torch.isfinite(final_grads).all():
        raise FloatingPointError("Cannot apply a non-finite projected gradient")

    offset = 0
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter, shape in zip(trainable_parameters, shapes):
        count = math.prod(shape)
        parameter.grad = final_grads[offset : offset + count].view(shape).to(parameter.dtype)
        offset += count
    if offset != final_grads.numel():
        raise RuntimeError("Projected gradient size does not match model parameters")

    optimizer.step()
    model.zero_grad(set_to_none=True)


def project_gradients_one_optimizer_robust(model, loss1, loss2, loss3, loss4, optimizer):
    """Backward-compatible four-objective entry point used by the original trainer."""
    final, shapes, raw = project_priority_gradients(model, [loss1, loss2, loss3, loss4])
    return final, shapes, raw[0], raw[1], raw[2], optimizer


def find_linearly_independent_vectors(vectors):
    """Compatibility helper retained for downstream imports and tests."""
    if vectors.numel() == 0:
        return torch.zeros(vectors.shape[1], dtype=torch.bool, device=vectors.device)
    _, singular_values, vh = torch.linalg.svd(vectors.double(), full_matrices=False)
    threshold = torch.finfo(torch.float64).eps * max(vectors.shape) * singular_values.max()
    rank = int((singular_values > threshold).sum().item())
    scores = vh[:rank].abs().sum(dim=0) if rank else torch.zeros(vectors.shape[1], device=vectors.device)
    keep = torch.zeros(vectors.shape[1], dtype=torch.bool, device=vectors.device)
    if rank:
        keep[torch.topk(scores, rank).indices] = True
    return keep
