from typing import Protocol

import numpy as np
import onnxruntime as ort


class Optimizer(Protocol):
    """An update rule for optimization_loop: given the current iterate and the
    loss gradient at that iterate, returns the next iterate. Every optimizer
    (SGD, AdamW, Newton-Raphson, ...) is a class implementing this same `step`
    method, so optimization_loop can treat them all identically -- state (Adam's
    moment buffers, NR's previous grad/y for its curvature estimate, ...) lives
    on `self` rather than being threaded through the loop. Construct a fresh
    instance per y being optimized: state is shaped to the first grad it sees,
    so one instance must not be reused across ys of different shape.
    """

    def step(self, y: np.ndarray, grad: np.ndarray) -> np.ndarray: ...


class SGD:
    """Plain gradient descent: y -= lr * grad."""

    def __init__(self, lr: float = 1e-1):
        self.lr = lr

    def step(self, y: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return y - self.lr * grad


class AdamW:
    """AdamW, matching torch.optim.AdamW's update rule."""

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._t = 0

    def step(self, y: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(grad)
            self._v = np.zeros_like(grad)
        self._t += 1

        y = y - self.lr * self.weight_decay * y

        self._m = self.beta1 * self._m + (1 - self.beta1) * grad
        self._v = self.beta2 * self._v + (1 - self.beta2) * (grad * grad)
        m_hat = self._m / (1 - self.beta1**self._t)
        v_hat = self._v / (1 - self.beta2**self._t)

        return y - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class NewtonRaphson:
    """Newton-Raphson root-finding on the gradient (i.e. Newton's method for
    minimization): y -= H^-1 grad. The gradient graph only gives a first-order
    oracle (d(loss)/d(y), no Hessian), so H is never formed exactly -- instead
    this estimates a diagonal Hessian from the secant equation applied
    elementwise, H_diag ~= (grad - grad_prev) / (y - y_prev), using the previous
    step's (y, grad) as the secant point. This is the same "diagonal secant /
    quasi-Newton" trick behind e.g. the Barzilai-Borwein step size.

    The first call has no previous point to form a secant from, so it falls
    back to a plain gradient step scaled by `lr` to get one going.

    damping is added to |H_diag| before dividing, both to avoid division by
    ~0 where the secant denominator vanishes (a flat direction in y) and to
    keep the step bounded early on when the curvature estimate is still noisy.
    """

    def __init__(self, lr: float = 1e-1, damping: float = 1e-4):
        self.lr = lr
        self.damping = damping
        self._y_prev: np.ndarray | None = None
        self._grad_prev: np.ndarray | None = None

    def step(self, y: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self._y_prev is None:
            y_next = y - self.lr * grad
        else:
            dy = y - self._y_prev
            dgrad = grad - self._grad_prev
            h_diag = dgrad / np.where(dy == 0, self.damping, dy)
            y_next = y - grad / (np.abs(h_diag) + self.damping)

        self._y_prev = y
        self._grad_prev = grad
        return y_next


def optimization_loop(
    gradient_session: ort.InferenceSession,
    y0: np.ndarray,
    target: np.ndarray,
    input_name: str,
    steps: int,
    optimizer: Optimizer,
    x: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Runs a manual adversarial search: at each step, evaluates the
    AdversarialLoss_* training graph at the current `y` to get both the loss
    value and d(loss)/d(y), then hands (y, grad) to `optimizer.step` to get the
    next iterate.

    A manual update loop is used instead of onnxblock's SGD/AdamW optimizer
    graph because that graph is built from the trainable *parameters* of the
    model (graph initializers) -- input_name is a plain runtime input, never
    registered as trainable (construct.pullback.generate_loss's
    _add_loss_inputs call never sets `trainable=` for it), so onnxblock's
    optimizer build never sees it as a parameter to update.

    Parameters:
        gradient_session: ORT session for the AdversarialLoss_* training graph
            (construct.pullback.generate_subgraph_pullback's "gradient" output).
        y0: Initial value for the input being optimized, matching input_name's
            graph shape.
        target: Fixed target output (e.g. f(x)) fed to the loss each step.
        input_name: Name of the graph input being optimized (SublayerConfig.input).
        steps: Number of update steps to run.
        optimizer: Update rule applied each step as `y = optimizer.step(y, grad)`.
            See SGD, AdamW, NewtonRaphson -- build your own (momentum, RMSProp,
            ...) by matching the same Optimizer.step(y, grad) -> y_new interface.
            Construct a fresh instance per call (state is shaped to the first
            grad it sees).
        x: Fixed anchor input, required by AdversarialLoss_NormalizedSquaredDiffLoss
            and AdversarialLoss_EnvelopeDiffLoss (both take {x, input_name, target}
            as three distinct graph inputs). AdversarialLoss_SquaredDiffLoss only
            takes {input_name, target} and doesn't use x -- pass None for that loss
            type. Fed automatically only if the graph declares an "x" input that is
            distinct from input_name (input_name can itself be "x" for MaskedSumLoss/
            SquaredDiffLoss-style graphs, in which case no separate anchor exists).

    Returns:
        (y_final, loss_history): the optimized input and the per-step loss values.
    """
    y = y0.copy()
    loss_history = []

    grad_output_name = f"{input_name}_grad"
    session_output_names = {o.name for o in gradient_session.get_outputs()}
    if grad_output_name not in session_output_names:
        raise ValueError(
            f"{grad_output_name!r} not found in gradient_session outputs: "
            f"{sorted(session_output_names)}",
        )
    loss_output_name = next(
        name for name in session_output_names if name != grad_output_name
    )

    session_input_names = {i.name for i in gradient_session.get_inputs()}
    needs_anchor = "x" in session_input_names and input_name != "x"
    if needs_anchor and x is None:
        raise ValueError(
            "gradient_session declares an 'x' input distinct from input_name "
            "but no x was provided",
        )

    for _ in range(steps):
        feed = {input_name: y, "target": target}
        if needs_anchor:
            feed["x"] = x

        loss_val, grad_y = gradient_session.run(
            [loss_output_name, grad_output_name], feed,
        )
        loss_history.append(float(loss_val))

        y = optimizer.step(y, grad_y)

    return y, loss_history
