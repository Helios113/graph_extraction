from typing import Protocol

import numpy as np
import onnxruntime as ort
import tqdm

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

    def __init__(self, lr: float = 1e-1, dtype: np.dtype = np.float64):
        self.lr = np.array(lr, dtype=dtype)

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
        dtype: np.dtype = np.float64,
    ):
        self.lr = np.array(lr, dtype=dtype)
        self.beta1, self.beta2 = (np.array(b, dtype=dtype) for b in betas)
        self.eps = np.array(eps, dtype=dtype)
        self.weight_decay = np.array(weight_decay, dtype=dtype)
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


class GaussNewton:
    """GaussNewton root-finding on the gradient (i.e. Newton's method for
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

    def __init__(self, lr: float = 1e-1, damping: float = 1e-4, dtype: np.dtype = np.float64):
        self.lr = np.array(lr, dtype=dtype)
        self.damping = np.array(damping, dtype=dtype)
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
    
    
class NewtonRaphson:
    """Newton-Raphson root-finding on the gradient (i.e. Newton's method for
    minimization): y -= f(y)/ grad f(y).
    """

    def __init__(self, lr: float = 1e-1, damping: float = 1e-4, dtype: np.dtype = np.float64):
        self.lr = np.array(lr, dtype=dtype)
        self.damping = np.array(damping, dtype=dtype)
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
    x: np.ndarray,
    target: np.ndarray,
    y0: np.ndarray,
    input_name: str,
    steps: int,
    optimizer: Optimizer,
) -> tuple[np.ndarray, np.ndarray]:
   
    print(x.shape)
    print(target.shape)
    print(y0.shape)
    
    # Just iterate over y and then no need to copy the x and target
    loss_history = np.zeros((steps, *y0.shape[:-1]))
    y_n = np.zeros_like(y0)
    for i in range(y0.shape[0]):
        y = y0[i].copy()

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

        for s in tqdm.tqdm(range(steps)):
            feed = {input_name: y, "target": target,"x": x}
            loss_val, grad_y = gradient_session.run(
                [loss_output_name, grad_output_name], feed,
            )

            loss_history[s, i] = loss_val
            y = optimizer.step(y, grad_y)
        y_n[i] = y
        print(loss_history[:,i,:,:].mean())
    return y_n, loss_history
