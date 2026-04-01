"""IBM QAOA solver wrapper for quantum blockchain mining.

Analogous to dwave_sampler.py — encapsulates all QAOA-specific logic
(cost operator construction, circuit building, variational optimization,
result conversion) behind a simple (h, J) → dimod.SampleSet interface.
"""
from __future__ import annotations

import logging
import multiprocessing
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, cast

import dimod
import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from shared.quantum_proof_of_work import energy_of_solution
from shared.miner_types import Variable

logger = logging.getLogger(__name__)


class SolveCancelled(Exception):
    """Raised inside the optimizer objective when stop_event fires."""
    pass


# ---------------------------------------------------------------------------
# Optimizer configurations
# ---------------------------------------------------------------------------

OPTIMIZER_CONFIGS = {
    'COBYLA': {
        'method': 'COBYLA',
        'options': {'maxiter': 100, 'rhobeg': 0.5},
        'description': 'Gradient-free, good default for noisy landscapes',
    },
    'NELDER_MEAD': {
        'method': 'Nelder-Mead',
        'options': {'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4},
        'description': 'Gradient-free simplex, robust but slower',
    },
    'POWELL': {
        'method': 'Powell',
        'options': {'maxiter': 100},
        'description': 'Gradient-free direction-set method',
    },
    'L_BFGS_B': {
        'method': 'L-BFGS-B',
        'options': {'maxiter': 100},
        'description': 'Gradient-based, fast if landscape is smooth',
    },
}


class SPSAOptimizer:
    """Simultaneous Perturbation Stochastic Approximation.

    Designed for noisy objective functions — only needs 2 evaluations per
    iteration regardless of parameter count.  Preferred on real QPU hardware
    where every evaluation carries shot noise.
    """

    def __init__(
        self,
        maxiter: int = 100,
        a: float = 0.1,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
    ):
        self.maxiter = maxiter
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma

    def minimize(self, objective, x0, stop_event=None):
        params = np.array(x0, dtype=float)
        best_params = params.copy()
        best_value = objective(params)

        for k in range(1, self.maxiter + 1):
            if stop_event and stop_event.is_set():
                break

            ak = self.a / (k ** self.alpha)
            ck = self.c / (k ** self.gamma)

            delta = np.random.choice([-1, 1], size=len(params))

            params_plus = params + ck * delta
            params_minus = params - ck * delta
            plus_val = objective(params_plus)
            minus_val = objective(params_minus)

            gradient = (plus_val - minus_val) / (2 * ck * delta)
            params = params - ak * gradient

            if plus_val < best_value:
                best_value = plus_val
                best_params = params_plus.copy()
            if minus_val < best_value:
                best_value = minus_val
                best_params = params_minus.copy()

        return _OptResult(x=best_params, fun=best_value, nfev=self.maxiter * 2)


@dataclass
class _OptResult:
    """Minimal result object matching scipy.optimize.OptimizeResult shape."""
    x: np.ndarray
    fun: float
    nfev: int


# ---------------------------------------------------------------------------
# QAOA Future — lightweight wrapper for async solve
# ---------------------------------------------------------------------------

class QAOAFuture:
    """Future-like object for asynchronous QAOA execution via multiprocessing.

    Wraps a multiprocessing.Process and Queue to provide the same polling
    interface as dwave_sampler.EmbeddedFuture, so the miner can use a
    consistent pattern.
    """

    def __init__(
        self,
        process: multiprocessing.Process,
        result_queue: multiprocessing.Queue,
        solve_start: float,
    ):
        self._process = process
        self._result_queue = result_queue
        self._solve_start = solve_start
        self._result: Optional[dimod.SampleSet] = None
        self._done = False

    @property
    def sampleset(self) -> Optional[dimod.SampleSet]:
        """Block until the QAOA solve completes and return the SampleSet.

        Returns None if the solve was interrupted, terminated, or the
        child process died without returning a result.
        """
        if not self._done:
            try:
                self._result = self._result_queue.get(timeout=5)
            except Exception:
                # Child process died without putting a result
                if not self._process.is_alive():
                    logger.error("[QAOA] Child process died without returning a result")
                    self._result = None
                else:
                    # Still running, keep waiting
                    self._result = self._result_queue.get()
            self._process.join()
            self._done = True
        return self._result

    def done(self) -> bool:
        if self._done:
            return True
        if not self._result_queue.empty():
            return True
        return not self._process.is_alive()

    def cancel(self) -> bool:
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        self._done = True
        return True

    def wait(self, timeout: Optional[float] = None):
        self._process.join(timeout=timeout)

    @property
    def elapsed(self) -> float:
        return time.time() - self._solve_start

    def __hash__(self):
        return id(self._process)

    def __eq__(self, other):
        if isinstance(other, QAOAFuture):
            return self._process is other._process
        return False


# ---------------------------------------------------------------------------
# Module-level solve function for multiprocessing
# ---------------------------------------------------------------------------

def _qaoa_solve_in_process(
    solver_config: Dict[str, Any],
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    stop_event,
    params: Optional[Dict[str, Any]],
    result_queue: multiprocessing.Queue,
):
    """Run a QAOA solve in a child process.

    This is a module-level function (not a method) so that
    multiprocessing.Process can pickle it.  A fresh QAOASolverWrapper is
    constructed from ``solver_config`` inside the child process.

    Args:
        solver_config: Serializable kwargs for QAOASolverWrapper.__init__.
        h: Linear biases.
        J: Quadratic biases.
        stop_event: multiprocessing.Event for cooperative cancellation.
        params: Per-solve parameter overrides (p, shots, etc.).
        result_queue: Queue to send the resulting SampleSet back.
    """
    try:
        solver = QAOASolverWrapper(**solver_config)
        sampleset = solver.solve_ising(h, J, stop_event=stop_event, params=params)
        result_queue.put(sampleset)
    except Exception as e:
        logger.error(f"[QAOA] Process solve failed: {e}")


# ---------------------------------------------------------------------------
# Main solver wrapper
# ---------------------------------------------------------------------------

class QAOASolverWrapper:
    """Wrapper that presents (h, J) → dimod.SampleSet via QAOA.

    Encapsulates cost-operator construction, circuit building, variational
    optimization, and result conversion.  Higher-level code (IBMQAOAMiner)
    interacts only through ``solve_ising`` / ``solve_ising_async``.
    """

    def __init__(
        self,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        backend: Optional[Any] = None,
        p: int = 1,
        optimizer: str = 'COBYLA',
        shots: int = 1024,
        final_shots: Optional[int] = None,
        max_iter: Optional[int] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
        optimization_level: int = 1,
    ):
        """
        Args:
            nodes: Logical node list for the Ising topology.
            edges: Logical edge list for the Ising topology.
            backend: Qiskit backend. Defaults to AerSimulator if None.
            p: QAOA circuit depth (number of layers). Higher = better
               quality but deeper circuit.
            optimizer: One of 'COBYLA', 'NELDER_MEAD', 'POWELL',
                      'L_BFGS_B', 'SPSA'.
            shots: Shots per circuit evaluation during optimization.
            final_shots: Shots for the final sampling run after optimization
                        completes.  Defaults to 4 × shots.
            max_iter: Override default max iterations for the optimizer.
            optimizer_options: Additional overrides merged with defaults.
            optimization_level: Qiskit transpiler optimization level (0-3).
        """
        # Topology (protocol-level, same graph as D-Wave)
        self.nodes: List[int] = nodes
        self.edges: List[Tuple[int, int]] = edges

        # Backend
        if backend is None:
            self.backend = AerSimulator()
            logger.info("[QAOA] Using AerSimulator (local)")
        else:
            self.backend = backend
            logger.info(f"[QAOA] Using backend: {backend.name}")

        self.is_simulator = isinstance(self.backend, AerSimulator)

        # QAOA parameters
        self.p = p
        self.shots = shots
        self.final_shots = final_shots or (4 * shots)
        self.optimization_level = optimization_level

        # Optimizer setup
        if optimizer not in OPTIMIZER_CONFIGS and optimizer != 'SPSA':
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Choose from: {list(OPTIMIZER_CONFIGS.keys()) + ['SPSA']}"
            )

        self.optimizer_name = optimizer

        if optimizer == 'SPSA':
            default_opts = {'maxiter': 100, 'a': 0.1, 'c': 0.1}
        else:
            default_opts = OPTIMIZER_CONFIGS[optimizer]['options'].copy()

        if max_iter is not None:
            default_opts['maxiter'] = max_iter
        if optimizer_options:
            default_opts.update(optimizer_options)
        self.optimizer_options = default_opts

        # Expose sampler-like properties for BaseMiner compatibility
        self.sampler_type = "qaoa"
        self.is_qpu = not self.is_simulator

        # Satisfy Sampler protocol (required by BaseMiner)
        self.nodelist: List[Variable] = cast(List[Variable], nodes)
        self.edgelist: List[Tuple[Variable, Variable]] = cast(List[Tuple[Variable, Variable]], edges)
        self.properties: Dict[str, Any] = {
            'backend': self.backend.name if hasattr(self.backend, 'name') else 'aer_simulator',
            'p': p,
            'optimizer': optimizer,
            'shots': shots,
        }

        logger.info(
            f"[QAOA] Solver ready: {len(nodes)} nodes, {len(edges)} edges, "
            f"p={p}, optimizer={optimizer}, shots={shots}"
        )

    # -- Core QAOA pipeline -------------------------------------------------

    def _build_cost_operator(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
    ) -> SparsePauliOp:
        """Convert (h, J) Ising parameters to a SparsePauliOp cost Hamiltonian.

        The Ising energy  E = Σ_i h_i Z_i  +  Σ_{ij} J_ij Z_i Z_j
        maps directly to a sum of Pauli-Z operators.
        """
        n = len(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        pauli_list = []

        # Single-qubit terms: h_i * Z_i
        for node, bias in h.items():
            if bias != 0.0:
                label = ['I'] * n
                label[node_to_idx[node]] = 'Z'
                pauli_list.append((''.join(label), bias))

        # Two-qubit terms: J_ij * Z_i Z_j
        for (u, v), coupling in J.items():
            label = ['I'] * n
            label[node_to_idx[u]] = 'Z'
            label[node_to_idx[v]] = 'Z'
            pauli_list.append((''.join(label), coupling))

        return SparsePauliOp.from_list(pauli_list)

    def _build_circuit(self, cost_op: SparsePauliOp, p: Optional[int] = None) -> Any:
        """Build a parameterized QAOA ansatz from the cost operator.

        Args:
            cost_op: The cost Hamiltonian as a SparsePauliOp.
            p: Circuit depth (number of QAOA layers). If None, uses self.p.
        """
        if p is None:
            p = self.p
        ansatz = QAOAAnsatz(cost_operator=cost_op, reps=p)
        return ansatz

    def _transpile(self, circuit: Any) -> Any:
        """Transpile circuit for the target backend."""
        pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level,
            backend=self.backend,
        )
        return pm.run(circuit)

    def _evaluate_circuit(self, circuit, params: np.ndarray) -> float:
        """Bind parameters, run the circuit, and return expectation energy.

        Used as the objective function inside the classical optimizer loop.
        Uses self._solve_shots (set per-solve) for shot count.
        """
        bound = circuit.assign_parameters(params)
        bound.measure_all()

        shots = getattr(self, '_solve_shots', self.shots)
        job = self.backend.run(bound, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Compute expectation value:  <E> = Σ_x  P(x) · E(x)
        total_shots = sum(counts.values())
        expectation = 0.0
        node_list = self.nodes

        for bitstring, count in counts.items():
            # Qiskit bitstrings are little-endian: qubit 0 is rightmost
            spins = [1 - 2 * int(b) for b in reversed(bitstring)]
            energy = energy_of_solution(spins, self._current_h, self._current_J, node_list)
            expectation += (count / total_shots) * energy

        return expectation

    def _optimize(
        self,
        circuit: Any,
        stop_event: Optional[Any] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """Run the classical optimizer to find optimal QAOA angles.

        Args:
            circuit: Transpiled QAOA circuit.
            stop_event: Event for cooperative cancellation.
            optimizer_options: Per-solve optimizer options (may override __init__ defaults).

        Returns the best parameter vector, or None if interrupted.
        """
        if optimizer_options is None:
            optimizer_options = self.optimizer_options

        iteration_count = 0

        def objective(params):
            nonlocal iteration_count
            iteration_count += 1

            if stop_event and stop_event.is_set():
                raise SolveCancelled("Mining cancelled")

            value = self._evaluate_circuit(circuit, params)

            if iteration_count % 10 == 0:
                logger.debug(
                    f"[QAOA] Optimizer iter {iteration_count}: energy={value:.4f}"
                )
            return value

        # Initial random angles: use circuit's actual parameter count
        # (accounts for per-solve p override)
        num_params = circuit.num_parameters
        x0 = np.random.uniform(0, 2 * np.pi, num_params)

        try:
            if self.optimizer_name == 'SPSA':
                spsa = SPSAOptimizer(
                    maxiter=optimizer_options.get('maxiter', 100),
                    a=optimizer_options.get('a', 0.1),
                    c=optimizer_options.get('c', 0.1),
                )
                result = spsa.minimize(objective, x0, stop_event)
            else:
                config = OPTIMIZER_CONFIGS[self.optimizer_name]
                result = minimize(
                    objective,
                    x0,
                    method=config['method'],
                    options=optimizer_options,
                )
        except SolveCancelled:
            logger.info("[QAOA] Optimization interrupted by stop_event")
            return None

        logger.info(
            f"[QAOA] Optimization done: {self.optimizer_name}, "
            f"iters={iteration_count}, best_energy={result.fun:.4f}"
        )
        return result.x

    def _final_sample(
        self, circuit: Any, best_params: np.ndarray
    ) -> Dict[str, int]:
        """Run the optimized circuit with more shots to collect diverse solutions.

        Uses self._solve_final_shots (set per-solve) for shot count.
        """
        bound = circuit.assign_parameters(best_params)
        bound.measure_all()

        final_shots = getattr(self, '_solve_final_shots', self.final_shots)
        job = self.backend.run(bound, shots=final_shots)
        result = job.result()
        return result.get_counts()

    def _counts_to_sampleset(
        self,
        counts: Dict[str, int],
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
    ) -> dimod.SampleSet:
        """Convert Qiskit measurement counts to a dimod.SampleSet.

        Handles:
          1. Bitstring (0/1) → Ising spin (-1/+1) conversion
          2. Energy computation for each unique solution
          3. Packing into dimod.SampleSet with correct variable labels
        """
        node_list = self.nodes
        samples = []
        energies = []
        num_occurrences = []

        for bitstring, count in counts.items():
            # Qiskit returns little-endian: qubit 0 is rightmost character
            spins = [1 - 2 * int(b) for b in reversed(bitstring)]

            energy = energy_of_solution(spins, h, J, node_list)

            samples.append(dict(zip(node_list, spins)))
            energies.append(energy)
            num_occurrences.append(count)

        return dimod.SampleSet.from_samples(
            samples,
            vartype=dimod.SPIN,
            energy=energies,
            num_occurrences=num_occurrences,
        )

    # -- Public interface ---------------------------------------------------

    def sample_ising(
        self,
        h,
        J,
        **kwargs,
    ) -> dimod.SampleSet:
        """Sampler protocol compatibility wrapper.

        BaseMiner expects a sampler with this method signature.
        Delegates to solve_ising, ignoring kwargs that don't apply to QAOA.
        """
        result = self.solve_ising(h, J)
        if result is None:
            # Return empty SampleSet if interrupted
            return dimod.SampleSet.from_samples([], vartype=dimod.SPIN, energy=[])
        return result

    def solve_ising(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        stop_event: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[dimod.SampleSet]:
        """Synchronous QAOA solve:  (h, J) → dimod.SampleSet.

        This is the primary interface.  Returns None if interrupted via
        stop_event.

        Args:
            h: Linear biases (field parameters).
            J: Quadratic biases (coupling parameters).
            stop_event: multiprocessing.Event for cooperative cancellation.
            params: Optional per-solve parameter overrides from adapt_parameters().
                   Keys: 'p', 'shots', 'final_shots', 'max_iter'.
                   If None, uses the values set during __init__.
        """
        solve_start = time.time()

        # Apply per-solve parameter overrides (from adapt_parameters)
        p = params.get('p', self.p) if params else self.p
        shots = params.get('shots', self.shots) if params else self.shots
        final_shots = params.get('final_shots', self.final_shots) if params else self.final_shots
        max_iter = params.get('max_iter', None) if params else None

        # Override optimizer max iterations if provided
        optimizer_options = self.optimizer_options.copy()
        if max_iter is not None:
            optimizer_options['maxiter'] = max_iter

        # Store h, J for use in _evaluate_circuit objective
        self._current_h = h
        self._current_J = J

        # Store per-solve values for use by pipeline methods
        self._solve_shots = shots
        self._solve_final_shots = final_shots

        # 1. Build cost Hamiltonian
        logger.info("[QAOA] Step 1/5: Building cost Hamiltonian (h,J → SparsePauliOp)")
        cost_op = self._build_cost_operator(h, J)

        # 2. Build & transpile QAOA circuit (p may differ per solve)
        logger.info(f"[QAOA] Step 2/5: Building QAOA circuit (p={p})")
        circuit = self._build_circuit(cost_op, p=p)
        logger.info(f"[QAOA] Step 3/5: Transpiling circuit for backend")
        circuit = self._transpile(circuit)

        # 4. Variational optimization
        logger.info(
            f"[QAOA] Step 4/5: Variational optimization "
            f"({self.optimizer_name}, max_iter={optimizer_options.get('maxiter', '?')}, "
            f"{shots} shots/eval)"
        )
        best_params = self._optimize(circuit, stop_event, optimizer_options=optimizer_options)
        if best_params is None:
            return None  # interrupted

        # 5. Final sampling with optimized angles
        logger.info(f"[QAOA] Step 5/5: Final sampling ({final_shots} shots)")
        counts = self._final_sample(circuit, best_params)

        # Convert to dimod.SampleSet
        sampleset = self._counts_to_sampleset(counts, h, J)

        solve_time = time.time() - solve_start
        logger.info(
            f"[QAOA] Solve complete in {solve_time:.2f}s — "
            f"{len(counts)} unique bitstrings from {final_shots} shots, "
            f"p={p}, max_iter={optimizer_options.get('maxiter', '?')}"
        )
        return sampleset

    def _get_solver_config(self) -> Dict[str, Any]:
        """Return a serializable config dict to reconstruct this solver.

        Used by solve_ising_async to create a fresh solver instance inside
        a child process.  The backend is passed as None so the child
        constructs its own AerSimulator (the backend object itself is not
        reliably picklable).
        """
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'backend': None,  # child process creates its own AerSimulator
            'p': self.p,
            'optimizer': self.optimizer_name,
            'shots': self.shots,
            'final_shots': self.final_shots,
            'optimizer_options': self.optimizer_options.copy(),
            'optimization_level': self.optimization_level,
        }

    def solve_ising_async(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        stop_event: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> QAOAFuture:
        """Asynchronous QAOA solve — returns a QAOAFuture immediately.

        The solve runs in a child process via multiprocessing.Process,
        giving it its own GIL for true parallelism.  The stop_event
        (a multiprocessing.Event) passes natively to the child.

        Args:
            h: Linear biases (field parameters).
            J: Quadratic biases (coupling parameters).
            stop_event: multiprocessing.Event for cooperative cancellation.
            params: Optional per-solve parameter overrides from adapt_parameters().
        """
        solve_start = time.time()

        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_qaoa_solve_in_process,
            args=(self._get_solver_config(), h, J, stop_event, params, result_queue),
        )
        process.start()

        return QAOAFuture(
            process=process,
            result_queue=result_queue,
            solve_start=solve_start,
        )