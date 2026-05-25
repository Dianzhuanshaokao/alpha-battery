import json
from datetime import datetime
from pathlib import Path


class BayesianOptimization:
    FAILURE_PARAM_KEYS = [
        "Inner SEI lithium interstitial diffusivity [m2.s-1]",
        "Dead lithium decay constant [s-1]",
        "Lithium plating kinetic rate constant [m.s-1]",
        "Negative electrode LAM constant proportional term [s-1]",
        "Positive electrode LAM constant proportional term [s-1]",
        "Negative electrode cracking rate",
        "Outer SEI partial molar volume [m3.mol-1]",
        "SEI growth activation energy [J.mol-1]",
        "Negative cracking growth activation energy [J.mol-1]",
        "Negative electrode diffusivity activation energy [J.mol-1]",
        "Positive electrode diffusivity activation energy [J.mol-1]",
    ]

    def __init__(
        self,
        fun,
        bounds,
        n_init_points=5,
        acq="ucb",
        kappa=2.576,
        xi=0.01,
        noise=1e-8,
        invalid_registry_path=None,
        invalid_param_keys=None,
    ):
        self.fun = fun
        self.bounds = bounds
        self.n_init_points = n_init_points
        self.acq = acq
        self.kappa = kappa
        self.xi = xi
        self.noise = noise
        default_registry = Path(__file__).resolve().parent / "invalid_params.jsonl"
        self.invalid_registry_path = Path(invalid_registry_path) if invalid_registry_path else default_registry
        self.invalid_param_keys = invalid_param_keys or self.FAILURE_PARAM_KEYS
        self._invalid_signature_cache = self._load_invalid_signatures()

    def _normalize_value(self, value):
        if isinstance(value, float):
            return f"{value:.16e}"
        return value

    def _canonicalize_params(self, params):
        if not isinstance(params, dict):
            return ""
        if self.invalid_param_keys:
            filtered = {k: params[k] for k in self.invalid_param_keys if k in params}
        else:
            filtered = dict(params)
        normalized = {k: self._normalize_value(v) for k, v in filtered.items()}
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False)

    def _load_invalid_signatures(self):
        signatures = set()
        if not self.invalid_registry_path.exists():
            return signatures
        with self.invalid_registry_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    signature = item.get("signature")
                    if signature:
                        signatures.add(signature)
        return signatures

    def refresh_invalid_cache(self):
        self._invalid_signature_cache = self._load_invalid_signatures()
        return self._invalid_signature_cache

    def is_invalid_params(self, params):
        signature = self._canonicalize_params(params)
        return bool(signature) and signature in self._invalid_signature_cache

    def register_invalid_params(self, params, reason="simulation_failed"):
        signature = self._canonicalize_params(params)
        if not signature or signature in self._invalid_signature_cache:
            return False
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
            "signature": signature,
            "params": {k: params[k] for k in self.invalid_param_keys if k in params},
        }
        self.invalid_registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.invalid_registry_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._invalid_signature_cache.add(signature)
        return True

    def evaluate(self, params):
        if self.is_invalid_params(params):
            return float("inf")
        try:
            value = self.fun(params)
        except Exception:
            self.register_invalid_params(params, reason="simulator_exception")
            return float("inf")
        if value is None:
            self.register_invalid_params(params, reason="simulator_return_none")
            return float("inf")
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            self.register_invalid_params(params, reason="simulator_return_invalid_number")
            return float("inf")
        return value
