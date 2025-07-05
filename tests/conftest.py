from timecopilot.agent import MODELS
from timecopilot.models.foundational.chronos import Chronos

benchmark_models = [
    "AutoARIMA",
    "SeasonalNaive",
    "ZeroModel",
    "ADIDA",
    "TimesFM",
]
models = [MODELS[str_model] for str_model in benchmark_models]
models.extend(
    [
        Chronos(repo_id="amazon/chronos-t5-tiny", alias="Chronos-T5"),
        Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt"),
    ]
)
