from rsl_rl.storage.rollout_storage import RolloutStorage

class LongContextRolloutStorage(RolloutStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_hidden_states(self, hidden_states: torch.Tensor) -> None:
        