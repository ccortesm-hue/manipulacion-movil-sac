"""SAC único: regularización docente balanceada y checkpoints de inferencia ligeros."""
import numpy as np
from demo_regularized_sac_v9 import DemoRegularizedSAC


class VXSAC(DemoRegularizedSAC):
    def _excluded_save_params(self):
        # The dataset lives in NPZ, not inside the deployed actor/critics file.
        return super()._excluded_save_params() + [k for k in self.__dict__ if k.startswith('_demo_')]

    def _sample_demo_indices(self):
        groups = getattr(self, '_demo_group_indices', None)
        if not groups:
            return super()._sample_demo_indices()
        phases = self._demo_rng.integers(0, 3, self._demo_batch_size)
        result = np.empty(self._demo_batch_size, dtype=np.int64)
        for phase in (0, 1, 2):
            names = [n for n in groups if n.startswith(f'p{phase}:')]
            if not names:
                return super()._sample_demo_indices()
            positions = np.flatnonzero(phases == phase)
            choices = self._demo_rng.integers(len(names), size=positions.size)
            for group_id, name in enumerate(names):
                selected = positions[choices == group_id]
                result[selected] = self._demo_rng.choice(groups[name], size=selected.size)
        return result
