"""The v2 family judges (metric ``canonical_spatial_task_h1_v2``) live in the RL env.

``vagen/envs/active_spatial/canonical_task_metrics_v2.py`` is the single source: the env scores
rows with it and the generator imports it under this name, so ``from . import judges_v2 as J``
gives the env module itself (private helpers included).
"""

import sys

from vagen.envs.active_spatial import canonical_task_metrics_v2 as _impl

sys.modules[__name__] = _impl
