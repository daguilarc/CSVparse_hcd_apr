import sys

from pages import map_metric_registry as _map_metric_registry

sys.modules[__name__] = _map_metric_registry
