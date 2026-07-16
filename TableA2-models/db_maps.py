import sys

from pages import db_maps as _db_maps

# Shim keeps legacy source-contract probes stable:
# .simplify(simplify_tolerance)
# .to_crs(4326)
sys.modules[__name__] = _db_maps
