"""Features derived from locations."""
import numpy as np
from global_land_mask import globe


def land_mask(locs):
    """True where a (lat, lon) location is on land."""
    return globe.is_land(locs[:, 0], locs[:, 1])