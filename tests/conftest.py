# SPDX-License-Identifier: AGPL-3.0-only
"""pytest fixtures for i.delineate.opencl.

The delineation tests need a Delineate-Anything weights file and an
OpenCL device, neither of which can be assumed on a test machine. They
are skipped unless the path of a weights file (DelineateAnything-S.pt is
the fastest) is given in I_DELINEATE_OPENCL_CHECKPOINT.
I_DELINEATE_OPENCL_PLATFORM optionally selects the OpenCL platform (for
Mesa rusticl, also export RUSTICL_ENABLE=radeonsi or the matching
driver).
"""

import os
from pathlib import Path

import grass.script as gs
import pytest
from grass.tools import Tools

# Five flat-coloured fields of a 256 x 256 cell, 10 m image: (name, north
# row, south row, west col, east col, (r, g, b)) with r.mapcalc 1-based
# rows; the last one fills the rest of the image.
SIZE = 256
RES = 10
RECTANGLES = [
    ("red", 1, 60, 1, 100, (200, 60, 60)),
    ("green", 1, 60, 101, SIZE, (60, 200, 60)),
    ("blue", 61, SIZE, 1, 120, (60, 60, 200)),
    ("yellow", 61, 180, 121, SIZE, (200, 200, 60)),
    ("cyan", 181, SIZE, 121, SIZE, (60, 200, 200)),
]


def _band_expression(name, band):
    expr = "0"
    for _, n, s, w, e, rgb in reversed(RECTANGLES):
        expr = (
            f"if(row() >= {n} && row() <= {s} && col() >= {w} && col() <= {e}, "
            f"{rgb[band]}, {expr})"
        )
    return f"{name} = int({expr} + noise)"


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Session with a 3-band group <rgb> of the fields image and a 2-band
    group <two>, in a metric projected project."""
    project = tmp_path_factory.mktemp("i_delineate_opencl") / "project"
    gs.create_project(project, epsg="32636")
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(
            n=5100000 + SIZE * RES,
            s=5100000,
            e=220000 + SIZE * RES,
            w=220000,
            res=RES,
        )
        tools.r_mapcalc(expression="noise = rand(0, 12)", seed=1)
        for band, name in enumerate(("band_r", "band_g", "band_b")):
            tools.r_mapcalc(expression=_band_expression(name, band))
        tools.r_mapcalc(expression="landcover = if(col() <= 100, 1, 2)")
        tools.i_group(group="rgb", input="band_r,band_g,band_b")
        tools.i_group(group="two", input="band_r,band_g")
        yield session


@pytest.fixture(scope="module")
def checkpoint():
    path = os.environ.get("I_DELINEATE_OPENCL_CHECKPOINT")
    if not path or not Path(path).is_file():
        pytest.skip(
            "set I_DELINEATE_OPENCL_CHECKPOINT to a Delineate-Anything .pt file"
        )
    return path


@pytest.fixture(scope="module")
def device_options():
    platform = os.environ.get("I_DELINEATE_OPENCL_PLATFORM")
    return {"platform": platform} if platform else {}


def cell_center(row, col):
    """Map coordinates of the centre of the cell at 1-based (row, col)."""
    return (220000 + (col - 0.5) * RES, 5100000 + SIZE * RES - (row - 0.5) * RES)
