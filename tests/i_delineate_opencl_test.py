# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for i.delineate.opencl."""

import functools
import http.server
import threading
import zipfile
from pathlib import Path

import pytest
from conftest import RECTANGLES, RES, SIZE, cell_center
from grass.tools import ToolError, Tools


def _tools_with_env(session, **env):
    return Tools(env={**session.env, **env})


def test_group_must_hold_three_bands(session, tmp_path):
    """A group with other than three raster maps is rejected up front."""
    tools = Tools(session=session)
    dummy = tmp_path / "dummy.pt"
    dummy.write_bytes(b"not used")

    with pytest.raises(ToolError, match="exactly 3"):
        tools.i_delineate_opencl(group="two", checkpoint=str(dummy), output="out")


def test_rejects_file_that_is_not_a_checkpoint(session, tmp_path):
    """A file that is not a PyTorch zip archive fails before any OpenCL
    work."""
    tools = Tools(session=session)
    bogus = tmp_path / "bogus.pt"
    bogus.write_bytes(b"\0" * 4096)

    with pytest.raises(ToolError, match="not a zip archive"):
        tools.i_delineate_opencl(group="rgb", checkpoint=str(bogus), output="out")


def test_rejects_zip_without_pickle(session, tmp_path):
    """A zip archive without a pickle is not a checkpoint."""
    tools = Tools(session=session)
    archive = tmp_path / "other.pt"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("other/readme.txt", "hello")

    with pytest.raises(ToolError, match="not a PyTorch checkpoint"):
        tools.i_delineate_opencl(group="rgb", checkpoint=str(archive), output="out")


def test_rejects_state_dict(session, tmp_path):
    """A bare state_dict has no network description: Delineate-Anything
    files hold a whole pickled model."""
    torch = pytest.importorskip("torch")
    path = tmp_path / "state_dict.pt"
    torch.save({"model.0.conv.weight": torch.zeros(16, 3, 3, 3)}, path)
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="does not hold a whole"):
        tools.i_delineate_opencl(group="rgb", checkpoint=str(path), output="out")


def test_mask_needs_classes(session, tmp_path):
    """A land cover mask without any class list is an error."""
    tools = Tools(session=session)
    with pytest.raises(ToolError, match="mask"):
        tools.i_delineate_opencl(
            group="rgb",
            checkpoint=str(tmp_path / "missing.pt"),
            mask="landcover",
            output="out",
        )


def test_classes_need_mask(session, tmp_path):
    """Class lists without a land cover mask are an error."""
    tools = Tools(session=session)
    with pytest.raises(ToolError, match="mask"):
        tools.i_delineate_opencl(
            group="rgb",
            checkpoint=str(tmp_path / "missing.pt"),
            clip_classes=1,
            output="out",
        )


def test_default_weights_dir(session, tmp_path):
    """Without checkpoint or weights_dir, $DELINEATE_MODELS_DIR is
    searched."""
    tools = _tools_with_env(session, DELINEATE_MODELS_DIR=str(tmp_path))

    with pytest.raises(ToolError, match=r"No\s+large_v2\s+weights") as e:
        tools.i_delineate_opencl(group="rgb", output="out", flags="n")
    assert str(tmp_path) in " ".join(str(e.value).split())


def test_weights_dir_without_requested_model(session, tmp_path):
    """Asking for a model that is not in the folder is an error, not a
    silent fallback to another model."""
    tools = Tools(session=session)
    (tmp_path / "DelineateAnything.pt").write_bytes(b"not used")

    with pytest.raises(ToolError, match=r"No\s+small\s+weights"):
        tools.i_delineate_opencl(
            group="rgb",
            weights_dir=str(tmp_path),
            model="small",
            output="out",
            flags="n",
        )


@pytest.fixture
def mirror(tmp_path):
    """A local HTTP server standing in for the download site; yields the
    folder it serves and its URL."""
    root = tmp_path / "mirror"
    root.mkdir()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_download_rejects_wrong_size(session, tmp_path, mirror):
    """A download that does not have the published size is discarded, not
    installed in the weights folder."""
    root, url = mirror
    (root / "DelineateAnything-S.pt").write_bytes(b"truncated")
    weights = tmp_path / "weights"
    tools = _tools_with_env(session, I_DELINEATE_OPENCL_DOWNLOAD_URL=url)

    with pytest.raises(ToolError, match=r"instead\s+of\s+17635629"):
        tools.i_delineate_opencl(
            group="rgb", weights_dir=str(weights), model="small", output="out"
        )
    assert not list(weights.glob("*.pt*"))


def test_download_fetches_missing_model(session, checkpoint, tmp_path, mirror):
    """A model missing from the weights folder (created if needed) is
    downloaded there and used."""
    if Path(checkpoint).stat().st_size != 17635629:
        pytest.skip("checkpoint is not DelineateAnything-S.pt")
    root, url = mirror
    (root / "DelineateAnything-S.pt").symlink_to(checkpoint)
    weights = tmp_path / "new" / "weights"
    tools = _tools_with_env(session, I_DELINEATE_OPENCL_DOWNLOAD_URL=url)

    # A platform that does not exist stops the run right after the
    # weights have been fetched and read.
    with pytest.raises(ToolError, match=r"OpenCL"):
        tools.i_delineate_opencl(
            group="rgb",
            weights_dir=str(weights),
            model="small",
            output="out",
            platform="no_such_platform",
        )
    assert (weights / "DelineateAnything-S.pt").stat().st_size == 17635629
    assert not list(weights.glob("*.part"))


def _category_at(tools, name, row, col):
    """Category of the area at a cell centre, or None (both the older flat
    and the newer nested v.what JSON layouts)."""
    x, y = cell_center(row, col)
    what = tools.v_what(map=name, coordinates=(x, y), format="json").json
    if not what:
        return None
    if "Category" in what[0]:
        return what[0]["Category"]
    data = what[0].get("data") or []
    return data[0]["category"] if data else None


def _fields_at_rectangles(tools, name):
    """Category and area of the field at the centre of each rectangle."""
    found = {}
    for rect, n, s, w, e, _ in RECTANGLES:
        cat = _category_at(tools, name, (n + s) // 2, (w + e) // 2)
        assert cat is not None, f"{rect} rectangle not delineated"
        row = tools.v_db_select(map=name, where=f"cat = {cat}", format="json")
        found[rect] = (cat, row.json["records"][0]["area"])
    return found


@pytest.fixture(scope="module")
def delineated(session, checkpoint, device_options):
    """Delineate the rectangles once for the result checks, in single
    precision so that the result does not depend on the device found."""
    tools = Tools(session=session)
    tools.i_delineate_opencl(
        group="rgb",
        checkpoint=checkpoint,
        output="fields",
        raster="fields",
        precision="fp32",
        **device_options,
    )
    return tools


def test_each_rectangle_is_one_field(delineated):
    """Every rectangle is recovered as its own field of the right area."""
    found = _fields_at_rectangles(delineated, "fields")
    assert len({cat for cat, _ in found.values()}) == len(RECTANGLES)
    for rect, n, s, w, e, _ in RECTANGLES:
        area = (s - n + 1) * (e - w + 1) * RES * RES
        assert found[rect][1] == pytest.approx(area, rel=0.03), rect


def test_fields_are_areas_with_attributes(delineated):
    """One centroid per field, no background without a land cover mask."""
    records = delineated.v_db_select(map="fields", format="json").json["records"]
    topo = delineated.v_info(map="fields", flags="t", format="json").json
    assert topo["centroids"] == len(records) == len(RECTANGLES)
    assert all(r["background"] == 0 and r["class"] == 0 for r in records)


def test_raster_is_at_half_resolution(delineated):
    """10 m imagery is delineated at 5 m: the field raster has twice the
    rows and columns of the region."""
    info = delineated.r_info(map="fields", format="json").json
    assert info["nsres"] == pytest.approx(RES / 2)
    assert info["rows"] == 2 * 256


def test_clip_classes_cut_fields(session, checkpoint, device_options):
    """Cells of a clip class are cut out of the fields: the red field,
    entirely in class 1, disappears."""
    tools = Tools(session=session)
    tools.i_delineate_opencl(
        group="rgb",
        checkpoint=checkpoint,
        output="clipped",
        mask="landcover",
        clip_classes=1,
        precision="fp32",
        **device_options,
    )
    assert _category_at(tools, "clipped", 30, 50) is None


@pytest.mark.parametrize(
    ("profile", "precision"), [("cpu", "fp32"), ("dgpu", "fp32"), ("igpu", "fp16")]
)
def test_profiles_give_the_same_fields(
    session, checkpoint, device_options, profile, precision
):
    """A forced device profile recovers the same fields (profiles other
    than cpu run the GPU kernels, slow on a CPU device but valid). Half
    precision storage moves scores by about 0.005, which can tip a
    detection lying at the confidence threshold: there, every rectangle
    must still be covered by fields."""
    tools = Tools(session=session)
    out = f"fields_{profile}_{precision}"
    try:
        tools.i_delineate_opencl(
            group="rgb",
            checkpoint=checkpoint,
            output=out,
            profile=profile,
            precision=precision,
            **device_options,
        )
    except ToolError as e:
        if "needs a device sharing memory" in str(e):
            pytest.skip("igpu needs a device sharing memory with the host")
        raise
    found = _fields_at_rectangles(tools, out)
    if precision == "fp16":
        records = tools.v_db_select(map=out, format="json").json["records"]
        total = sum(r["area"] for r in records)
        assert total == pytest.approx(SIZE * SIZE * RES * RES, rel=0.03)
        return
    for rect, n, s, w, e, _ in RECTANGLES:
        area = (s - n + 1) * (e - w + 1) * RES * RES
        assert found[rect][1] == pytest.approx(area, rel=0.03), rect
