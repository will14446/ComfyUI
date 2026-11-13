"""Which folders a pack that runs in its own process can make ComfyUI list as model folders."""

import asyncio
import copy

import pytest

import folder_paths
from comfy_api.v4 import _pack_host, _pack_processes

from conftest import evaluate, needs_pack_process, start_probe

pytestmark = needs_pack_process


def process_of(pack) -> _pack_host.PackProcess:
    return _pack_host.PackProcess(str(pack), "pack")


@pytest.fixture
def model_folders(monkeypatch, tmp_path):
    """ComfyUI's table of model folders, restored after the test; a pack folder and a folder
    outside it that only ComfyUI may read."""
    monkeypatch.setattr(
        folder_paths,
        "folder_names_and_paths",
        copy.deepcopy(folder_paths.folder_names_and_paths),
    )
    folder_paths.filename_list_cache.clear()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "decoy.safetensors").write_bytes(b"")
    pack = tmp_path / "custom_nodes" / "pack"
    (pack / "models").mkdir(parents=True)
    yield pack, outside
    folder_paths.filename_list_cache.clear()


@pytest.mark.parametrize("link", ["models/out", "other/out"])
def test_a_folder_of_a_pack_with_a_link_out_of_it_cannot_be_a_model_folder(
    model_folders, link
):
    pack, outside = model_folders
    (pack / link).parent.mkdir(exist_ok=True)
    (pack / link).symlink_to(outside)
    if link.startswith("other"):
        (pack / "models" / "other").symlink_to(pack / "other")

    with pytest.raises(ValueError, match="cannot be a model folder"):
        _pack_host._folders(
            process_of(pack),
            "add_model_folder_path",
            ("v4_linked", str(pack / "models")),
            {},
        )

    assert "v4_linked" not in folder_paths.folder_names_and_paths


def test_a_folder_of_a_pack_whose_links_stay_in_it_is_a_model_folder(model_folders):
    pack, _ = model_folders
    (pack / "weights").mkdir()
    (pack / "weights" / "small.safetensors").write_bytes(b"")
    (pack / "models" / "weights").symlink_to(pack / "weights")

    _pack_host._folders(
        process_of(pack),
        "add_model_folder_path",
        ("v4_linked", str(pack / "models")),
        {},
    )

    assert folder_paths.get_filename_list("v4_linked") == ["weights/small.safetensors"]


@pytest.mark.parametrize(
    "entry",
    [
        lambda models: ("", {".bin"}),
        lambda models: ([models], 5),
        lambda models: ([models, 3], {".bin"}),
        lambda models: ([models], {".bin"}, "more"),
    ],
    ids=["text", "number", "not a path", "three parts"],
)
def test_a_model_folder_entry_that_is_not_paths_and_extensions_is_refused(
    model_folders, entry
):
    pack, _ = model_folders
    before = copy.deepcopy(folder_paths.folder_names_and_paths["vae"])

    with pytest.raises((TypeError, ValueError)):
        _pack_host._assign_folder(process_of(pack), "vae", entry(str(pack / "models")))

    assert folder_paths.folder_names_and_paths["vae"] == before


def test_a_model_folder_entry_named_by_anything_but_a_string_is_refused(model_folders):
    pack, _ = model_folders
    before = dict(folder_paths.folder_names_and_paths)

    with pytest.raises(TypeError, match="named by a string"):
        _pack_host._assign_folder(
            process_of(pack), ("vae",), ([str(pack / "models")], {".bin"})
        )

    assert folder_paths.folder_names_and_paths == before


def test_what_a_pack_process_changed_in_the_model_folders_is_undone_when_it_is_taken_away(
    model_folders, tmp_path, monkeypatch
):
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "models"))
    table = folder_paths.folder_names_and_paths
    vae, loras = table["vae"], list(table["loras"][0])
    added = str(tmp_path / "models" / "more")
    process = start_probe(tmp_path)
    try:
        asyncio.run(
            evaluate(
                process,
                f"folders.add_model_folder_path('v4_new', {added!r}),"
                f"folders.add_model_folder_path('loras', {added!r}),"
                f"folders.folder_names_and_paths.__setitem__('vae', ([{added!r}], {{'.x'}})),"
                "folders.folder_names_and_paths.__setitem__("
                f"'upscale_models', ([{added!r}], {{'.x'}}))",
            )
        )
        during = (list(table["v4_new"][0]), added in table["loras"][0], table["vae"][0])
        changed_since = table["upscale_models"] = ([], set())
    finally:
        _pack_processes.unload(process.module_path)

    assert during == ([added], True, [added])
    assert "v4_new" not in table
    assert table["loras"][0] == loras and table["vae"] is vae
    assert table["upscale_models"] is changed_since
