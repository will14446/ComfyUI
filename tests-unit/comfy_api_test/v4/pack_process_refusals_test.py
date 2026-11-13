import fractions
import io as stdio
import pickle
import pickletools

import pytest

from comfy_api.latest import _io
from comfy_api.v4 import _codec, _pack_host

from conftest import needs_pack_process

pytestmark = needs_pack_process


class _Side:
    peer = "pack test"
    io_types = {}

    def __init__(self):
        self.objects = {}

    def remote(self, key, type_name, methods, values):
        return _codec.Remote(self, key, type_name, methods, values)

    def resolve(self, key, chain):
        return self.objects[key]


class _Sent:
    """Stands for one of ComfyUI's objects that the pack sends back."""


SENT = _Sent()


def _ops(value):
    buffer = stdio.BytesIO()
    pickler = pickle.Pickler(buffer, protocol=2)
    pickler.fast = True
    pickler.persistent_id = lambda obj: ("yours", "held") if obj is SENT else None
    pickler.dump(value)
    return buffer.getvalue()[2:-1]


class _Held:
    pass


@pytest.mark.parametrize(
    "target, opcodes",
    [
        (
            "a class",
            b"ccomfy_api.latest._io\nComfyNode\n"
            + _ops((None, {"changed": 1}))
            + b"b.",
        ),
        ("a class", b"cfractions\nFraction\n" + _ops((None, {"changed": 1})) + b"b."),
        (
            "an enum member",
            b"ccomfy_api.latest._io\nFolderType\n"
            + _ops(("output",))
            + b"R"
            + _ops(({"_value_": "x"}, None))
            + b"b.",
        ),
        (
            "an object it did not make",
            _ops(SENT) + _ops((None, {"changed": 1})) + b"b.",
        ),
        ("an object it did not make", _ops(SENT) + _ops("key") + _ops("value") + b"s."),
    ],
)
def test_a_message_cannot_change_a_class_an_enum_member_or_an_object_it_did_not_make(
    target, opcodes
):
    side = _Side()
    held = side.objects["held"] = _Held()

    with pytest.raises(pickle.UnpicklingError, match="refused to change"):
        _codec.loads(b"\x80\x05" + opcodes, side, 1)

    assert "changed" not in vars(_io.ComfyNode) and "changed" not in vars(
        fractions.Fraction
    )
    assert _io.FolderType.output.value == "output"
    assert vars(held) == {}


CHECKED = "ADDITEMS APPEND APPENDS BUILD NEWOBJ REDUCE SETITEM SETITEMS"
REFUSED = "EXT1 EXT2 EXT4 INST NEWOBJ_EX OBJ"
AS_PYTHON = """
    BINBYTES BINBYTES8 BINFLOAT BINGET BININT BININT1 BININT2 BINPERSID BINPUT BINSTRING
    BINUNICODE BINUNICODE8 BYTEARRAY8 DICT DUP EMPTY_DICT EMPTY_LIST EMPTY_SET EMPTY_TUPLE FLOAT
    FRAME FROZENSET GET GLOBAL INT LIST LONG LONG1 LONG4 LONG_BINGET LONG_BINPUT MARK MEMOIZE
    NEWFALSE NEWTRUE NEXT_BUFFER NONE PERSID POP POP_MARK PROTO PUT READONLY_BUFFER
    SHORT_BINBYTES SHORT_BINSTRING SHORT_BINUNICODE STACK_GLOBAL STOP STRING TUPLE TUPLE1 TUPLE2
    TUPLE3 UNICODE
"""


def test_the_decoder_has_decided_every_instruction_of_pythons_unpickler():
    """The decoder is built on `pickle._Unpickler`, which Python may change: an instruction it
    gains, or one the decoder no longer checks, fails here before a message can use it."""
    python, ours = pickle._Unpickler.dispatch, _codec._Unpickler.dispatch
    names = {ord(opcode.code): opcode.name for opcode in pickletools.opcodes}

    def decided(code):
        if ours[code] is _codec._refuse:
            return "refused"
        return "as Python" if ours[code] is python[code] else "checked"

    assert {names[code]: decided(code) for code in python} == {
        **dict.fromkeys(CHECKED.split(), "checked"),
        **dict.fromkeys(REFUSED.split(), "refused"),
        **dict.fromkeys(AS_PYTHON.split(), "as Python"),
    }


@pytest.mark.parametrize("name", REFUSED.split())
def test_a_message_with_an_instruction_no_value_needs_is_refused(name):
    code = next(o.code for o in pickletools.opcodes if o.name == name).encode("latin-1")

    with pytest.raises(pickle.UnpicklingError, match="a value does not need"):
        _codec.loads(b"\x80\x05" + _ops(()) + code + b".", _Side(), 1)


def test_comfyui_keeps_only_what_it_uses_of_what_a_pack_offers():
    process = _pack_host.PackProcess("/nowhere/pack", "pack")
    process.channel = type("Channel", (), {"release": staticmethod(lambda key: None)})()

    remote = process.remote(
        (0, 1),
        "Video",
        ("get_stream_source", "generate_noise"),
        {"seed": 3, "format": "gltf"},
    )

    assert remote.seed == 3 and callable(remote.generate_noise)
    for name in ("get_stream_source", "format"):
        with pytest.raises(AttributeError):
            getattr(remote, name)


@pytest.mark.parametrize(
    "web, refused",
    [
        (None, False),
        ("web", False),
        ("web/../js", False),
        ("..", True),
        ("/etc", True),
        ("link", True),
        (3, True),
    ],
)
def test_a_web_folder_of_a_pack_process_is_served_only_from_inside_the_pack(
    tmp_path, web, refused
):
    pack = tmp_path / "pack"
    (pack / "web").mkdir(parents=True)
    (pack / "link").symlink_to(tmp_path)

    assert _pack_host._web_directory(str(pack), web) == (None if refused else web)
