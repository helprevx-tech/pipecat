"""Lossless PNG upload compatibility at the actual LemonSlice API boundary.

All images are tiny synthetic PIL values. Tests preserve caller ownership,
exact unsigned16-bit pixel bounds and genuine encoding failures; no transport is
used. Session tests replace only _post to assert pre-dispatch rejection or the
real encoded multipart bytes. Existing provider/client policy stays unchanged.
"""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image
from pipecat.transports.lemonslice.api import LemonSliceApi, _encode_image


@pytest.mark.parametrize("value", [0, 1, 255, 256, 65535])
def test_mode_I_uint16_edges_preserve_exact_decoded_pixel(value):
    """Inclusive scalar edges preserve decoded values without closing/modifying the caller's signed32-bit image."""
    with Image.new("I", (1, 1), value) as original:
        data = _encode_image(original)
        with Image.open(io.BytesIO(data)) as decoded:
            assert decoded.format == "PNG" and decoded.size == (1, 1)
            assert decoded.getpixel((0, 0)) == value
        assert original.mode == "I" and original.getpixel((0, 0)) == value


def test_mode_I_mixed_pixels_preserve_complete_order_and_range():
    """Whole-image extrema, not a single sample, admit all five representable pixel boundaries in order."""
    values = [0, 1, 255, 256, 65535]
    with Image.new("I", (5, 1)) as original:
        original.putdata(values)
        data = _encode_image(original)
        with Image.open(io.BytesIO(data)) as decoded:
            assert [decoded.getpixel((x, 0)) for x in range(5)] == values
        assert [original.getpixel((x, 0)) for x in range(5)] == values


@pytest.mark.parametrize("value", [-2147483648, -1, 65536, 2147483647])
def test_out_of_range_mode_I_refuses_instead_of_clipping(value):
    """Valid signed32-bit input outside PNG's range yields a clear ValueError and preserves the original pixel."""
    with Image.new("I", (1, 1), value) as original:
        with pytest.raises(ValueError, match=r"0\.\.65535.*lossless"):
            _encode_image(original)
        assert original.getpixel((0, 0)) == value


@pytest.mark.parametrize("values", [[1, -1], [1, 65536]])
def test_later_out_of_range_pixels_cannot_bypass_full_extrema(values):
    """A valid first pixel cannot conceal a later negative/overflow pixel or authorize partial output."""
    with Image.new("I", (2, 1)) as original:
        original.putdata(values)
        with pytest.raises(ValueError, match="lossless"):
            _encode_image(original)
        assert [original.getpixel((x, 0)) for x in range(2)] == values


@pytest.mark.parametrize("size", [(0, 0), (0, 1), (1, 0)])
def test_empty_mode_I_refuses_without_closing_input(size):
    """An empty PIL image has no extrema; refuse explicitly before conversion/upload and leave caller cleanup intact."""
    with Image.new("I", size) as original:
        with pytest.raises(ValueError, match="empty mode I"):
            _encode_image(original)
        assert original.size == size and original.getextrema() is None


@pytest.mark.parametrize("value", [None, b"", "not an image", object()])
def test_invalid_image_type_has_clear_contract_error(value):
    """Missing/byte/string/arbitrary objects do not become implicit image decodes or transport requests."""
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        _encode_image(value)


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "RGB", "RGBA", "I;16"])
def test_other_supported_modes_keep_native_PNG_bytes_and_caller_ownership(mode):
    """Unchanged modes retain the original direct-Pillow PNG path, exact pixels and still-open caller image."""
    with Image.new(mode, (2, 2), 1) as original:
        with io.BytesIO() as buffer:
            original.save(buffer, format="PNG")
            expected = buffer.getvalue()
        actual = _encode_image(original)
        assert actual == expected
        # Native PNG decoding canonicalizes mode-1 true pixels to255 even when
        # Image.new stored1. The unchanged direct-Pillow PNG is the strict oracle.
        with Image.open(io.BytesIO(actual)) as decoded, Image.open(io.BytesIO(expected)) as native:
            assert (
                decoded.format == "PNG"
                and decoded.size == original.size
                and decoded.mode == native.mode
            )
            assert [decoded.getpixel((x, y)) for y in range(2) for x in range(2)] == [
                native.getpixel((x, y)) for y in range(2) for x in range(2)
            ]
        assert original.getpixel((0, 0)) is not None


def test_unsupported_float_PNG_error_is_not_hidden_or_converted():
    """The existing native error for a non-PNG float mode propagates rather than gaining an invented lossy fallback."""
    with Image.new("F", (1, 1), 1.0) as original:
        with pytest.raises(OSError, match="mode F"):
            _encode_image(original)
        assert original.getpixel((0, 0)) == 1.0


@pytest.mark.parametrize("fail_save", [False, True])
def test_temporary_conversion_closes_and_original_survives_encoding_outcome(monkeypatch, fail_save):
    """Own only the temporary image on success/genuine failure; preserve the exact encoder exception and caller ownership."""
    converted = []
    native_convert = Image.Image.convert
    native_save = Image.Image.save
    failure = OSError("synthetic PNG encoder failure")

    def capture(image, mode=None, *args, **kwargs):
        """Track the real conversion so its resource can be checked after the original save path settles."""
        value = native_convert(image, mode, *args, **kwargs)
        if mode == "I;16":
            converted.append(value)
        return value

    def save(image, *args, **kwargs):
        """Inject only the selected converted-image encoder failure; all other writes stay normal local BytesIO operations."""
        if fail_save and any(image is item for item in converted):
            raise failure
        return native_save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", capture)
    monkeypatch.setattr(Image.Image, "save", save)
    with Image.new("I", (1, 1), 256) as original:
        if fail_save:
            with pytest.raises(OSError) as observed:
                _encode_image(original)
            assert observed.value is failure
        else:
            with Image.open(io.BytesIO(_encode_image(original))) as decoded:
                assert decoded.getpixel((0, 0)) == 256
        assert len(converted) == 1
        with pytest.raises(ValueError):
            converted[0].getpixel((0, 0))
        assert original.mode == "I" and original.getpixel((0, 0)) == 256


@pytest.mark.asyncio
async def test_create_session_sends_exact_lossless_PNG_through_existing_post_owner():
    """A valid mode-I avatar reaches the unchanged session owner as real PNG bytes; the transport itself remains mocked."""
    api = LemonSliceApi("synthetic-key", MagicMock())
    api._post = AsyncMock(return_value={"session_id": "synthetic"})
    with Image.new("I", (1, 1), 65535) as original:
        assert await api.create_session(agent_image=original) == {"session_id": "synthetic"}
        api._post.assert_awaited_once()
        with Image.open(io.BytesIO(api._post.await_args.kwargs["image_bytes"])) as decoded:
            assert decoded.getpixel((0, 0)) == 65535
        assert original.getpixel((0, 0)) == 65535


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [-1, 65536])
async def test_create_session_rejects_lossy_image_before_post(value):
    """A rejected avatar cannot issue a partial session request; the exact local input remains caller-owned."""
    api = LemonSliceApi("synthetic-key", MagicMock())
    api._post = AsyncMock()
    with Image.new("I", (1, 1), value) as original:
        with pytest.raises(ValueError, match="lossless"):
            await api.create_session(agent_image=original)
        api._post.assert_not_awaited()
        assert original.getpixel((0, 0)) == value
