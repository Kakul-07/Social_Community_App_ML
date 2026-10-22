import re
import unicodedata


def clean_text(value):
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", value).strip()


def test_whitespace_is_collapsed():
    assert clean_text("  hello\n  world ") == "hello world"


def test_empty_input():
    assert clean_text("   ") == ""


def test_unicode_normalization():
    assert clean_text("ﬁne") == "fine"


def test_zero_width_and_bom_are_removed():
    assert clean_text("\ufeffhello\u200b world") == "hello world"
