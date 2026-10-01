from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]

REVISION = "b4be1225c9b593ffa79f2bf46d8a84a83d385c67"

TOKENIZER_PATH = (
    ROOT
    / ".cache"
    / "upstream"
    / "morena-1.5b-instruct"
    / REVISION
    / "tokenizer.json"
)


EXPECTED_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<reserved_0>": 3,
    "<reserved_1>": 4,
    "<reserved_2>": 5,
    "<reserved_3>": 6,
    "<reserved_4>": 7,
    "<reserved_5>": 8,
    "<reserved_6>": 9,
    "<reserved_7>": 10,
}


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def test_tokenizer_file_exists() -> None:
    assert TOKENIZER_PATH.is_file()


def test_vocabulary_size() -> None:
    tokenizer = load_tokenizer()

    assert tokenizer.get_vocab_size(with_added_tokens=True) == 65536


def test_special_token_ids_are_frozen() -> None:
    tokenizer = load_tokenizer()

    observed = {
        token: tokenizer.token_to_id(token)
        for token in EXPECTED_SPECIAL_TOKENS
    }

    assert observed == EXPECTED_SPECIAL_TOKENS


def test_reserved_user_marker_is_single_token() -> None:
    tokenizer = load_tokenizer()

    encoded = tokenizer.encode(
        "<reserved_0>",
        add_special_tokens=False,
    )

    assert encoded.ids == [3]


def test_reserved_assistant_marker_is_single_token() -> None:
    tokenizer = load_tokenizer()

    encoded = tokenizer.encode(
        "<reserved_1>",
        add_special_tokens=False,
    )

    assert encoded.ids == [4]


def test_nfc_normalization_is_stable() -> None:
    tokenizer = load_tokenizer()

    composed = "môre"
    decomposed = "mo\u0302re"

    composed_ids = tokenizer.encode(
        composed,
        add_special_tokens=False,
    ).ids

    decomposed_ids = tokenizer.encode(
        decomposed,
        add_special_tokens=False,
    ).ids

    assert composed_ids == decomposed_ids


def test_multilingual_text_round_trip() -> None:
    tokenizer = load_tokenizer()

    text = "Dumela! Ngikhona. Goeie môre."

    encoded = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    decoded = tokenizer.decode(encoded.ids)

    assert decoded == text
