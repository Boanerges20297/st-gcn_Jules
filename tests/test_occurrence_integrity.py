from decimal import Decimal

import pandas as pd
import pytest

from src.core.occurrence_integrity import (
    InvalidOccurrenceId,
    OccurrenceIdentityConflict,
    collect_mysql_ids,
    deduplicate_mysql_rows,
    normalize_mysql_id,
    sanitize_death_count,
    split_mysql_ids,
)


@pytest.mark.parametrize("value", [12346, "12346", "12346.0", 12346.0, Decimal("12346")])
def test_mysql_id_formats_share_identity(value):
    assert normalize_mysql_id(value) == "12346"


def test_large_integer_strings_keep_exact_identity():
    assert normalize_mysql_id("9007199254740993") == "9007199254740993"
    assert normalize_mysql_id("9007199254740993.0") == "9007199254740993"
    pandas_integer = pd.Series([9007199254740993], dtype="int64").iloc[0]
    assert normalize_mysql_id(pandas_integer) == "9007199254740993"


@pytest.mark.parametrize("value", [True, "abc", "1.5", 0, -1, "Infinity", "NaN", float(2**53)])
def test_invalid_ids_are_rejected(value):
    with pytest.raises(InvalidOccurrenceId):
        normalize_mysql_id(value)


@pytest.mark.parametrize("value", [None, pd.NA, float("nan"), "", "  "])
def test_missing_ids_have_no_identity(value):
    assert split_mysql_ids(value) == ()


def test_grouped_ids_are_normalized_without_inventing_victims():
    assert split_mysql_ids("5956 | 5956.0 | 5957") == ("5956", "5957")
    assert collect_mysql_ids(["5956 | 5957", "5957.0", None]) == {"5956", "5957"}


def test_malformed_group_does_not_silently_lose_ids():
    with pytest.raises(InvalidOccurrenceId):
        split_mysql_ids("42 | | 43")


def test_equivalent_copies_are_removed_and_processing_is_repeatable():
    original = pd.DataFrame([
        {"id": "42", "id_evento": "900", "tipo": "cvli", "qtd_mortes": 3},
        {"id": "42.0", "id_evento": "900.0", "tipo": "cvli", "qtd_mortes": 3},
        {"id": "43", "id_evento": "900", "tipo": "cvli", "qtd_mortes": 1},
    ], index=[7, 7, 8])
    untouched = original.copy(deep=True)
    result = deduplicate_mysql_rows(original)
    assert result["id"].tolist() == ["42", "43"]
    assert result["qtd_mortes"].tolist() == [3, 1]
    pd.testing.assert_frame_equal(deduplicate_mysql_rows(result), result)
    pd.testing.assert_frame_equal(original, untouched)


def test_grouped_copies_preserve_existing_death_count():
    frame = pd.DataFrame([
        {"id": "42 | 43", "qtd_mortes": 1},
        {"id": "43.0 | 42.0", "qtd_mortes": 1},
    ])
    result = deduplicate_mysql_rows(frame)
    assert len(result) == 1
    assert result.iloc[0]["qtd_mortes"] == 1


@pytest.mark.parametrize("changed", [
    {"id_evento": "901"}, {"data": "2026-01-02"}, {"qtd_mortes": 2},
])
def test_same_mysql_id_with_conflicting_content_is_reported(changed):
    first = {"id": "42", "id_evento": "900", "data": "2026-01-01", "qtd_mortes": 1}
    second = dict(first, id="42.0", **changed)
    with pytest.raises(OccurrenceIdentityConflict) as error:
        deduplicate_mysql_rows(pd.DataFrame([first, second]))
    assert error.value.ids == ("42",)
    assert error.value.positions == (0, 1)


def test_partial_group_overlap_is_not_discarded_or_counted_twice():
    frame = pd.DataFrame([
        {"id": "42 | 43", "qtd_mortes": 2},
        {"id": "43 | 44", "qtd_mortes": 2},
    ])
    with pytest.raises(OccurrenceIdentityConflict):
        deduplicate_mysql_rows(frame)


def test_missing_ids_are_kept_independently():
    frame = pd.DataFrame({"id": [None, None], "qtd_mortes": [1, 1]})
    assert len(deduplicate_mysql_rows(frame)) == 2
    without_id = frame.drop(columns="id")
    pd.testing.assert_frame_equal(deduplicate_mysql_rows(without_id), without_id)


@pytest.mark.parametrize("value", [None, pd.NA, "bad", 0, -1, float("nan"), float("inf")])
def test_invalid_counts_use_cvli_fallback(value):
    assert sanitize_death_count(value, " CVLI ") == 1
    assert sanitize_death_count(value, "cvp") == 0


def test_explicit_multiple_deaths_are_preserved():
    assert sanitize_death_count("3", "cvli") == 3
    assert sanitize_death_count(2, "cvli") == 2
