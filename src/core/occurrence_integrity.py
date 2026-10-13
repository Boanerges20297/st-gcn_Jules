"""Pure helpers for MySQL row identity; no downloads, writes or enrichment.

``id`` identifies a MySQL row, not a victim or an occurrence. Legacy CSVs
may store several original row IDs separated by ``|``. An overlap between
different groups cannot safely be resolved by dropping either whole row.
"""

from decimal import Decimal, InvalidOperation
import math
from numbers import Integral, Real

import pandas as pd


class InvalidOccurrenceId(ValueError):
    """A supplied MySQL ID cannot be interpreted without losing identity."""


class OccurrenceIdentityConflict(ValueError):
    """The same MySQL ID appears in rows that cannot safely be collapsed."""

    def __init__(self, ids, first_position, second_position):
        self.ids = tuple(ids)
        self.positions = (first_position, second_position)
        super().__init__(
            f"Conflicting MySQL IDs {self.ids} at row positions {self.positions}"
        )


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))


def normalize_mysql_id(value):
    """Return a canonical positive integer string, or None for an empty ID.

    Parse strings with Decimal, so large integer IDs never pass through a
    float. Already floating-point IDs outside the exact integer range are
    rejected: their original identity may have been lost by the caller.
    """
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise InvalidOccurrenceId("Boolean values are not MySQL row IDs")
    if isinstance(value, Real) and not isinstance(value, Integral):
        if abs(value) > 2**53 - 1:
            raise InvalidOccurrenceId("Floating-point ID exceeds exact integer range")
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise InvalidOccurrenceId("MySQL ID must be a positive integer") from exc
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise InvalidOccurrenceId("MySQL ID must be a positive integer")
    return str(int(number))


def split_mysql_ids(value):
    """Extract unique original IDs from a scalar or a legacy ``|`` group."""
    if _is_missing(value):
        return ()
    parts = value.split("|") if isinstance(value, str) else [value]
    ids = []
    for part in parts:
        normalized = normalize_mysql_id(part)
        if normalized is None:
            raise InvalidOccurrenceId("Grouped IDs must not contain empty members")
        if normalized not in ids:
            ids.append(normalized)
    return tuple(ids)


def collect_mysql_ids(values):
    """Build the set of committed row IDs, including legacy grouped IDs."""
    return {row_id for value in values for row_id in split_mysql_ids(value)}


def sanitize_death_count(value, event_type):
    """Keep an explicit positive count; default a CVLI row to one death.

    This function deliberately does not infer victims from MySQL IDs.
    """
    try:
        count = float(value)
    except (TypeError, ValueError, OverflowError):
        count = float("nan")
    if math.isfinite(count) and count > 0:
        return count
    return 1.0 if str(event_type).strip().lower() == "cvli" else 0.0


def _equivalent_rows(first, second):
    # Comparing all supplied fields is conservative. Different enrichment,
    # event identifiers or counts must not silently overwrite another row.
    for column in first.index:
        if column == "id":
            continue
        left, right = first[column], second[column]
        left_missing, right_missing = _is_missing(left), _is_missing(right)
        if left_missing or right_missing:
            if left_missing != right_missing:
                return False
            continue
        if column == "id_evento":
            # Event identity is separate from MySQL row identity.
            left, right = normalize_mysql_id(left), normalize_mysql_id(right)
        if left != right:
            return False
    return True


def deduplicate_mysql_rows(frame):
    """Return a copy with canonical IDs and equivalent copies removed.

    Rows without an ID are preserved independently. Conflicting content or
    partial overlaps between grouped IDs raise an exception before returning
    any result. The caller decides how to report/reconcile these conflicts.
    Existing death counts are kept verbatim, including on grouped rows.
    """
    result = frame.copy()
    if "id" not in result.columns:
        return result
    result["id"] = result["id"].astype(object)
    id_column = result.columns.get_loc("id")
    owners = {}
    groups = {}
    keep = []
    for position in range(len(result)):
        row = result.iloc[position]
        ids = split_mysql_ids(row["id"])
        if not ids:
            keep.append(position)
            continue
        group = frozenset(ids)
        overlap = [row_id for row_id in ids if row_id in owners]
        if overlap:
            first_position = owners[overlap[0]]
            if (
                groups[first_position] != group
                or any(owners[row_id] != first_position for row_id in overlap)
                or not _equivalent_rows(result.iloc[first_position], row)
            ):
                raise OccurrenceIdentityConflict(overlap, first_position, position)
            continue
        result.iat[position, id_column] = " | ".join(ids)
        for row_id in ids:
            owners[row_id] = position
        groups[position] = group
        keep.append(position)
    return result.iloc[keep].copy()
