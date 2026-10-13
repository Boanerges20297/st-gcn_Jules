import pandas as pd

from scripts.reconcile_mysql_data import reconcile


def source(row_id, event, victim, **changes):
    row = {'id': row_id, 'id_evento': event, 'data': '2026-01-01', 'hora': '13:00:00',
           'tipo': 'cvli', 'cidade': 'FORTALEZA', 'latitude': '-3.7',
           'longitude': '-38.5', 'nome_vitima': victim}
    return dict(row, **changes)


def local(row, **changes):
    return dict({key: value for key, value in row.items() if key != 'nome_vitima'},
                qtd_mortes=1, **changes)


def test_different_victims_at_same_event_are_not_deduplicated():
    first, second = source(1, 101, 'PERSON ONE'), source(2, 102, 'PERSON TWO')
    result, report = reconcile(pd.DataFrame([local(first), local(second)]), [first, second])
    assert result['id'].tolist() == ['1', '2']
    assert result['qtd_mortes'].sum() == 2
    assert report['unresolved_rows_preserved'] == 0


def test_same_victim_name_is_not_enough_to_remove_distinct_mysql_rows():
    first, second = source(1, 101, 'SAME NAME'), source(2, 102, 'SAME NAME')
    result, _ = reconcile(pd.DataFrame([local(first), local(second)]), [first, second])
    assert len(result) == 2


def test_source_verified_duplicate_is_removed_and_wrong_id_reassigned():
    row = source(1, 101, 'PERSON ONE')
    result, report = reconcile(pd.DataFrame([local(row, id='99'), local(row)]), [row])
    assert result['id'].tolist() == ['1']
    assert result['qtd_mortes'].sum() == 1
    assert report['actions']['verified_id_associations_corrected'] == 1
    assert report['actions']['verified_duplicate_representations_removed'] == 1


def test_lost_group_quantity_is_restored_only_with_verified_individual_victims():
    first, second = source(1, 101, 'PERSON ONE'), source(2, 102, 'PERSON TWO')
    result, report = reconcile(pd.DataFrame([local(first, id='1 | 2')]), [first, second])
    assert result['id'].tolist() == ['1', '2']
    assert result['id_evento'].tolist() == ['101', '102']
    assert result['qtd_mortes'].sum() == 2
    assert 'nome_vitima' not in result
    assert report['actions']['lost_multi_victim_counts_recovered'] == 1


def test_group_does_not_swallow_other_victims_at_same_place():
    rows = [source(i, 100 + i, f'PERSON {i}') for i in range(1, 5)]
    result, _ = reconcile(pd.DataFrame([local(rows[0], id='1 | 2')]), rows)
    assert result['id'].tolist() == ['1', '2']


def test_same_name_or_unknown_group_members_remain_unresolved():
    first, second = source(1, 101, 'SAME NAME'), source(2, 102, 'SAME NAME')
    for group in ['1 | 2', '1 | 999']:
        result, report = reconcile(pd.DataFrame([local(first, id=group)]), [first, second])
        assert result.iloc[0]['id'] == group
        assert report['unresolved_rows_preserved'] == 1


def test_absent_source_record_is_preserved():
    row = local(source(1, 101, 'PERSON ONE'))
    result, report = reconcile(pd.DataFrame([row]), [])
    assert len(result) == 1
    assert report['unresolved_rows_preserved'] == 1


def test_explicit_count_is_preserved_and_not_summed_for_copies():
    row = source(1, 101, 'PERSON ONE')
    first = local(row)
    first['qtd_mortes'] = 3
    result, _ = reconcile(pd.DataFrame([first, local(row)]), [row])
    assert result.iloc[0]['qtd_mortes'] == 3


def test_common_event_id_does_not_collapse_different_victims():
    first, second = source(1, 101, 'PERSON ONE'), source(2, 101, 'PERSON TWO')
    result, report = reconcile(pd.DataFrame([local(first), local(second)]), [first, second])
    assert len(result) == 2
    assert report['unresolved_rows_preserved'] == 2


def test_group_and_individual_copies_count_each_verified_victim_once():
    first, second = source(1, 101, 'PERSON ONE'), source(2, 102, 'PERSON TWO')
    rows = [local(first, id='1 | 2'), local(first), local(second)]
    result, report = reconcile(pd.DataFrame(rows), [first, second])
    assert result['id'].tolist() == ['1', '2']
    assert result['qtd_mortes'].sum() == 2
    assert report['actions']['verified_duplicate_representations_removed'] == 2
    assert 'PERSON ONE' not in str(report)
    assert 'PERSON TWO' not in str(report)
