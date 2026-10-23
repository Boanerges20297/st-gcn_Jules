import pandas as pd
try:
    import pytest
except ImportError:
    from contextlib import contextmanager
    class _FallbackPytest:
        @staticmethod
        @contextmanager
        def raises(exception):
            try:
                yield
            except exception:
                return
            raise AssertionError("Expected exception not raised")
    pytest = _FallbackPytest()
from src.core.cvli_source_canonical import source_aliases
from src.core.occurrence_integrity import collect_mysql_ids
from scripts.repair_enriched_cvli_from_mysql import build_candidate
def source(i,victim,agency,**extra):
    return dict(id=i,id_evento=1000+i,nome_vitima=victim,agencia=agency,
        tipo="CVLI",tipo_evento="HOMICIDIO DOLOSO",sexo="M",idade=31,arma="ARMA DE FOGO",
        cidade="FORTALEZA",bairro="CENTRO",latitude="-3.72000",
        longitude="-38.53000",data="2025-02-14",hora="13:45:00",ais="AIS1",**extra)
def local(rows,quantities=None):
    clean=[]
    for i,r in enumerate(rows):
        item={k:v for k,v in r.items() if k!="nome_vitima"}
        item["qtd_mortes"]=quantities[i] if quantities is not None else (1 if i%2==0 else None)
        clean.append(item)
    return pd.DataFrame(clean)
def test_same_victim_distinct_mysql_id_is_one_victim():
    a,b=source(1,"PERSON ONE","AGENCY A"),source(2,"PERSON ONE","AGENCY B")
    aliases,stats=source_aliases([a,b])
    assert aliases==[("1","2")]
    result,report=build_candidate(local([a,b]),[a,b])
    assert len(result)==1
    assert result.id.iloc[0]=="1 | 2"
    assert result.qtd_mortes.sum()==1
    assert collect_mysql_ids(result.id)=={"1","2"}
    assert report["removed_duplicate_representations"]==1
def test_two_victims_in_same_incident_are_preserved():
    a,b=source(1,"PERSON ONE","AGENCY A"),source(2,"PERSON TWO","AGENCY B")
    result,report=build_candidate(local([a,b]),[a,b])
    assert len(result)==2
    assert result.qtd_mortes.sum()==2
    assert report["removed_duplicate_representations"]==0
def test_conflicting_age_blocks_merge():
    a=source(1,"SAME NAME","AGENCY A")
    b=dict(source(2,"SAME NAME","AGENCY B"),idade=57)
    groups,stats=source_aliases([a,b])
    assert groups==[]
    assert stats["ambiguous_groups_preserved"]==1
    result,_=build_candidate(local([a,b]),[a,b])
    assert len(result)==2
def test_no_cross_agency_leaves_uncertain_pair():
    a,b=source(1,"SAME NAME","AGENCY A"),source(2,"SAME NAME","AGENCY A")
    groups,_=source_aliases([a,b])
    assert groups==[]
def test_empty_name_does_not_merge():
    a,b=source(1,"","A"),source(2,"","B")
    assert source_aliases([a,b])[0]==[]
def test_duplicate_id_blocks_instead_of_dropping_records():
    a=source(1,"PERSON ONE","A")
    with pytest.raises(ValueError):
        source_aliases([a,a])
def test_non_cvli_preserved():
    a,b=source(1,"PERSON ONE","A"),source(2,"PERSON ONE","B")
    c=dict(source(3,"OTHER","A"),tipo="CVP")
    result,_=build_candidate(local([a,b,c]),[a,b,c])
    assert len(result)==2
    assert result.loc[result.tipo=="CVP"].id.tolist()==["3"]
def test_repeated_execution_is_idempotent_and_preserves_ids():
    a,b=source(1,"PERSON ONE","A"),source(2,"PERSON ONE","B")
    first,_=build_candidate(local([a,b]),[a,b])
    second,report=build_candidate(first,[a,b])
    assert len(second)==len(first)
    assert report["removed_duplicate_representations"]==0
    assert report["already_grouped"]==1

def test_multivictim_flag_survives_victim_level_deduplication():
    from src.core.cvli_source_canonical import multi_victim_event_flags
    a=source(1,"PERSON A","ORG1")
    b=source(2,"PERSON A","ORG2")
    c=source(3,"PERSON B","ORG2")
    clean,_=build_candidate(local([a,b,c]),[a,b,c])
    flags=multi_victim_event_flags(clean)
    assert len(clean)==2
    assert flags.tolist()==[True,True]
    assert clean.qtd_mortes.sum()==2
def test_multivictim_flag_does_not_count_other_time_or_cvp():
    from src.core.cvli_source_canonical import multi_victim_event_flags
    a=source(1,"PERSON A","ORG1")
    b=dict(source(2,"PERSON B","ORG2"),hora="17:00:00")
    c=dict(source(3,"PERSON C","ORG3"),tipo="CVP")
    clean,_=build_candidate(local([a,b,c]),[a,b,c])
    flags=multi_victim_event_flags(clean)
    assert len(clean)==3
    assert flags.tolist()==[False,False,False]
def test_source_alias_extension_adds_new_copy_without_extra_death():
    a=source(1,"PERSON A","ORG1")
    b=source(2,"PERSON A","ORG2")
    c=source(3,"PERSON A","ORG3")
    first,_=build_candidate(local([a,b]),[a,b])
    third=local([c])
    combined=pd.concat([first,third],ignore_index=True)
    repaired,report=build_candidate(combined,[a,b,c])
    assert len(repaired)==1
    assert repaired.id.iloc[0]=="1 | 2 | 3"
    assert repaired.qtd_mortes.sum()==1
    assert collect_mysql_ids(repaired.id)=={"1","2","3"}
    assert report["removed_duplicate_representations"]==1
