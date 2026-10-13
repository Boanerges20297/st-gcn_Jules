from src.core.fortaleza_bairros import (
    canonicalize_fortaleza_bairro,
    load_official_fortaleza_bairros,
)


def test_official_mesh_has_exactly_121_unique_bairros_with_coordinates():
    bairros = load_official_fortaleza_bairros()

    assert len(bairros) == 121
    assert bairros['name'].is_unique
    assert bairros[['lat', 'long']].notna().all().all()


def test_reviewed_aliases_resolve_but_rmf_city_does_not_become_fortaleza_bairro():
    official = set(load_official_fortaleza_bairros()['name'])

    assert canonicalize_fortaleza_bairro('PANAMERICANO', official) == 'PAN AMERICANO'
    assert canonicalize_fortaleza_bairro('PRAIA DO FURUTO II', official) == 'PRAIA DO FUTURO II'
    assert canonicalize_fortaleza_bairro('AQUIRAZ', official) is None
    assert canonicalize_fortaleza_bairro('IPARANA', official) is None
