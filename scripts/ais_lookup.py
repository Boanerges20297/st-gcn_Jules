#!/usr/bin/env python3
"""
AIS Lookup Module
=================
Provides reverse-lookup from (cidade, bairro) to the correct AIS and RISP region
using the official AIS_Territorios.csv mapping (34 AIS).

Usage:
    from ais_lookup import AISLookup
    lookup = AISLookup("/path/to/project")
    ais_id, regiao_risp = lookup.resolve("Fortaleza", "Parangaba")
    # → ("AIS 05", "CAPITAL OESTE")
"""

import csv
import os
import re
import unicodedata
from pathlib import Path


def _normalize(text: str) -> str:
    """Remove accents, uppercase, strip extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = text.upper().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


class AISLookup:
    """Builds a reverse lookup table from AIS_Territorios.csv."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        csv_path = self.project_root / "data" / "raw" / "AIS_Territorios.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"AIS_Territorios.csv not found at {csv_path}")

        # bairro_map: (CIDADE_NORM, BAIRRO_NORM) → (ais_id, regiao_risp)
        self._bairro_map: dict[tuple[str, str], tuple[str, str]] = {}
        # cidade_map: CIDADE_NORM → (ais_id, regiao_risp)
        self._cidade_map: dict[str, tuple[str, str]] = {}

        self._load(csv_path)
        self._add_bairro_aliases()
        print(f"[AIS Lookup] Carregado: {len(self._bairro_map)} bairros + {len(self._cidade_map)} cidades/municipios mapeados para 34 AIS")

    def _load(self, csv_path: Path) -> None:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                tipo = (row.get("Tipo") or "").strip()
                ais_id = (row.get("Identificador_AIS") or "").strip()
                regiao_risp = (row.get("Regiao_RISP") or "").strip()
                localidades = (row.get("Localidades_e_Bairros_Abrangidos") or "").strip()

                if not ais_id or not localidades:
                    continue

                value = (ais_id, regiao_risp)

                if tipo == "Bairros":
                    # Format: "Fortaleza: Bairro1, Bairro2, ..."
                    self._parse_bairros(localidades, value)
                elif tipo == "Município":
                    # Format: "Caucaia (Sede e Distritos...)"
                    self._parse_municipio(localidades, value)
                elif tipo == "Agrupado":
                    # Format: "Cidade1, Cidade2, ..."
                    self._parse_agrupado(localidades, value)

    def _parse_bairros(self, text: str, value: tuple[str, str]) -> None:
        """Parse 'Fortaleza: Bairro1, Bairro2, ...'"""
        # Split on ':' to get city and bairros
        if ":" in text:
            city_part, bairros_part = text.split(":", 1)
        else:
            city_part = "Fortaleza"
            bairros_part = text

        city_norm = _normalize(city_part)
        bairros = [b.strip() for b in bairros_part.split(",") if b.strip()]

        for bairro in bairros:
            bairro_norm = _normalize(bairro)
            if bairro_norm:
                self._bairro_map[(city_norm, bairro_norm)] = value

    def _parse_municipio(self, text: str, value: tuple[str, str]) -> None:
        """Parse 'Caucaia (Sede e Distritos...)' → extract city name only."""
        # Remove parenthetical
        city = re.sub(r'\(.*\)', '', text).strip()
        city_norm = _normalize(city)
        if city_norm:
            # First match wins (AIS 12 Caucaia before AIS 26 Caucaia)
            if city_norm not in self._cidade_map:
                self._cidade_map[city_norm] = value

    def _parse_agrupado(self, text: str, value: tuple[str, str]) -> None:
        """Parse 'Cidade1, Cidade2, ...' → each city maps to this AIS."""
        # Handle special cases like "Santana do Cariri (Sul)"
        cities = [c.strip() for c in text.split(",") if c.strip()]
        for city in cities:
            # Remove parenthetical qualifiers
            city_clean = re.sub(r'\(.*\)', '', city).strip()
            city_norm = _normalize(city_clean)
            if city_norm:
                # Don't overwrite if already mapped (first match wins)
                if city_norm not in self._cidade_map:
                    self._cidade_map[city_norm] = value

    def _add_bairro_aliases(self) -> None:
        """
        Registers bairros that exist in real occurrence data but are not
        explicitly listed in AIS_Territorios.csv. Includes:
        - Fortaleza bairros with slightly different names
        - Caucaia bairro-level split between AIS 12 and AIS 26
        """
        FORT = "FORTALEZA"
        fortaleza_aliases = {
            # Bairro               → AIS (known geographic location)
            "AMADEU FURTADO":       ("AIS 06", "CAPITAL OESTE"),
            "CAJAZEIRAS":           ("AIS 16", "CAPITAL LESTE"),
            "ELLERY":               ("AIS 18", "CAPITAL OESTE"),
            "GUADALAJARA":          ("AIS 17", "CAPITAL OESTE"),
            "GUARARAPES":           ("AIS 19", "CAPITAL LESTE"),
            "JARDIM GUANABARA":     ("AIS 05", "CAPITAL OESTE"),
            "JOSE DE ALENCAR":      ("AIS 16", "CAPITAL LESTE"),
            "MARAPONGA":            ("AIS 05", "CAPITAL OESTE"),
            "PANAMERICANO":         ("AIS 05", "CAPITAL OESTE"),
            "PAN AMERICANO":        ("AIS 05", "CAPITAL OESTE"),
            "PARQUE SANTA ROSA":    ("AIS 20", "CAPITAL OESTE"),
            "TAUAPE":               ("AIS 19", "CAPITAL LESTE"),
            "SAO JOAO TAUAPE":      ("AIS 19", "CAPITAL LESTE"),
            "VILA VELHA":           ("AIS 18", "CAPITAL OESTE"),
            "ALAGADICO NOVO":       ("AIS 06", "CAPITAL OESTE"),
            "RODOLFO TEOFILO":      ("AIS 06", "CAPITAL OESTE"),
            
            # Additional Fortaleza Bairros & Naming Variations
            "VILA PERI":            ("AIS 05", "CAPITAL OESTE"),
            "VILA PERY":            ("AIS 05", "CAPITAL OESTE"),
            "PARQUE SAO JOSE":      ("AIS 20", "CAPITAL OESTE"),
            "PARQUE SÃO JOSÉ":      ("AIS 20", "CAPITAL OESTE"),
            "PARQUELANDIA":         ("AIS 06", "CAPITAL OESTE"),
            "PARQUELÂNDIA":         ("AIS 06", "CAPITAL OESTE"),
            "BOA VISTA/CASTELAO":   ("AIS 22", "CAPITAL LESTE"),
            "BOA VISTA/CASTELÃO":   ("AIS 22", "CAPITAL LESTE"),
            "ENGENHEIRO LUCIANO CAVALCANTE": ("AIS 21", "CAPITAL LESTE"),
            "LUCIANO CAVALCANTE":   ("AIS 21", "CAPITAL LESTE"),
            "JOQUEI CLUBE":         ("AIS 06", "CAPITAL OESTE"),
            "JÓQUEI CLUBE":         ("AIS 06", "CAPITAL OESTE"),
            "NOVO MONDUBIM":        ("AIS 20", "CAPITAL OESTE"),
            "CONJUNTO ESPERANCA":   ("AIS 20", "CAPITAL OESTE"),
            "CONJUNTO ESPERANÇA":   ("AIS 20", "CAPITAL OESTE"),
            "PRESIDENTE KENNEDY":   ("AIS 18", "CAPITAL OESTE"),
            "PARQUE SANTA MARIA":   ("AIS 16", "CAPITAL LESTE"),
            "PRAIA DO FUTURO I":    ("AIS 19", "CAPITAL LESTE"),
            "PARQUE ARAXA":         ("AIS 06", "CAPITAL OESTE"),
            "PARQUE ARAXÁ":         ("AIS 06", "CAPITAL OESTE"),
            "MANOEL SATIRO":        ("AIS 20", "CAPITAL OESTE"),
            "MANOEL SÁTIRO":        ("AIS 20", "CAPITAL OESTE"),
            "JARDIM CEARENSE":      ("AIS 20", "CAPITAL OESTE"),
            "CONJUNTO CEARA II":    ("AIS 17", "CAPITAL OESTE"),
            "CONJUNTO CEARÁ II":    ("AIS 17", "CAPITAL OESTE"),
            "CONJUNTO CEARA I":     ("AIS 17", "CAPITAL OESTE"),
            "CONJUNTO CEARÁ I":     ("AIS 17", "CAPITAL OESTE"),
            "PARQUE IRACEMA":       ("AIS 16", "CAPITAL LESTE"),
            "SAPIRANGA / COITE":    ("AIS 21", "CAPITAL LESTE"),
            "SAPIRANGA / COITÉ":    ("AIS 21", "CAPITAL LESTE"),
            "JARDIM IRACEMA":       ("AIS 18", "CAPITAL OESTE"),
            "FLORESTA":             ("AIS 18", "CAPITAL OESTE"),
            "PRAIA DO FUTURO II":   ("AIS 19", "CAPITAL LESTE"),
            "ARACAPE":              ("AIS 20", "CAPITAL OESTE"),
            "ARACAPÉ":              ("AIS 20", "CAPITAL OESTE"),
            "RACHEL DE QUEIROZ":    ("AIS 17", "CAPITAL OESTE"),
            "SALINAS":              ("AIS 21", "CAPITAL LESTE"),
            "CIDADE NOVA":          ("AIS 20", "CAPITAL OESTE"),
            "GENIBAU":              ("AIS 06", "CAPITAL OESTE"),
            "GENIBAÚ":              ("AIS 06", "CAPITAL OESTE"),
            "PARQUE PRESIDENTE VARGAS": ("AIS 17", "CAPITAL OESTE"),
            "SAO MIGUEL":           ("AIS 19", "CAPITAL LESTE"),
            "SÃO MIGUEL":           ("AIS 19", "CAPITAL LESTE"),
            "ALTO ALEGRE II":       ("AIS 20", "CAPITAL OESTE"),
            "SABIAGUABA":           ("AIS 19", "CAPITAL LESTE"),
            "PRECABURA":            ("AIS 21", "CAPITAL LESTE"),
            "SANTA CLARA":          ("AIS 20", "CAPITAL OESTE"),

            # Mislabeled neighborhoods (recorded as Fortaleza but belong to other cities)
            "MARECHAL RONDON":      ("AIS 26", "RMF OESTE"),
            "DIF III":              ("AIS 14", "RMF LESTE"),
            "INDUSTRIAL":           ("AIS 14", "RMF LESTE"),
            "URUCUTUBA":            ("AIS 26", "RMF OESTE"),
            "PARQUE ALBANO":        ("AIS 26", "RMF OESTE"),
            "PARQUE SOLEDADE":      ("AIS 12", "RMF OESTE"),
            "PARQUE DAS NACOES":    ("AIS 26", "RMF OESTE"),
            "PARQUE DAS NAÇÕES":    ("AIS 26", "RMF OESTE"),
            "TABAPUA":              ("AIS 26", "RMF OESTE"),
            "TABAPUÃ":              ("AIS 26", "RMF OESTE"),
            "CARARU":               ("AIS 24", "RMF LESTE"),
            "IPARANA":              ("AIS 12", "RMF OESTE"),
            "PARQUE LEBLON":        ("AIS 12", "RMF OESTE"),
        }
        for bairro, value in fortaleza_aliases.items():
            key = (FORT, bairro)
            if key not in self._bairro_map:
                self._bairro_map[key] = value

        # --- CAUCAIA: AIS 12 (Sede/Oeste) e AIS 26 (Jurema/Leste) ---
        CAUC = "CAUCAIA"
        ais12 = ("AIS 12", "RMF OESTE")
        ais26 = ("AIS 26", "RMF OESTE")

        caucaia_ais12 = [
            "ALTO DO GARROTE", "BOM JESUS", "CABATAN", "ACUDE", "BARRA NOVA",
            "CAMURUPIM", "CUMBUCO", "CURICACA", "ANIL", "CAPUAN", "CENTRO",
            "CIGANA", "CIPO", "JARDIM ICARAI", "GARROTE", "GRILO", "ITAPOA",
            "ICARAI", "ITAMBE", "MESTRE ANTONIO", "LAGOA DO BANANA", "IPARANA",
            "JANDAIGUABA", "NOVA CIGANA", "NOVO PABUSSU", "PABUSSU", "PACHECO",
            "PADRE JULIO MARIA", "PADRE ROMUALDO", "PARQUE LEBLON", "PAUMIRIM",
            "PLANALTO CAUCAIA", "PARQUE SOLEDADE", "PITOMBEIRA", "MIXIRA",
            "CAUIPE", "TABUBA", "GUAGIRU", "CARAUBAS", "COITE", "MATOES",
            "LAGOA DOS PORCOS", "GENIPABU", "CORREGO DO ALEXANDRE", "PORTEIRAS",
            "ANGICO", "GUARARU", "CAUIPE 2", "PAU BRANCO", "ARATICUBA",
            "BOQUEIRAO DA ARARA", "BOQUEIRÃOZINHO", "CAMARA",
            "JACURUTU", "BOM TEMPO", "TABULEIRO GRANDE", "SANTA EDWIGES",
            "SAO BENTO", "INDUSTRIAL", "JUNCO", "SANTA ROSA", "JAPUARA",
            "CATUANA", "SAO PEDRO", "PRIMAVERA", "MANGABEIRA",
            "SITIOS NOVOS", "SEDE",
        ]
        caucaia_ais26 = [
            "CONJUNTO METROPOLITANO", "GUADALAJARA", "LAGO VERDE", "PARQUE ALBANO",
            "PARQUE DAS NACOES", "POTIRA", "SAO MIGUEL", "PATRICIA GOMES",
            "SOBRADINHO", "TABAPUA", "TABAPUA BRASILIA", "TABAPUA BRASILIA II",
            "MARECHAL RONDON", "ARATURI", "NOVA METROPOLE", "ARIANOPOLES",
            "CAMPO GRANDE", "URUCUTUBA", "TOCO", "RIACHAO", "CARRAPICHO",
            "TUCUNDUBA", "BOM PRINCIPIO", "MIRAMBE", "JUREMA",
        ]

        for bairro in caucaia_ais12:
            key = (CAUC, _normalize(bairro))
            if key not in self._bairro_map:
                self._bairro_map[key] = ais12

        for bairro in caucaia_ais26:
            key = (CAUC, _normalize(bairro))
            if key not in self._bairro_map:
                self._bairro_map[key] = ais26

        # --- Cidades que aparecem nos dados mas faltam no AIS_Territorios.csv ---
        extra_cities = {
            "CAUCAIA":           ("AIS 12", "RMF OESTE"),     # default fallback for empty bairro Caucaia
            "VARJOTA":           ("AIS 07", "NORTE"),         # near Crateus/Nova Russas
            "IPU":               ("AIS 07", "NORTE"),         # near Crateus
            "FRECHEIRINHA":      ("AIS 11", "NORDESTE"),      # near Tiangua/Vicosa
            "ACARAPE":           ("AIS 28", "NORTE"),         # near Caninde/Baturite
            "BOA VIAGEM":        ("AIS 28", "NORTE"),         # near Caninde/Madalena
            "MISSAO VELHA":      ("AIS 32", "SUL"),           # near Brejo Santo/Milagres
            "SANTA QUITERIA":    ("AIS 07", "NORTE"),         # near Crateus
            "CARIRE":            ("AIS 07", "NORTE"),         # near Crateus
            "URUOCA":            ("AIS 27", "NORTE"),         # near Camocim/Granja
            "IRAUCUBA":          ("AIS 04", "NORDESTE"),      # near Itapipoca
            "BATURITE":          ("AIS 28", "NORTE"),         # near Caninde/Acarape
            "BELA CRUZ":         ("AIS 27", "NORTE"),         # near Camocim/Acarau
            "AURORA":            ("AIS 34", "SUL"),           # near Ico/Lavras
            "IRACEMA":           ("AIS 30", "SUDESTE"),       # near Taua/Parambu
            "SAO LUIS DO CURU":  ("AIS 15", "RMF LESTE"),     # near Maranguape/Pacatuba
            "GROAIRAS":          ("AIS 03", "NORTE"),         # near Sobral
            "GROA":              ("AIS 03", "NORTE"),         # alias Groairas
            "HIDROLANDIA":       ("AIS 07", "NORTE"),         # near Crateus
            "GRACA":             ("AIS 11", "NORDESTE"),      # near Tiangua
            "PENTECOSTE":        ("AIS 28", "NORTE"),         # near Caninde
            "QUIXERE":           ("AIS 09", "SUDESTE"),       # near Russas (already in AIS 09 text)
            "JARDIM":            ("AIS 02", "SUL"),           # already in AIS 02 text

            # Additional interior cities
            "ARACOIABA":         ("AIS 28", "NORTE"),
            "ARATUBA":           ("AIS 28", "NORTE"),
            "ITAPIUNA":          ("AIS 28", "NORTE"),
            "ITAPIÚNA":          ("AIS 28", "NORTE"),
            "ALTO SANTO":        ("AIS 09", "SUDESTE"),
            "CAPISTRANO":        ("AIS 28", "NORTE"),
            "MULUNGU":           ("AIS 28", "NORTE"),
            "JAGUARETAMA":       ("AIS 09", "SUDESTE"),
            "JAGUARIBE":         ("AIS 34", "SUL"),
            "BARREIRA":          ("AIS 28", "NORTE"),
            "COREAU":            ("AIS 27", "NORTE"),
            "COREAÚ":            ("AIS 27", "NORTE"),
            "MUCAMBO":           ("AIS 11", "NORDESTE"),
            "REDENCAO":          ("AIS 28", "NORTE"),
            "REDENÇÃO":          ("AIS 28", "NORTE"),
            "JAGUARIBARA":       ("AIS 09", "SUDESTE"),
            "PEREIRO":           ("AIS 34", "SUL"),
            "PALMACIA":          ("AIS 28", "NORTE"),
            "PALMÁCIA":          ("AIS 28", "NORTE"),
            "OCARA":             ("AIS 28", "NORTE"),
            "RERIUTABA":         ("AIS 11", "NORDESTE"),
            "PACOTI":            ("AIS 28", "NORTE"),
            "PACUJA":            ("AIS 11", "NORDESTE"),
            "PACUJÁ":            ("AIS 11", "NORDESTE"),
            "CATUNDA":           ("AIS 07", "NORTE"),
            "PORTEIRAS":         ("AIS 32", "SUL"),
            "MARTINOPOLE":       ("AIS 27", "NORTE"),
            "MARTINÓPOLE":       ("AIS 27", "NORTE"),
            "GUARAMIRANGA":      ("AIS 28", "NORTE"),
            "MORAUJO":           ("AIS 27", "NORTE"),
            "MORAÚJO":           ("AIS 27", "NORTE"),
            "POTIRETAMA":        ("AIS 09", "SUDESTE"),
            "SENADOR SA":        ("AIS 03", "NORTE"),
            "SENADOR SÁ":        ("AIS 03", "NORTE"),
            "PIRES FERREIRA":    ("AIS 11", "NORDESTE"),
            "ERERE":             ("AIS 34", "SUL"),
            "ERERÊ":             ("AIS 34", "SUL"),
        }
        for city, value in extra_cities.items():
            city_norm = _normalize(city)
            if city_norm not in self._cidade_map:
                self._cidade_map[city_norm] = value

    def resolve(self, cidade: str, bairro: str) -> tuple[str, str]:
        """
        Resolve (cidade, bairro) → (ais_id, regiao_risp).
        
        Strategy:
        1. Try exact (city, bairro) match (Fortaleza/Caucaia bairros)
        2. Fallback: If 'bairro' is in another city's bairros, resolve to that city's AIS
        3. Fallback: If 'bairro' is actually a known city name, resolve to that city's AIS
        4. Try city-only match (municípios e agrupados)
        5. Return ("", "") if no match
        """
        cidade_norm = _normalize(cidade)
        bairro_norm = _normalize(bairro)

        # 1. Bairro-level match (Fortaleza/Caucaia)
        if bairro_norm:
            result = self._bairro_map.get((cidade_norm, bairro_norm))
            if result:
                return result

        # 2. Fallback: search for this neighborhood in any other city's mapping
        if bairro_norm:
            for (c, b), val in self._bairro_map.items():
                if b == bairro_norm:
                    return val

        # 3. Fallback: Check if the neighborhood field contains a known city name
        if bairro_norm:
            result = self._cidade_map.get(bairro_norm)
            if result:
                return result

        # 4. City-level match
        result = self._cidade_map.get(cidade_norm)
        if result:
            return result

        # 5. No match
        return ("", "")

    def resolve_series(self, cidades, bairros):
        """
        Vectorized resolve for pandas Series.
        Returns (ais_series, risp_series).
        """
        import pandas as pd

        ais_list = []
        risp_list = []

        for cidade, bairro in zip(cidades.fillna(""), bairros.fillna("")):
            ais_id, risp = self.resolve(str(cidade), str(bairro))
            ais_list.append(ais_id)
            risp_list.append(risp)

        return pd.Series(ais_list), pd.Series(risp_list)


if __name__ == "__main__":
    # Quick test
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1])
    lookup = AISLookup(root)

    tests = [
        ("Fortaleza", "Parangaba"),      # AIS 05
        ("Fortaleza", "Aldeota"),         # AIS 08
        ("Fortaleza", "Messejana"),       # AIS 16
        ("Fortaleza", "Bom Jardim"),      # AIS 17
        ("Fortaleza", "Centro"),          # AIS 18
        ("Fortaleza", "Aerolândia"),      # AIS 21
        ("Caucaia", ""),                  # AIS 12
        ("Maracanaú", ""),                # AIS 14
        ("Sobral", ""),                   # AIS 03
        ("Juazeiro do Norte", ""),        # AIS 02
        ("Quixadá", ""),                  # AIS 01
        ("Iguatu", ""),                   # AIS 10
        ("Horizonte", ""),                # AIS 25
        ("Cidade Inexistente", "Bairro"), # ("", "")
    ]

    print("\n--- AIS Lookup Tests ---")
    for cidade, bairro in tests:
        ais_id, risp = lookup.resolve(cidade, bairro)
        label = f"{cidade}/{bairro}" if bairro else cidade
        print(f"  {label:40} -> {ais_id:8} | {risp}")
