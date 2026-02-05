"""Visualization utilities for faction data in web interface.

Provides functions to generate faction-aware visualizations for the web frontend.
"""

import json
import os
import pandas as pd
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KML_MAPPING = os.path.join(BASE_DIR, 'data', 'processed', 'faction_from_kml_mapping.json')
ANALYSIS_FILE = os.path.join(BASE_DIR, 'reports', 'faction_territory_analysis_corrected.json')


def get_faction_colors():
    """Return standardized faction color scheme for visualizations."""
    return {
        'COMANDO VERMELHO': '#A52714',
        'PRIMEIRO COMANDO DA CAPITAL': '#2CA02C',
        'TCP / GDE - TERCEIRO COMANDO PURO E GUARDIÕES DO ESTADO': '#1F77B4',
        'MASSA': '#FF7F0E',
        'COMUNIDADES EM DISPUTA': '#D62728',
        'OKAIDA': '#9467BD',
        'TERRITÓRIOS FANTASMAS': '#8C564B',
        'N/A': '#7F7F7F'
    }


def get_faction_symbols():
    """Return Unicode symbols for each faction."""
    return {
        'COMANDO VERMELHO': '🔴',
        'PRIMEIRO COMANDO DA CAPITAL': '🟢',
        'TCP / GDE - TERCEIRO COMANDO PURO E GUARDIÕES DO ESTADO': '🔵',
        'MASSA': '🟠',
        'COMUNIDADES EM DISPUTA': '⚠️',
        'OKAIDA': '🟣',
        'TERRITÓRIOS FANTASMAS': '👻',
        'N/A': '⚪'
    }


def load_faction_analysis():
    """Load faction analysis data."""
    if not os.path.exists(ANALYSIS_FILE):
        return None
    
    try:
        with open(ANALYSIS_FILE, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except:
        return None


def get_faction_stats_for_api():
    """Generate faction statistics for API endpoint.
    
    Returns:
        dict with faction rankings and statistics
    """
    analysis = load_faction_analysis()
    if not analysis:
        return None
    
    dominance = analysis.get('dominance', {})
    faction_data = analysis.get('analysis', {}).get('factions', {})
    
    stats = []
    for faction_name, dom_data in sorted(
        dominance.items(), 
        key=lambda x: -x[1]['placemarks']
    ):
        info = faction_data.get(faction_name, {})
        
        stats.append({
            'name': faction_name,
            'rank': dom_data['rank'],
            'placemarks': dom_data['placemarks'],
            'percentage': dom_data['percentage'],
            'territories': info.get('unique_territories', 0),
            'cities': info.get('geographic_spread', 0),
            'color': get_faction_colors().get(faction_name, '#7F7F7F'),
            'symbol': get_faction_symbols().get(faction_name, '•'),
            'sample_communities': info.get('sample_communities', [])
        })
    
    return {
        'factions': stats,
        'total_placemarks': analysis.get('analysis', {}).get('total_placemarks', 0),
        'num_factions': len(dominance),
        'timestamp': '2026-02-05'
    }


def generate_faction_legend_html():
    """Generate HTML legend for faction visualization."""
    colors = get_faction_colors()
    symbols = get_faction_symbols()
    
    html = '<div class="faction-legend">\n'
    html += '<h3>Facções Territoriais</h3>\n'
    html += '<ul style="list-style: none; padding: 0;">\n'
    
    for faction, color in sorted(colors.items()):
        symbol = symbols.get(faction, '•')
        html += f'  <li>'
        html += f'    <span style="color: {color}; font-size: 18px;">{symbol}</span>'
        html += f'    <span style="margin-left: 8px;">{faction}</span>'
        html += f'  </li>\n'
    
    html += '</ul>\n'
    html += '</div>\n'
    
    return html


def generate_faction_summary_for_node(node_name, faction):
    """Generate a summary card for a specific node's faction.
    
    Args:
        node_name: Name of the node
        faction: Faction name
    
    Returns:
        dict with node faction information
    """
    analysis = load_faction_analysis()
    if not analysis:
        return None
    
    faction_data = analysis.get('analysis', {}).get('factions', {}).get(faction)
    if not faction_data:
        return None
    
    return {
        'node': node_name,
        'faction': faction,
        'color': get_faction_colors().get(faction),
        'symbol': get_faction_symbols().get(faction),
        'territories': faction_data.get('unique_territories', 0),
        'cities': faction_data.get('geographic_spread', 0),
        'territory_strength': faction_data.get('territory_strength', 0),
        'main_cities': faction_data.get('cities', [])[:3],
        'main_communities': faction_data.get('sample_communities', [])[:3]
    }


def create_faction_comparison_data():
    """Create data for comparing factions.
    
    Returns:
        Formatted data for radar chart or comparison visualization
    """
    analysis = load_faction_analysis()
    if not analysis:
        return None
    
    dominance = analysis.get('dominance', {})
    faction_data = analysis.get('analysis', {}).get('factions', {})
    total_placemarks = analysis.get('analysis', {}).get('total_placemarks', 1)
    
    comparison = []
    for faction_name in sorted(dominance.keys(), key=lambda x: -dominance[x]['placemarks']):
        info = faction_data.get(faction_name, {})
        
        comparison.append({
            'faction': faction_name,
            'territorial_control': (dominance[faction_name]['placemarks'] / total_placemarks) * 100,
            'geographic_spread': info.get('geographic_spread', 0),
            'territorial_fragmentation': 100 - (info.get('territory_strength', 0.01) / 100),
            'placemarks': dominance[faction_name]['placemarks']
        })
    
    return comparison


if __name__ == '__main__':
    # Test
    print("Faction Visualization Utilities")
    print("=" * 50)
    
    stats = get_faction_stats_for_api()
    if stats:
        print("\n📊 Faction Statistics:")
        for faction in stats['factions']:
            print(f"{faction['symbol']} {faction['name']}: {faction['percentage']:.2f}%")
    
    print("\n🎨 Colors:")
    for faction, color in get_faction_colors().items():
        print(f"  {faction}: {color}")
    
    print("\n📈 Comparison Data:")
    comparison = create_faction_comparison_data()
    if comparison:
        for item in comparison[:3]:
            print(f"  {item['faction']}: {item['territorial_control']:.2f}%")
    
    print("\n✅ All utilities loaded successfully")
