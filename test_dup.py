try:
    from app import app
    from app import orchestrator
    nodes_gdf = getattr(orchestrator, 'global_nodes_gdf', None)
    if nodes_gdf is not None:
        import pandas as pd
        print(nodes_gdf['name'].value_counts().head(10))
    else:
        print("nodes_gdf is None")
except Exception as e:
    import traceback
    traceback.print_exc()
