import pickle
import pandas as pd
import numpy as np

DATA_FILE = 'data/processed/processed_graph_data.pkl'

def check_seasonality():
    try:
        with open(DATA_FILE, 'rb') as f:
            pack = pickle.load(f)

        dates = pd.to_datetime(pack['dates'])
        node_features = pack['node_features']

        print(f"Dataset Range: {dates.min().date()} to {dates.max().date()}")
        print(f"Total Days: {len(dates)}")

        last_date = dates.max()
        next_day = last_date + pd.Timedelta(days=1)
        print(f"\nLast Date in Data: {last_date.day_name()} {last_date.date()}")
        print(f"Prediction Target (Horizon 1): {next_day.day_name()} {next_day.date()}")

        # Analyze CVP (feature index 1) by day of week
        # Create a dataframe for easier aggregation
        # We need to aggregate across all nodes first? Or average across nodes?
        # Let's average across nodes to get a "City-wide CVP Index" per day

        # node_features shape: (Nodes, Timesteps, Features)
        cvp_data = node_features[:, :, 1] # (Nodes, Timesteps)
        daily_cvp_sum = np.sum(cvp_data, axis=0) # (Timesteps,)

        df = pd.DataFrame({'date': dates, 'cvp': daily_cvp_sum})
        df['day_name'] = df['date'].dt.day_name()

        print("\nAverage CVP Incidents by Day of Week:")
        stats = df.groupby('day_name')['cvp'].mean().sort_values()
        print(stats)

        target_day_name = next_day.day_name()
        avg_target = stats[target_day_name]
        overall_avg = df['cvp'].mean()

        print(f"\nTarget Day ({target_day_name}) Average: {avg_target:.2f}")
        print(f"Overall Average: {overall_avg:.2f}")

        if avg_target < overall_avg * 0.9:
            print(f"CONCLUSION: {target_day_name} has significantly lower activity than average.")
        elif avg_target > overall_avg * 1.1:
             print(f"CONCLUSION: {target_day_name} has significantly higher activity than average.")
        else:
             print(f"CONCLUSION: {target_day_name} is an average day.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_seasonality()
