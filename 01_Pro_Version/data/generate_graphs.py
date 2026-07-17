import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV data
csv_file = "traffic_log.csv"

if not os.path.exists(csv_file):
    print("No traffic log found. Run the simulation first.")
    exit()

try:
    df = pd.read_csv(csv_file)
    
    # Check if we have enough data
    if len(df) < 2:
        print("Not enough data in CSV to generate a graph. Run the simulation longer.")
        exit()
        
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot densities
    plt.plot(df['Time'], df['North Cars'], label='North Lane', color='blue', linewidth=2)
    plt.plot(df['Time'], df['South Cars'], label='South Lane', color='cyan', linewidth=2)
    plt.plot(df['Time'], df['East Cars'], label='East Lane', color='red', linewidth=2)
    plt.plot(df['Time'], df['West Cars'], label='orange', color='orange', linewidth=2)
    
    plt.title('AI Traffic Quantum Pro: Vehicle Density Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Number of Waiting Vehicles', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the graph
    output_path = "traffic_density_chart.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Beautiful graph generated: {output_path}")

except Exception as e:
    print(f"Error generating graph: {e}")
