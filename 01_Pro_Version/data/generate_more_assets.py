import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

csv_file = "traffic_log.csv"

if not os.path.exists(csv_file):
    print("No traffic log found.")
    exit()

try:
    df = pd.read_csv(csv_file)
    if len(df) < 2:
        print("Not enough data.")
        exit()

    # --- GRAPH 1: Total Intersection Congestion ---
    plt.figure(figsize=(10, 6))
    total_cars = df['North Cars'] + df['South Cars'] + df['East Cars'] + df['West Cars']
    plt.plot(df['Time'], total_cars, color='purple', linewidth=3, marker='o', markersize=4)
    plt.fill_between(df['Time'], total_cars, color='purple', alpha=0.2)
    plt.title('Intersection Congestion (Total Waiting Vehicles)', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Total Cars Across All Lanes', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("total_congestion_chart.png", dpi=300)
    plt.close()

    # --- GRAPH 2: Green Light Time Allocation (Bar Chart) ---
    plt.figure(figsize=(8, 6))
    
    # Count how many times each light was "Green"
    green_counts = [
        (df['North Action'] == 'Green').sum(),
        (df['South Action'] == 'Green').sum(),
        (df['East Action'] == 'Green').sum(),
        (df['West Action'] == 'Green').sum()
    ]
    total_time = len(df)
    green_percentages = [(count / total_time) * 100 for count in green_counts]
    
    labels = ['North', 'South', 'East', 'West']
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']
    
    bars = plt.bar(labels, green_percentages, color=colors)
    plt.title('AI Phase Allocation: Green Light Percentage per Lane', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage of Time (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("signal_allocation_chart.png", dpi=300)
    plt.close()

    # --- Generate Tables (Markdown) ---
    with open("report_tables.md", "w") as f:
        f.write("### Table 1: Lane Traffic Density Analytics\n\n")
        f.write("| Lane Direction | Average Vehicles Waiting | Maximum Vehicles Waiting | Traffic Burden (%) |\n")
        f.write("|----------------|--------------------------|--------------------------|--------------------|\n")
        
        sum_total = total_cars.sum()
        for lane in ['North Cars', 'South Cars', 'East Cars', 'West Cars']:
            avg = df[lane].mean()
            maximum = df[lane].max()
            burden = (df[lane].sum() / sum_total * 100) if sum_total > 0 else 0
            name = lane.split(' ')[0]
            f.write(f"| {name} | {avg:.2f} | {maximum} | {burden:.1f}% |\n")
            
        f.write("\n### Table 2: AI Signal State Efficiency\n\n")
        f.write("| Intersection Axis | Total Green Time (sec) | Total Red Time (sec) | Signal Efficiency Ratio |\n")
        f.write("|-------------------|------------------------|----------------------|-------------------------|\n")
        
        for name, col in zip(['North', 'South', 'East', 'West'], ['North Action', 'South Action', 'East Action', 'West Action']):
            green_sec = (df[col] == 'Green').sum()
            red_sec = (df[col] == 'Red').sum()
            ratio = (green_sec / red_sec) if red_sec > 0 else green_sec
            f.write(f"| {name} | {green_sec}s | {red_sec}s | {ratio:.2f} |\n")

    print("Success")
except Exception as e:
    print(f"Error: {e}")
