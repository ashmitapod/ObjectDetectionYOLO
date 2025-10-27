"""
Analytics Viewer for YOLO Detection System
Generates visual reports and statistics from detection logs

Usage: python analytics_viewer.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import glob
import os
import numpy as np
from collections import Counter

def load_detection_logs():
    """Load all detection CSV logs"""
    log_files = glob.glob("outputs/logs/detections_*.csv")
    
    if not log_files:
        print("❌ No detection logs found in outputs/logs/")
        return None
    
    print(f"📂 Found {len(log_files)} log file(s)")
    
    all_data = []
    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)
            all_data.append(df)
            print(f"   ✅ Loaded: {os.path.basename(log_file)}")
        except Exception as e:
            print(f"   ⚠️  Error reading {log_file}: {e}")
    
    if not all_data:
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Convert timestamp to datetime
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
    
    # Convert confidence to float
    combined_df['Confidence'] = combined_df['Confidence'].astype(float)
    
    print(f"\n📊 Total detections loaded: {len(combined_df)}")
    
    return combined_df

def generate_summary_statistics(df):
    """Generate text summary of detection statistics"""
    print("\n" + "="*60)
    print("📈 DETECTION SUMMARY STATISTICS")
    print("="*60)
    
    # Time range
    start_date = df['Timestamp'].min()
    end_date = df['Timestamp'].max()
    print(f"\n📅 Data Range: {start_date} to {end_date}")
    print(f"⏱️  Duration: {(end_date - start_date).days} days, {(end_date - start_date).seconds // 3600} hours")
    
    # Total detections
    print(f"\n🔍 Total Detections: {len(df)}")
    
    # ROI vs Non-ROI
    roi_count = df['In_ROI'].sum()
    non_roi_count = len(df) - roi_count
    print(f"   • In ROI: {roi_count} ({roi_count/len(df)*100:.1f}%)")
    print(f"   • Outside ROI: {non_roi_count} ({non_roi_count/len(df)*100:.1f}%)")
    
    # Alerts triggered
    alert_count = df['Alert_Triggered'].sum()
    print(f"\n🚨 Alerts Triggered: {alert_count}")
    
    # Top detected objects
    print(f"\n🎯 Top 10 Detected Objects:")
    object_counts = df['Object'].value_counts().head(10)
    for i, (obj, count) in enumerate(object_counts.items(), 1):
        print(f"   {i}. {obj}: {count} ({count/len(df)*100:.1f}%)")
    
    # Average confidence
    avg_confidence = df['Confidence'].mean()
    print(f"\n💯 Average Confidence: {avg_confidence:.2%}")
    
    # Zone statistics
    print(f"\n📍 Detection by Zone:")
    zone_counts = df['Zone'].value_counts()
    for zone, count in zone_counts.items():
        print(f"   • {zone}: {count} ({count/len(df)*100:.1f}%)")
    
    # Hourly distribution
    df['Hour'] = df['Timestamp'].dt.hour
    hourly_counts = df['Hour'].value_counts().sort_index()
    peak_hour = hourly_counts.idxmax()
    print(f"\n⏰ Peak Activity Hour: {peak_hour}:00 ({hourly_counts[peak_hour]} detections)")
    
    # Daily distribution
    df['DayOfWeek'] = df['Timestamp'].dt.day_name()
    daily_counts = df['DayOfWeek'].value_counts()
    busiest_day = daily_counts.idxmax()
    print(f"📆 Busiest Day: {busiest_day} ({daily_counts[busiest_day]} detections)")
    
    print("\n" + "="*60)
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'total_detections': len(df),
        'roi_detections': roi_count,
        'alerts': alert_count,
        'avg_confidence': avg_confidence,
        'peak_hour': peak_hour,
        'busiest_day': busiest_day
    }

def plot_detections_over_time(df):
    """Plot detections over time"""
    # Group by hour
    df['DateTime'] = df['Timestamp'].dt.floor('H')
    hourly_detections = df.groupby('DateTime').size()
    
    plt.figure(figsize=(14, 5))
    plt.plot(hourly_detections.index, hourly_detections.values, 
             linewidth=2, color='#667eea', marker='o', markersize=4)
    plt.fill_between(hourly_detections.index, hourly_detections.values, 
                     alpha=0.3, color='#667eea')
    
    plt.title('Detections Over Time (Hourly)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date & Time', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Detections', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.tight_layout()

def plot_top_objects(df):
    """Plot top detected objects"""
    top_objects = df['Object'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(top_objects)), top_objects.values, 
                    color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_objects))))
    
    plt.yticks(range(len(top_objects)), top_objects.index)
    plt.xlabel('Number of Detections', fontsize=12, fontweight='bold')
    plt.title('Top 10 Detected Objects', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_objects.values)):
        plt.text(value + max(top_objects.values)*0.01, i, str(value), 
                va='center', fontweight='bold')
    
    plt.tight_layout()

def plot_roi_distribution(df):
    """Plot ROI vs Non-ROI distribution"""
    roi_counts = df['In_ROI'].value_counts()
    labels = ['In ROI', 'Outside ROI']
    colors = ['#ff6b6b', '#4ecdc4']
    
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(roi_counts.values, 
                                        labels=labels,
                                        colors=colors,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        explode=(0.05, 0))
    
    # Style the text
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    plt.title('ROI vs Non-ROI Detections', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

def plot_hourly_heatmap(df):
    """Plot hourly activity heatmap"""
    df['Hour'] = df['Timestamp'].dt.hour
    df['Date'] = df['Timestamp'].dt.date
    
    # Create pivot table
    heatmap_data = df.groupby(['Date', 'Hour']).size().unstack(fill_value=0)
    
    if len(heatmap_data) > 0:
        plt.figure(figsize=(14, 6))
        plt.imshow(heatmap_data.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(label='Number of Detections')
        
        plt.title('Activity Heatmap (Hour vs Day)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Hour of Day', fontsize=12, fontweight='bold')
        
        # Set ticks
        plt.yticks(range(24), [f"{h:02d}:00" for h in range(24)])
        plt.xticks(range(len(heatmap_data)), 
                  [str(d) for d in heatmap_data.index], 
                  rotation=45, ha='right')
        
        plt.tight_layout()

def plot_confidence_distribution(df):
    """Plot confidence score distribution"""
    plt.figure(figsize=(10, 6))
    
    plt.hist(df['Confidence'], bins=20, color='#667eea', 
             edgecolor='black', alpha=0.7)
    plt.axvline(df['Confidence'].mean(), color='red', 
                linestyle='--', linewidth=2, 
                label=f'Mean: {df["Confidence"].mean():.2f}')
    
    plt.title('Detection Confidence Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Confidence Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()

def plot_zone_comparison(df):
    """Plot detections by zone"""
    zone_counts = df['Zone'].value_counts()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(zone_counts)), zone_counts.values,
                   color=plt.cm.Set3(np.linspace(0, 1, len(zone_counts))))
    
    plt.xticks(range(len(zone_counts)), zone_counts.index, rotation=45, ha='right')
    plt.ylabel('Number of Detections', fontsize=12, fontweight='bold')
    plt.title('Detections by Zone', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bar, value in zip(bars, zone_counts.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()

def save_summary_report(df, stats, output_dir="outputs"):
    """Save text summary report"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f"summary_report_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("YOLO DETECTION SYSTEM - ANALYTICS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Range: {stats['start_date']} to {stats['end_date']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"Total Detections: {stats['total_detections']}\n")
        f.write(f"ROI Detections: {stats['roi_detections']} ({stats['roi_detections']/stats['total_detections']*100:.1f}%)\n")
        f.write(f"Alerts Triggered: {stats['alerts']}\n")
        f.write(f"Average Confidence: {stats['avg_confidence']:.2%}\n")
        f.write(f"Peak Activity Hour: {stats['peak_hour']}:00\n")
        f.write(f"Busiest Day: {stats['busiest_day']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("TOP DETECTED OBJECTS\n")
        f.write("-"*70 + "\n\n")
        
        top_objects = df['Object'].value_counts().head(15)
        for i, (obj, count) in enumerate(top_objects.items(), 1):
            f.write(f"{i:2d}. {obj:20s}: {count:5d} ({count/len(df)*100:5.1f}%)\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("ZONE STATISTICS\n")
        f.write("-"*70 + "\n\n")
        
        zone_counts = df['Zone'].value_counts()
        for zone, count in zone_counts.items():
            f.write(f"{zone:20s}: {count:5d} ({count/len(df)*100:5.1f}%)\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\n💾 Summary report saved: {report_path}")
    return report_path

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║         YOLO Detection Analytics Viewer                 ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Load data
    df = load_detection_logs()
    
    if df is None or len(df) == 0:
        print("\n❌ No data to analyze. Run detection system first to generate logs.")
        return
    
    # Generate statistics
    stats = generate_summary_statistics(df)
    
    # Create visualizations
    print("\n📊 Generating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    print("   1/6 Plotting detections over time...")
    plt.subplot(2, 3, 1)
    plot_detections_over_time(df)
    plt.subplot(2, 3, 1)  # Reset to first subplot for proper rendering
    
    print("   2/6 Plotting top objects...")
    plt.subplot(2, 3, 2)
    plot_top_objects(df)
    
    print("   3/6 Plotting ROI distribution...")
    plt.subplot(2, 3, 3)
    plot_roi_distribution(df)
    
    print("   4/6 Plotting hourly heatmap...")
    plt.subplot(2, 3, 4)
    plot_hourly_heatmap(df)
    
    print("   5/6 Plotting confidence distribution...")
    plt.subplot(2, 3, 5)
    plot_confidence_distribution(df)
    
    print("   6/6 Plotting zone comparison...")
    plt.subplot(2, 3, 6)
    plot_zone_comparison(df)
    
    # Adjust layout and save
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join("outputs", f"analytics_report_{timestamp}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Analytics dashboard saved: {output_path}")
    
    # Save text report
    save_summary_report(df, stats)
    
    # Show plots
    print("\n🖼️  Displaying analytics dashboard...")
    print("   Close the window to exit.")
    plt.show()
    
    print("\n" + "="*60)
    print("✅ Analytics generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()