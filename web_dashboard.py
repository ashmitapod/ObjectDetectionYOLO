"""
Simple Web Dashboard for YOLO Detection System
View detection logs and clips through a web browser
Requires Flask: pip install flask

Run: python web_dashboard.py
Then open: http://localhost:5000
"""

try:
    from flask import Flask, render_template_string, send_file, jsonify
    import pandas as pd
    import os
    import glob
    from datetime import datetime
except ImportError:
    print("❌ Flask not installed. Install with: pip install flask")
    exit(1)

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Detection Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            text-align: center;
            transition: transform 0.3s;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .stat-card h3 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .stat-card p {
            color: #666;
            font-size: 1.1em;
        }
        .section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .video-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s;
        }
        .video-card:hover {
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102,126,234,0.3);
        }
        .video-card h4 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .video-card p {
            color: #666;
            font-size: 0.9em;
            margin: 5px 0;
        }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 10px;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #764ba2;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #667eea;
            color: white;
            font-weight: 600;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .badge-success {
            background: #d4edda;
            color: #155724;
        }
        .badge-danger {
            background: #f8d7da;
            color: #721c24;
        }
        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 15px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 50px;
            font-size: 1.1em;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            transition: all 0.3s;
        }
        .refresh-btn:hover {
            background: #764ba2;
            transform: scale(1.05);
        }
        .no-data {
            text-align: center;
            padding: 40px;
            color: #999;
            font-size: 1.2em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 YOLO Detection Dashboard</h1>
            <p>Real-time Monitoring & Analytics</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <h3>{{ total_detections }}</h3>
                <p>Total Detections</p>
            </div>
            <div class="stat-card">
                <h3>{{ roi_detections }}</h3>
                <p>ROI Detections</p>
            </div>
            <div class="stat-card">
                <h3>{{ alert_count }}</h3>
                <p>Alerts Triggered</p>
            </div>
            <div class="stat-card">
                <h3>{{ video_count }}</h3>
                <p>Recorded Clips</p>
            </div>
        </div>

        <div class="section">
            <h2>📹 Recent Alert Videos</h2>
            {% if videos %}
            <div class="video-grid">
                {% for video in videos %}
                <div class="video-card">
                    <h4>Alert #{{ loop.index }}</h4>
                    <p>📅 {{ video.date }}</p>
                    <p>⏱️ {{ video.time }}</p>
                    <p>📦 {{ video.size }} MB</p>
                    <a href="/video/{{ video.filename }}" class="btn" download>⬇️ Download</a>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <div class="no-data">No alert videos found</div>
            {% endif %}
        </div>

        <div class="section">
            <h2>📊 Recent Detections</h2>
            {% if recent_logs %}
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Object</th>
                        <th>Confidence</th>
                        <th>Location</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for log in recent_logs %}
                    <tr>
                        <td>{{ log.time }}</td>
                        <td><strong>{{ log.object }}</strong></td>
                        <td>{{ log.confidence }}%</td>
                        <td>{{ log.zone }}</td>
                        <td>
                            {% if log.in_roi %}
                            <span class="badge badge-danger">IN ROI</span>
                            {% else %}
                            <span class="badge badge-success">Normal</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <div class="no-data">No detection logs found</div>
            {% endif %}
        </div>
    </div>

    <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
</body>
</html>
"""

def get_statistics():
    """Get detection statistics"""
    stats = {
        'total_detections': 0,
        'roi_detections': 0,
        'alert_count': 0,
        'video_count': 0
    }
    
    # Count videos
    video_files = glob.glob("outputs/clips/alert_*.avi")
    stats['video_count'] = len(video_files)
    
    # Read logs
    log_files = glob.glob("outputs/logs/detections_*.csv")
    if log_files:
        all_data = []
        for log_file in log_files:
            try:
                df = pd.read_csv(log_file)
                all_data.append(df)
            except:
                pass
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            stats['total_detections'] = len(combined_df)
            stats['roi_detections'] = combined_df['In_ROI'].sum()
            stats['alert_count'] = combined_df['Alert_Triggered'].sum()
    
    return stats

def get_recent_videos(limit=12):
    """Get recent video clips"""
    video_files = sorted(glob.glob("outputs/clips/alert_*.avi"), reverse=True)[:limit]
    
    videos = []
    for video_path in video_files:
        filename = os.path.basename(video_path)
        timestamp = filename.replace('alert_', '').replace('.avi', '')
        
        try:
            date_part = timestamp[:8]
            time_part = timestamp[9:]
            date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
            time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
        except:
            date = "Unknown"
            time = "Unknown"
        
        size = os.path.getsize(video_path) / (1024 * 1024)
        
        videos.append({
            'filename': filename,
            'date': date,
            'time': time,
            'size': f"{size:.2f}"
        })
    
    return videos

def get_recent_logs(limit=20):
    """Get recent detection logs"""
    log_files = sorted(glob.glob("outputs/logs/detections_*.csv"), reverse=True)
    
    if not log_files:
        return []
    
    recent_logs = []
    for log_file in log_files[:3]:  # Check last 3 days
        try:
            df = pd.read_csv(log_file)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values('Timestamp', ascending=False)
            
            for _, row in df.head(limit).iterrows():
                recent_logs.append({
                    'time': row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'object': row['Object'],
                    'confidence': int(float(row['Confidence']) * 100),
                    'zone': row['Zone'],
                    'in_roi': row['In_ROI']
                })
                
                if len(recent_logs) >= limit:
                    break
            
            if len(recent_logs) >= limit:
                break
        except:
            continue
    
    return recent_logs

@app.route('/')
def index():
    """Main dashboard page"""
    stats = get_statistics()
    videos = get_recent_videos()
    recent_logs = get_recent_logs()
    
    return render_template_string(
        HTML_TEMPLATE,
        total_detections=stats['total_detections'],
        roi_detections=stats['roi_detections'],
        alert_count=stats['alert_count'],
        video_count=stats['video_count'],
        videos=videos,
        recent_logs=recent_logs
    )

@app.route('/video/<filename>')
def get_video(filename):
    """Serve video file"""
    video_path = os.path.join("outputs/clips", filename)
    if os.path.exists(video_path):
        return send_file(video_path, as_attachment=True)
    return "Video not found", 404

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    return jsonify(get_statistics())

if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║            YOLO Detection Web Dashboard                   ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝

🌐 Starting web server...
📡 Access dashboard at: http://localhost:5000
🔄 Press Ctrl+C to stop

    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)