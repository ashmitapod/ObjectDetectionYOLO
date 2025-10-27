"""
Alert Video Compilation Tool for YOLO Detection System
Manages, compiles, and organizes alert video clips

Features:
- List all alert clips with details
- Compile multiple clips into one video
- Delete old clips
- Generate video summary
- Export clips by date range

Usage: python compile_alerts.py
"""

import cv2
import os
import glob
from datetime import datetime, timedelta
import argparse
import shutil

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def get_video_info(video_path):
    """Extract information from video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Parse timestamp from filename
        filename = os.path.basename(video_path)
        timestamp_str = filename.replace('alert_', '').replace('.avi', '')
        
        try:
            date_part = timestamp_str[:8]
            time_part = timestamp_str[9:]
            date_obj = datetime.strptime(date_part, '%Y%m%d')
            time_obj = datetime.strptime(time_part, '%H%M%S').time()
            timestamp = datetime.combine(date_obj.date(), time_obj)
        except:
            timestamp = None
        
        size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        
        return {
            'path': video_path,
            'filename': filename,
            'timestamp': timestamp,
            'duration': duration,
            'size': size,
            'resolution': (width, height),
            'fps': fps,
            'frames': frame_count
        }
    except Exception as e:
        print(f"   ⚠️  Error reading {video_path}: {e}")
        return None

def list_all_clips():
    """List all alert video clips"""
    print_header("ALERT VIDEO CLIPS")
    
    video_files = sorted(glob.glob("outputs/clips/alert_*.avi"))
    
    if not video_files:
        print("\n❌ No alert clips found in outputs/clips/")
        return []
    
    print(f"\n📹 Found {len(video_files)} video clip(s)\n")
    
    clips_info = []
    total_size = 0
    total_duration = 0
    
    for i, video_path in enumerate(video_files, 1):
        info = get_video_info(video_path)
        if info:
            clips_info.append(info)
            total_size += info['size']
            total_duration += info['duration']
            
            timestamp_str = info['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if info['timestamp'] else 'Unknown'
            
            print(f"{i:3d}. {info['filename']}")
            print(f"     📅 Time: {timestamp_str}")
            print(f"     ⏱️  Duration: {info['duration']:.1f} sec")
            print(f"     📦 Size: {info['size']:.2f} MB")
            print(f"     🎬 Resolution: {info['resolution'][0]}x{info['resolution'][1]} @ {info['fps']:.0f} FPS")
            print()
    
    print("-"*60)
    print(f"📊 Total: {len(clips_info)} clips | {total_duration:.1f} sec | {total_size:.2f} MB")
    print("="*60)
    
    return clips_info

def compile_clips(clips_info, output_name=None, date_filter=None):
    """Compile multiple clips into one video"""
    print_header("COMPILING VIDEO CLIPS")
    
    # Filter by date if specified
    if date_filter:
        clips_info = [c for c in clips_info if c['timestamp'] and c['timestamp'].date() == date_filter]
        print(f"\n📅 Filtering clips from: {date_filter}")
    
    if not clips_info:
        print("\n❌ No clips to compile")
        return False
    
    print(f"\n🎬 Compiling {len(clips_info)} clip(s)...")
    
    # Generate output filename
    if output_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"compiled_{timestamp}.avi"
    
    output_path = os.path.join("outputs", output_name)
    
    # Get video properties from first clip
    first_clip = clips_info[0]
    width, height = first_clip['resolution']
    fps = first_clip['fps']
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Failed to create output video")
        return False
    
    # Process each clip
    for i, clip in enumerate(clips_info, 1):
        print(f"   Processing clip {i}/{len(clips_info)}: {clip['filename']}")
        
        cap = cv2.VideoCapture(clip['path'])
        
        if not cap.isOpened():
            print(f"   ⚠️  Could not open {clip['filename']}, skipping...")
            continue
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add timestamp overlay
            if clip['timestamp']:
                timestamp_text = clip['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(frame, timestamp_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add clip info overlay
            clip_info_text = f"Clip {i}/{len(clips_info)}"
            cv2.putText(frame, clip_info_text, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        print(f"   ✅ Added {frame_count} frames from {clip['filename']}")
    
    out.release()
    
    # Get final video info
    final_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Compilation complete!")
    print(f"   📂 Output: {output_path}")
    print(f"   📦 Size: {final_size:.2f} MB")
    print(f"   🎬 Clips combined: {len(clips_info)}")
    
    return True

def delete_old_clips(days=7):
    """Delete clips older than specified days"""
    print_header(f"DELETING CLIPS OLDER THAN {days} DAYS")
    
    video_files = glob.glob("outputs/clips/alert_*.avi")
    
    if not video_files:
        print("\n❌ No clips found")
        return
    
    cutoff_date = datetime.now() - timedelta(days=days)
    deleted_count = 0
    deleted_size = 0
    
    print(f"\n🗑️  Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}\n")
    
    for video_path in video_files:
        info = get_video_info(video_path)
        if info and info['timestamp']:
            if info['timestamp'] < cutoff_date:
                try:
                    os.remove(video_path)
                    deleted_count += 1
                    deleted_size += info['size']
                    print(f"   ❌ Deleted: {info['filename']} ({info['timestamp'].strftime('%Y-%m-%d')})")
                except Exception as e:
                    print(f"   ⚠️  Could not delete {info['filename']}: {e}")
    
    if deleted_count == 0:
        print("   ℹ️  No old clips to delete")
    else:
        print(f"\n✅ Deleted {deleted_count} clip(s), freed {deleted_size:.2f} MB")

def export_by_date_range(start_date, end_date, output_dir=None):
    """Export clips within date range to separate folder"""
    print_header(f"EXPORTING CLIPS: {start_date} to {end_date}")
    
    if output_dir is None:
        output_dir = os.path.join("outputs", f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n📁 Created export directory: {output_dir}")
    
    video_files = glob.glob("outputs/clips/alert_*.avi")
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
    
    exported_count = 0
    exported_size = 0
    
    print(f"\n📤 Exporting clips...\n")
    
    for video_path in video_files:
        info = get_video_info(video_path)
        if info and info['timestamp']:
            if start_dt <= info['timestamp'] < end_dt:
                try:
                    dest_path = os.path.join(output_dir, info['filename'])
                    shutil.copy2(video_path, dest_path)
                    exported_count += 1
                    exported_size += info['size']
                    print(f"   ✅ Exported: {info['filename']}")
                except Exception as e:
                    print(f"   ⚠️  Could not export {info['filename']}: {e}")
    
    if exported_count == 0:
        print("   ℹ️  No clips found in date range")
    else:
        print(f"\n✅ Exported {exported_count} clip(s), total {exported_size:.2f} MB")
        print(f"📂 Location: {output_dir}")

def generate_summary():
    """Generate summary of all clips"""
    print_header("VIDEO SUMMARY REPORT")
    
    video_files = glob.glob("outputs/clips/alert_*.avi")
    
    if not video_files:
        print("\n❌ No clips found")
        return
    
    clips_info = []
    for video_path in video_files:
        info = get_video_info(video_path)
        if info:
            clips_info.append(info)
    
    if not clips_info:
        print("\n❌ Could not read any clips")
        return
    
    # Calculate statistics
    total_clips = len(clips_info)
    total_size = sum(c['size'] for c in clips_info)
    total_duration = sum(c['duration'] for c in clips_info)
    avg_duration = total_duration / total_clips
    
    dates = [c['timestamp'].date() for c in clips_info if c['timestamp']]
    if dates:
        date_range = f"{min(dates)} to {max(dates)}"
        unique_days = len(set(dates))
    else:
        date_range = "Unknown"
        unique_days = 0
    
    # Print summary
    print(f"\n📊 STATISTICS")
    print(f"   Total Clips: {total_clips}")
    print(f"   Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"   Average Duration: {avg_duration:.1f} seconds")
    print(f"   Total Size: {total_size:.2f} MB")
    print(f"   Date Range: {date_range}")
    print(f"   Days Covered: {unique_days}")
    
    # Clips by day
    if dates:
        print(f"\n📅 CLIPS BY DAY:")
        from collections import Counter
        day_counts = Counter(dates)
        for date, count in sorted(day_counts.items()):
            print(f"   {date}: {count} clip(s)")
    
    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = os.path.join("outputs", f"video_summary_{timestamp}.txt")
    
    with open(summary_path, 'w') as f:
        f.write("VIDEO CLIPS SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Clips: {total_clips}\n")
        f.write(f"Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)\n")
        f.write(f"Average Duration: {avg_duration:.1f} seconds\n")
        f.write(f"Total Size: {total_size:.2f} MB\n")
        f.write(f"Date Range: {date_range}\n")
        f.write(f"Days Covered: {unique_days}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("CLIP DETAILS\n")
        f.write("-"*60 + "\n\n")
        
        for i, clip in enumerate(sorted(clips_info, key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min), 1):
            timestamp_str = clip['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if clip['timestamp'] else 'Unknown'
            f.write(f"{i}. {clip['filename']}\n")
            f.write(f"   Timestamp: {timestamp_str}\n")
            f.write(f"   Duration: {clip['duration']:.1f} sec\n")
            f.write(f"   Size: {clip['size']:.2f} MB\n")
            f.write(f"   Resolution: {clip['resolution'][0]}x{clip['resolution'][1]}\n\n")
    
    print(f"\n💾 Summary saved to: {summary_path}")

def interactive_menu():
    """Interactive menu for clip management"""
    while True:
        print("\n" + "="*60)
        print("  YOLO ALERT VIDEO MANAGER")
        print("="*60)
        print("\n1. List all clips")
        print("2. Compile all clips into one video")
        print("3. Compile clips from specific date")
        print("4. Delete old clips")
        print("5. Export clips by date range")
        print("6. Generate summary report")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            list_all_clips()
            input("\nPress Enter to continue...")
        
        elif choice == '2':
            clips_info = []
            video_files = sorted(glob.glob("outputs/clips/alert_*.avi"))
            for video_path in video_files:
                info = get_video_info(video_path)
                if info:
                    clips_info.append(info)
            
            if clips_info:
                output_name = input("\nOutput filename (press Enter for auto): ").strip()
                if not output_name:
                    output_name = None
                elif not output_name.endswith('.avi'):
                    output_name += '.avi'
                
                compile_clips(clips_info, output_name)
            else:
                print("\n❌ No clips found to compile")
            
            input("\nPress Enter to continue...")
        
        elif choice == '3':
            date_str = input("\nEnter date (YYYY-MM-DD): ").strip()
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                clips_info = []
                video_files = sorted(glob.glob("outputs/clips/alert_*.avi"))
                for video_path in video_files:
                    info = get_video_info(video_path)
                    if info:
                        clips_info.append(info)
                
                if clips_info:
                    output_name = f"compiled_{date_str}.avi"
                    compile_clips(clips_info, output_name, date_filter=date_obj)
                else:
                    print("\n❌ No clips found")
            except ValueError:
                print("\n❌ Invalid date format. Use YYYY-MM-DD")
            
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            days_str = input("\nDelete clips older than how many days? (default: 7): ").strip()
            try:
                days = int(days_str) if days_str else 7
                confirm = input(f"\n⚠️  This will delete clips older than {days} days. Continue? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    delete_old_clips(days)
                else:
                    print("\n❌ Cancelled")
            except ValueError:
                print("\n❌ Invalid number")
            
            input("\nPress Enter to continue...")
        
        elif choice == '5':
            start_date = input("\nStart date (YYYY-MM-DD): ").strip()
            end_date = input("End date (YYYY-MM-DD): ").strip()
            
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
                datetime.strptime(end_date, '%Y-%m-%d')
                export_by_date_range(start_date, end_date)
            except ValueError:
                print("\n❌ Invalid date format. Use YYYY-MM-DD")
            
            input("\nPress Enter to continue...")
        
        elif choice == '6':
            generate_summary()
            input("\nPress Enter to continue...")
        
        elif choice == '7':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid option. Please select 1-7.")
            input("\nPress Enter to continue...")

def main():
    parser = argparse.ArgumentParser(description="YOLO Alert Video Manager")
    parser.add_argument("--list", action="store_true", help="List all clips")
    parser.add_argument("--compile", action="store_true", help="Compile all clips")
    parser.add_argument("--compile-date", type=str, help="Compile clips from specific date (YYYY-MM-DD)")
    parser.add_argument("--delete-old", type=int, metavar="DAYS", help="Delete clips older than N days")
    parser.add_argument("--export", type=str, nargs=2, metavar=("START", "END"), help="Export clips by date range (YYYY-MM-DD)")
    parser.add_argument("--summary", action="store_true", help="Generate summary report")
    parser.add_argument("--output", type=str, help="Output filename for compilation")
    
    args = parser.parse_args()
    
    # Check if outputs directory exists
    if not os.path.exists("outputs/clips"):
        print("❌ outputs/clips directory not found")
        print("   Make sure you're running this from the project root directory")
        return
    
    # If no arguments, show interactive menu
    if not any(vars(args).values()):
        interactive_menu()
        return
    
    # Execute based on arguments
    if args.list:
        list_all_clips()
    
    if args.compile:
        clips_info = []
        video_files = sorted(glob.glob("outputs/clips/alert_*.avi"))
        for video_path in video_files:
            info = get_video_info(video_path)
            if info:
                clips_info.append(info)
        
        if clips_info:
            compile_clips(clips_info, args.output)
        else:
            print("\n❌ No clips found to compile")
    
    if args.compile_date:
        try:
            date_obj = datetime.strptime(args.compile_date, '%Y-%m-%d').date()
            clips_info = []
            video_files = sorted(glob.glob("outputs/clips/alert_*.avi"))
            for video_path in video_files:
                info = get_video_info(video_path)
                if info:
                    clips_info.append(info)
            
            if clips_info:
                output_name = args.output or f"compiled_{args.compile_date}.avi"
                compile_clips(clips_info, output_name, date_filter=date_obj)
            else:
                print("\n❌ No clips found")
        except ValueError:
            print("\n❌ Invalid date format. Use YYYY-MM-DD")
    
    if args.delete_old:
        delete_old_clips(args.delete_old)
    
    if args.export:
        start_date, end_date = args.export
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
            export_by_date_range(start_date, end_date)
        except ValueError:
            print("\n❌ Invalid date format. Use YYYY-MM-DD")
    
    if args.summary:
        generate_summary()

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║       YOLO Detection - Alert Video Manager              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    main()
    
    print("\n" + "="*60)
    print("✅ Operation complete!")
    print("="*60)