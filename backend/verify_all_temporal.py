import cv2
import numpy as np
import os
import wave
import struct
from forensic_engine import ForensicEngine

def create_advanced_dummy_video(filename, fps=30, duration_sec=6):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # A bit larger to give mediapipe a fighting chance if we drew a fake face (we won't for now)
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))
    
    # We also create a dummy audio file to test Lip Sync
    audio_file = "test_audio.wav"
    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # Create 6 seconds of random noise audio for lip sync
        audio_data = np.random.randint(-30000, 30000, 16000 * duration_sec, dtype=np.int16)
        wf.writeframes(struct.pack(f"<{len(audio_data)}h", *audio_data))
        
    for i in range(fps * duration_sec):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a shape that fluctuates wildly to trigger SSIM variance
        if i % 2 == 0:
            cv2.circle(frame, (320, 240), 100, (255, 255, 255), -1)
        else:
            cv2.rectangle(frame, (100, 100), (500, 300), (128, 128, 128), -1)
            
        out.write(frame)
    out.release()
    
    # Multiplex audio and video using ffmpeg
    final_vid = "final_test.mp4"
    import subprocess
    cmd = ['ffmpeg', '-y', '-i', filename, '-i', audio_file, '-c:v', 'copy', '-c:a', 'aac', final_vid]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    os.remove(filename)
    os.remove(audio_file)
    print(f"Created complete AV test file: {final_vid}")
    return final_vid
    
if __name__ == "__main__":
    test_file = "temp_av.mp4"
    final_file = create_advanced_dummy_video(test_file)
    
    engine = ForensicEngine()
    print("\n[VERIFICATION]: Running analyze_video()...\n")
    res = engine.analyze_video(final_file)
    
    print("--------------------------------------------------")
    print("1. SSIM FLICKERING")
    if 'temporal_flickering' in res:
        print(f"   Mean SSIM: {res['temporal_flickering']['mean_ssim']:.3f} | Variance: {res['temporal_flickering']['variance']:.3f}")
        print(f"   Verdict:   {res['temporal_flickering']['verdict']}")
    else: print("   Failed to extract SSIM.")
    
    print("--------------------------------------------------")
    print("2. BLINK DETECTION (EAR)")
    if 'blinks' in res:
        print(f"   Blink Count: {res['blinks']['count']} | Average EAR: {res['blinks']['avg_ear']:.3f}")
        print(f"   Verdict:     {res['blinks']['verdict']}")
    else: print("   Failed to extract Blinks.")
        
    print("--------------------------------------------------")
    print("3. LIP SYNC CORRELATION (MAR vs AUDIO)")
    if 'lip_sync' in res:
        print(f"   Correlation/Verdict: {res['lip_sync']['verdict']}")
    else: print("   Failed to extract Lip Sync.")
        
    print("--------------------------------------------------")
    print("4. rPPG HEARTBEAT (Blood Flow)")
    if 'rppg' in res:
        print(f"   Estimated BPM: {res['rppg']['bpm_estimate']:.1f}")
        print(f"   Verdict:       {res['rppg']['verdict']}")
    else: print("   Failed to extract rPPG.")
    print("--------------------------------------------------")
    
    print(f"\nAggregate Sequence Score: {res.get('aggregate_video_score', 0):.2f}")
    
    os.remove(final_file)
