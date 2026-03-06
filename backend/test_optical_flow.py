import cv2
import numpy as np
import os
from forensic_engine import ForensicEngine

def create_rigid_video(filename, fps=30, frames=90):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (320, 240))
    x_pos = 10
    for i in range(frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Draw a white rectangle moving predictably to the right
        cv2.rectangle(frame, (int(x_pos), 100), (int(x_pos)+50, 150), (255, 255, 255), -1)
        x_pos += 3.0 # Move 3 pixels per frame
        out.write(frame)
    out.release()
    print(f"Created {filename}")

def create_morphing_video(filename, fps=30, frames=90):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (320, 240))
    for i in range(frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Create chaotic, melting pixel movement
        # Draw hundreds of tiny rectangles moving in random directions
        for j in range(200):
            rx = int(160 + np.sin(i/10.0 + j) * 50 + np.random.randint(-10, 10))
            ry = int(120 + np.cos(i/10.0 + j*2) * 50 + np.random.randint(-10, 10))
            cv2.rectangle(frame, (rx, ry), (rx+5, ry+5), (255, 255, 255), -1)
        out.write(frame)
    out.release()
    print(f"Created {filename}")

if __name__ == "__main__":
    engine = ForensicEngine()
    
    rigid_file = "test_rigid.mp4"
    morph_file = "test_morph.mp4"
    
    create_rigid_video(rigid_file)
    create_morphing_video(morph_file)
    
    print("\n--- Testing Rigid Video ---")
    res1 = engine.analyze_video(rigid_file)
    print("Optical Flow Verdict:", res1.get('optical_flow', {}).get('verdict'))
    print("Angle Variance Peak:", res1.get('optical_flow', {}).get('angle_variance_peak'))
    print("Aggregate Score:", res1.get('aggregate_video_score'))
    
    print("\n--- Testing Morphing/AI Video ---")
    res2 = engine.analyze_video(morph_file)
    print("Optical Flow Verdict:", res2.get('optical_flow', {}).get('verdict'))
    print("Angle Variance Peak:", res2.get('optical_flow', {}).get('angle_variance_peak'))
    print("Aggregate Score:", res2.get('aggregate_video_score'))
    
    os.remove(rigid_file)
    os.remove(morph_file)
