import os
import io
import sys
from verify_all_temporal import create_advanced_dummy_video
from forensic_engine import ForensicEngine

def run_tests_to_file():
    test_file = "temp_av.mp4"
    final_file = create_advanced_dummy_video(test_file)
    engine = ForensicEngine()
    
    output = []
    output.append("# Temporal Module Verification Results\n")
    output.append("Tested the `analyze_video()` function on a dynamically generated 6-second audio/video file.\n")
    
    res = engine.analyze_video(final_file)
    
    output.append("### 1. SSIM Flickering")
    if 'temporal_flickering' in res:
        output.append(f"- Mean SSIM: {res['temporal_flickering']['mean_ssim']:.3f} | Variance: {res['temporal_flickering']['variance']:.3f}")
        output.append(f"- Verdict: {res['temporal_flickering']['verdict']}\n")
    else: output.append("- Failed to extract SSIM.\n")
    
    output.append("### 2. Blink Detection (EAR)")
    if 'blinks' in res:
        output.append(f"- Blink Count: {res['blinks']['count']} | Average EAR: {res['blinks']['avg_ear']:.3f}")
        output.append(f"- Verdict: {res['blinks']['verdict']}\n")
    else: output.append("- Failed to extract Blinks.\n")
        
    output.append("### 3. Lip Sync Correlation (MAR vs Audio)")
    if 'lip_sync' in res:
        output.append(f"- Correlation Status: {res['lip_sync']['verdict']}\n")
    else: output.append("- Failed to extract Lip Sync.\n")
        
    output.append("### 4. rPPG Heartbeat (Blood Flow)")
    if 'rppg' in res:
        output.append(f"- Estimated BPM: {res['rppg']['bpm_estimate']:.1f}")
        output.append(f"- Verdict: {res['rppg']['verdict']}\n")
    else: output.append("- Failed to extract rPPG.\n")
    
    output.append("### 5. Dense Optical Flow (Morphing)")
    if 'optical_flow' in res:
        output.append(f"- Peak Angle Variance: {res['optical_flow'].get('angle_variance_peak', 0.0):.3f}")
        output.append(f"- Verdict: {res['optical_flow']['verdict']}\n")
        
    output.append(f"**Aggregate Sequence Score**: {res.get('aggregate_video_score', 0):.2f}\n")
    
    with open(r"c:\Users\Karthik\.gemini\antigravity\brain\59649e46-1017-461e-9517-03d51ac926dd\temporal_verification.md", "w") as f:
        f.write("\n".join(output))
        
    os.remove(final_file)
    print("SAVED TO ARTIFACT")

if __name__ == "__main__":
    run_tests_to_file()
