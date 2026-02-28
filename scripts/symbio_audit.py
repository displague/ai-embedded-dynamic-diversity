import sys
import re
from pathlib import Path

def audit_symbiogenesis_log(log_path: str):
    if not Path(log_path).exists():
        print(f"Error: {log_path} not found.")
        return

    data = Path(log_path).read_text(encoding="utf-8")
    
    variances = [float(m) for m in re.findall(r"'mean_emergent_signal_loss':\s*(-?\d+\.\d+)", data)]
    detections = [float(m) for m in re.findall(r"'mean_detection_loss':\s*(\d+\.\d+)", data)]
    
    if not variances or not detections:
        print(f"[{Path(log_path).name}] No metrics found.")
        return
        
    v_gain = (variances[-1] / variances[0]) - 1.0 if variances[0] != 0 else 0
    d_imp = (detections[0] - detections[-1]) / detections[0] if detections[0] != 0 else 0
    
    print(f"[{Path(log_path).name}]")
    print(f"Generations analyzed: {len(variances)}")
    print(f"Emergent Signal Variance Gain: {v_gain*100:.2f}%")
    print(f"Signal Detection Improvement: {d_imp*100:.2f}%")
    
    if v_gain > 0.5 and d_imp > 0.15:
        print("Conclusion: Strong evidence of emergent symbiogenesis.\n")
    else:
        print("Conclusion: Weak evidence of symbiogenesis.\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audit_symbiogenesis_log(sys.argv[1])
    else:
        print("Usage: python symbio_audit.py <path_to_log>")
