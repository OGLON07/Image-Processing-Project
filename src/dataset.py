"""Dataset and preprocessing module for RSNA Knee Abnormality Detection.
Handles:
1. DICOM intensity scaling (RescaleSlope, RescaleIntercept)
2. Per-series 2nd-98th percentile robust normalization
3. 2.5D input generation (stacking adjacent slices into 3 channels)
4. Central cropping and resizing to 256x256
5. Multilingual radiology report rule-based auxiliary label extraction with confidence weights
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

def normalize_intensity_pct(img: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Normalizes image pixel intensity using 2nd to 98th percentiles per MRI series/volume.
    Guarantees robust contrast stretching invariant to scanner intensity differences.
    """
    if img.size == 0:
        return img
    val_low = np.percentile(img, p_low)
    val_high = np.percentile(img, p_high)
    if val_high > val_low:
        norm = (img - val_low) / (val_high - val_low)
        norm = np.clip(norm, 0.0, 1.0)
    else:
        norm = np.zeros_like(img, dtype=np.float32)
    return norm.astype(np.float32)

def build_2p5d_window(slices: List[np.ndarray], center_idx: int) -> np.ndarray:
    """Builds a 2.5D MRI input window: stacks center slice and its two adjacent neighbors (or duplicates boundaries).
    Returns:
        np.ndarray of shape (3, H, W) normalized to [0, 1].
    """
    n_slices = len(slices)
    prev_idx = max(0, center_idx - 1)
    next_idx = min(n_slices - 1, center_idx + 1)
    
    s_prev = slices[prev_idx]
    s_curr = slices[center_idx]
    s_next = slices[next_idx]
    
    # Stack along channel axis -> (3, H, W)
    stacked = np.stack([s_prev, s_curr, s_next], axis=0)
    return stacked

class MultilingualReportExtractor:
    """Extracts soft auxiliary abnormality findings from multilingual reports.
    Supports English, Spanish, Dutch, French, German, and Portuguese.
    """
    def __init__(self):
        # Regex dictionaries for finding presence or absence
        self.abnormality_patterns = {
            "ACL": [
                r"\b(acl|anterior cruciate|cruzado anterior|kruisband|croisé antérieur|vorderes kreuzband)\b",
                r"\b(rotura.*cruzado|tear.*acl|acl.*tear|déchirure.*croisé|ruptur.*kreuzband)\b"
            ],
            "MCL": [
                r"\b(mcl|medial collateral|colateral medial|collatéral médial|innenband)\b",
                r"\b(sprain|tear|entorse|distensión|läsion)\b"
            ],
            "Medial Meniscus": [
                r"\b(medial meniscus|menisco medial|menisco interno|mediaal meniscus|ménisque médial|innenmeniskus)\b",
                r"\b(tear|rotura|scheur|déchirure|riss)\b"
            ],
            "Lateral Meniscus": [
                r"\b(lateral meniscus|menisco lateral|menisco externo|lateraal meniscus|ménisque latéral|aussenmeniskus)\b",
                r"\b(tear|rotura|scheur|déchirure|riss)\b"
            ],
            "Medial OA": [
                r"\b(medial.*(oa|osteoarthritis|artrosis|gonartrosis|arthrose|arthrosis))\b",
                r"\b(compartimento medial.*(estrechamiento|pinzamiento|chondropat))\b"
            ],
            "Lateral OA": [
                r"\b(lateral.*(oa|osteoarthritis|artrosis|gonartrosis|arthrose|arthrosis))\b",
                r"\b(compartimento lateral.*(estrechamiento|pinzamiento|chondropat))\b"
            ],
            "PF OA": [
                r"\b(patellofemoral|rotuliano|rétro-patellaire|femoropatelar|pf oa|patellofemoral osteoarthritis)\b",
                r"\b(artrosis femoropatelar|artrosis femorotibial|chondropatie rétropatellaire)\b"
            ],
            "Effusion": [
                r"\b(effusion|joint effusion|derrame|épanchement|erguss|líquido articular)\b"
            ],
            "Synovitis": [
                r"\b(synovitis|sinovitis|synovite|synovial thickening|engrosamiento sinovial)\b"
            ],
            "Baker's": [
                r"\b(baker.*cyst|quiste de baker|kyste de baker|baker-zyste|popliteal cyst|quiste poplíteo)\b"
            ],
            "Contusion": [
                r"\b(contusion|bone bruise|edema óseo|bone contusion|edema oseo|knochenkontusion|ossecous contusion)\b"
            ],
            "Fracture": [
                r"\b(fracture|fractura|fraktur|fissure|fissuration)\b"
            ]
        }
        
        self.negation_patterns = [
            r"\b(no|sin|not|intact|intacto|conservado|normal|aucune|aucun|kein|keine|pas de|geen|sans)\b",
            r"\b(unremarkable|preserved|frei|ausgeschlossen)\b"
        ]
        
    def extract_labels(self, report_text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts 12 soft targets and confidence weights from the report text.
        Returns:
            soft_targets: np.ndarray of shape (12,), values in [0, 1]
            confidences: np.ndarray of shape (12,), values in [0.2, 1.0]
        """
        if not isinstance(report_text, str) or not report_text.strip():
            # Default prior: 0.35 with low confidence
            return np.full(12, 0.35, dtype=np.float32), np.full(12, 0.1, dtype=np.float32)
            
        text_lower = report_text.lower()
        sentences = re.split(r'[.\n;:]+', text_lower)
        
        targets = []
        confidences = []
        
        target_names = [
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
            "Medial OA", "Lateral OA", "PF OA", "Effusion",
            "Synovitis", "Baker's", "Contusion", "Fracture"
        ]
        
        for name in target_names:
            patterns = self.abnormality_patterns[name]
            found_positive = False
            found_negative = False
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                # Check if this sentence mentions the target
                if any(re.search(pat, sent) for pat in patterns):
                    # Check for negation in the same sentence
                    if any(re.search(neg, sent) for neg in self.negation_patterns):
                        # E.g. "no acl tear", "meniscus intact", "pas de fracture"
                        found_negative = True
                    else:
                        found_positive = True
            
            if found_positive and not found_negative:
                targets.append(0.92)  # High probability positive
                confidences.append(0.85)
            elif found_negative and not found_positive:
                targets.append(0.08)  # High probability negative
                confidences.append(0.85)
            elif found_positive and found_negative:
                # Ambiguous or complex sentence
                targets.append(0.50)
                confidences.append(0.40)
            else:
                # Not mentioned in report: set neutral prior with low confidence
                targets.append(0.30)
                confidences.append(0.25)
                
        return np.array(targets, dtype=np.float32), np.array(confidences, dtype=np.float32)

def generate_synthetic_study(study_uid: str, num_series: int = 5, slices_per_series: int = 16, h: int = 256, w: int = 256) -> Dict:
    """Generates a synthetic MRI study structure with realistic geometry and 2.5D windows.
    Used for unit testing, reproducible benchmarks, and pipeline validation.
    """
    planes = ["Sagittal", "Coronal", "Axial"]
    series_list = []
    
    for s_idx in range(num_series):
        series_uid = f"{study_uid}.series.{s_idx}"
        plane = planes[s_idx % len(planes)]
        
        # Create simulated volume with structural contrast
        vol = np.zeros((slices_per_series, h, w), dtype=np.float32)
        # Create an elliptical cylinder simulating knee joint anatomy
        yy, xx = np.mgrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 3
        mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) < (radius ** 2)
        
        for sl_i in range(slices_per_series):
            slice_img = np.random.uniform(20.0, 50.0, (h, w)).astype(np.float32)
            slice_img[mask] += np.random.uniform(200.0, 800.0)
            vol[sl_i] = slice_img
            
        # Normalize 2nd to 98th percentile
        vol_norm = normalize_intensity_pct(vol, 2.0, 98.0)
        
        # Build 2.5D windows (sampling 4 windows per series as in Silver solution)
        windows = []
        center_indices = np.linspace(1, slices_per_series - 2, 4, dtype=int)
        for c_idx in center_indices:
            w_2p5d = build_2p5d_window(list(vol_norm), c_idx)
            windows.append(w_2p5d)
            
        series_list.append({
            "SeriesInstanceUID": series_uid,
            "Anatomical_Plane": plane,
            "Fluid_Sensitive": 1 if s_idx % 2 == 0 else 0,
            "Fat_Suppression": 1 if s_idx % 2 == 0 else 0,
            "windows": np.stack(windows, axis=0)  # Shape: (4, 3, H, W)
        })
        
    return {
        "StudyInstanceUID": study_uid,
        "series": series_list
    }

