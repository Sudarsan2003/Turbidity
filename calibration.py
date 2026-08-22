"""
calibration.py
Blur -> turbidity (NTU) and blur-radius -> particle-size mappings.

IMPORTANT: the coefficients here are placeholders (adjustable sliders in
app.py), not a measured calibration curve. Populate CalibrationModel.fit()
with (blur_ratio, NTU) pairs collected from the physical rig (Fig 1a)
against known turbidity standards before trusting these numbers for real
fluid measurements. Until then this module exists so the full software
pipeline can be exercised end-to-end today.
"""

import json
import os
import numpy as np


class CalibrationModel:
    def __init__(self, slope=1.0, intercept=0.0, particle_k=2.5):
        """
        slope, intercept: NTU = slope * (blur_ratio * 100) + intercept
        particle_k: particle_size_um = particle_k * blur_radius_px
        """
        self.slope = slope
        self.intercept = intercept
        self.particle_k = particle_k

    def blur_to_ntu(self, mean_blur_ratio):
        return self.slope * (mean_blur_ratio * 100.0) + self.intercept

    def radius_to_particle_size(self, blur_radius_px):
        return self.particle_k * blur_radius_px

    def fit(self, blur_ratios, ntu_values):
        """
        Least-squares fit of NTU = slope * (blur_ratio*100) + intercept
        from paired real measurements. Call this once real calibration
        data exists from the physical rig.
        """
        x = np.asarray(blur_ratios, dtype=np.float64) * 100.0
        y = np.asarray(ntu_values, dtype=np.float64)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        self.slope, self.intercept = float(slope), float(intercept)
        return self.slope, self.intercept

    def to_dict(self):
        return {"slope": self.slope, "intercept": self.intercept,
                "particle_k": self.particle_k}

    @classmethod
    def from_dict(cls, d):
        return cls(slope=d.get("slope", 1.0),
                    intercept=d.get("intercept", 0.0),
                    particle_k=d.get("particle_k", 2.5))

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


if __name__ == "__main__":
    cal = CalibrationModel(slope=1.0, intercept=0.0, particle_k=2.5)
    print("Placeholder calibration:", cal.to_dict())
    print("Example: blur_ratio=0.4 ->", cal.blur_to_ntu(0.4), "NTU")
    cal.save("results/calibration.json")
    print("Saved results/calibration.json")