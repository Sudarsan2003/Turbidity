"""
particle_id.py
Fig 2's caption states the blur matrix and scattering angles are used
"to identify given particle & concentration." blur_matrix.py and
scattering.py implement everything up to the scattering-angle map; this
module implements the final step -- turning that map into a particle-type
label and a concentration estimate.

IMPORTANT (same caveat as calibration.py): this is a software-only
PLACEHOLDER pipeline. The "reference signatures" library is built by
simulating synthetic Gaussian blur at different radii as a stand-in for
different particle types/sizes -- it is not derived from real particle
suspensions. Particle-type labels and concentration bands are illustrative
of the *architecture* (feature extraction -> nearest-signature match ->
banding), not physically validated numbers. Once real fluid + particle
standards exist, rebuild the reference library from real captured data
(see build_reference_library()'s docstring).

No ML dependency is introduced -- classification is a simple nearest-
centroid match in normalized feature space, consistent with the project's
lightweight-dependency approach (numpy + opencv only).
"""

import numpy as np

from patterns import generate_pattern
from utils import simulate_blur
from blur import to_gray, blur_score
from blur_matrix import compute_blur_matrix, to_blur_ratio
from scattering import blur_ratio_to_angle

# Placeholder particle classes. Each is defined by a representative
# blur-radius range (px) that stands in for "how much this particle
# type/size forward-scatters light onto the imager" (Fig 1a/1c), plus a
# concentration_gain used only for the concentration-banding placeholder
# below. Replace radius_range with real measured values once particle
# size standards are run through the physical rig.
PARTICLE_CLASSES = {
    "fine_colloidal":  {"radius_range": (1, 3),  "concentration_gain": 0.6},
    "medium_silt":     {"radius_range": (4, 7),  "concentration_gain": 1.0},
    "coarse_sediment": {"radius_range": (8, 13), "concentration_gain": 1.6},
}

CONCENTRATION_BANDS = ["low", "medium", "high"]


def extract_features(matrix, ratio, angle, fft_score=None):
    """
    Compact feature vector describing one blur-matrix result. Used both to
    build reference signatures (build_reference_library) and to classify
    a new sample (identify).
    """
    feats = [
        float(ratio.mean()),                              # overall blur level
        float(ratio.std()),                                # spatial non-uniformity
        float(angle.mean()),                               # mean scattering angle
        float(angle.std()),                                # angle spread
        float(matrix.std() / (matrix.mean() + 1e-6)),       # relative spatial contrast
    ]
    if fft_score is not None:
        feats.append(float(fft_score))                     # frequency-domain sharpness
    return np.array(feats, dtype=np.float64)


def _signature_for_radius(reference_img, radius_px, grid_cols=12, method="laplacian"):
    """Simulate one blur radius and extract its feature vector."""
    blurred = simulate_blur(reference_img, radius_px, noise_pct=0)
    matrix = compute_blur_matrix(blurred, grid_cols=grid_cols, method=method)
    ratio = to_blur_ratio(matrix)
    angle = blur_ratio_to_angle(ratio)
    fft_score = blur_score(to_gray(blurred), "fft")
    return extract_features(matrix, ratio, angle, fft_score)


def build_reference_library(pattern_type="checker", w=480, h=360, feat=24,
                             grid_cols=12, method="laplacian", samples_per_class=4):
    """
    "Train" the nearest-signature classifier by simulating each placeholder
    particle class across its blur-radius range and averaging the resulting
    feature vectors into one centroid signature per class.

    Entirely software (no hardware needed to run this), but the classes
    and radii themselves are placeholders -- see module docstring. To move
    to real data: replace the simulate_blur() call in
    _signature_for_radius() with real captured frames for each known
    particle standard, keeping extract_features() the same.
    """
    reference_img = generate_pattern(pattern_type, w, h, feat)
    library = {}
    for name, cfg in PARTICLE_CLASSES.items():
        lo, hi = cfg["radius_range"]
        radii = np.linspace(lo, hi, samples_per_class)
        feats = [_signature_for_radius(reference_img, r, grid_cols, method) for r in radii]
        library[name] = {
            "centroid": np.mean(feats, axis=0),
            "spread": np.std(feats, axis=0) + 1e-6,  # avoid div-by-zero; floors sensitivity
            "radius_range": (lo, hi),
        }
    return library


def classify_particle(features, library):
    """
    Nearest-centroid classification: z-score the feature vector against
    each class's (centroid, spread), take the Euclidean norm, and pick the
    class with the smallest distance. Returns (best_label, distances_dict)
    so the UI/report can show confidence relative to the runner-up class.
    """
    distances = {}
    for name, sig in library.items():
        z = (features - sig["centroid"]) / sig["spread"]
        distances[name] = float(np.sqrt(np.sum(z ** 2)))
    best = min(distances, key=distances.get)
    return best, distances


def estimate_concentration(mean_blur_ratio, particle_label, library):
    """
    Placeholder concentration banding: scales mean blur ratio by the
    matched class's concentration_gain and buckets into low/medium/high.
    Needs real calibration against known-concentration samples (same
    caveat as CalibrationModel.fit()) before use on physical data.
    """
    gain = PARTICLE_CLASSES.get(particle_label, {}).get("concentration_gain", 1.0)
    score = mean_blur_ratio * gain
    if score < 0.33:
        band = "low"
    elif score < 0.66:
        band = "medium"
    else:
        band = "high"
    return band, float(score)


def identify(img, library, grid_cols=12, method="laplacian"):
    """
    End-to-end entry point: image -> blur matrix -> features -> particle
    type + concentration band. This is what app.py / other callers use.
    """
    matrix = compute_blur_matrix(img, grid_cols=grid_cols, method=method)
    ratio = to_blur_ratio(matrix)
    angle = blur_ratio_to_angle(ratio)
    fft_score = blur_score(to_gray(img), "fft")
    features = extract_features(matrix, ratio, angle, fft_score)

    label, distances = classify_particle(features, library)
    conc_band, conc_score = estimate_concentration(float(ratio.mean()), label, library)

    return {
        "particle_type": label,
        "distances": distances,
        "concentration_band": conc_band,
        "concentration_score": conc_score,
    }


if __name__ == "__main__":
    import sys
    from utils import load_image

    print("Building placeholder particle-signature library...")
    lib = build_reference_library()
    for name, sig in lib.items():
        print(f"  {name:16s} radius_range={sig['radius_range']}  "
              f"centroid={np.round(sig['centroid'], 3)}")

    path = sys.argv[1] if len(sys.argv) > 1 else "assets/checkerboard.png"
    img = load_image(path)
    result = identify(img, lib)

    print(f"\n{path}")
    print(f"  particle_type      : {result['particle_type']}")
    print(f"  concentration_band : {result['concentration_band']} "
          f"(score={result['concentration_score']:.3f})")
    print(f"  distances (lower=better match):")
    for name, d in sorted(result["distances"].items(), key=lambda kv: kv[1]):
        print(f"    {name:16s} {d:.3f}")