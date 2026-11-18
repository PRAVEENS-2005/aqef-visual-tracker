# AQEF — Adaptive Quad-Sensor Energy Field Tracker

> *Most trackers follow a target. AQEF hunts it back.*

A novel single-object tracker built from first principles — combining four independent sensing modalities into one real-time arbitration system. Recovers from **complete target loss** where standard OpenCV trackers fail permanently.

---

## The Problem This Solves

Standard correlation trackers (MOSSE, CSRT, KCF) are one-shot systems. Once the target is occluded or leaves frame — they drift, hallucinate, or die. They have no recovery mechanism.

AQEF solves this with a **quad-sensor pipeline** inspired by sensor fusion in robotics — where a lower-confidence signal corrects a higher-level estimator when the primary fails. Same principle. Different domain.

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                           AQEF PIPELINE                              │
│                                                                      │
│  Frame ──► Feature Extraction (Log Compress + Sobel)                 │
│                      │                                               │
│                      ▼                                               │
│         ┌────────────────────────┐                                   │
│         │   FFT Phase Correlator │ ◄── Pristine + Rolling Template   │
│         └────────────┬───────────┘                                   │
│                      │  PSR Score                                    │
│                      ▼                                               │
│         ┌────────────────────────┐                                   │
│         │  Continuous Arbitration│  gamma = σ(PSR - τ)               │
│         │  Model  (gamma gate)   │                                   │
│         └────────────┬───────────┘                                   │
│                      │                                               │
│                      ▼                                               │
│         ┌────────────────────────┐                                   │
│         │  Topological Gate (TDA)│  distractor reject (forces γ=0.1) │
│         └──────┬─────────┬───────┘                                   │
│                │         │                                           │
│          γ ≥ 0.4       γ < 0.4                                       │
│                │         │                                           │
│         FFT LOCKED    ORB-KF FUSION                                  │
│         CoM Gravity   (median keypoint → KF correct)                 │
│         Shape Morph       │                                          │
│                │       BLIND COAST                                   │
│                │       (Kalman momentum + friction)                  │
│                │         │                                          │
│                └────┬────┘                                           │
│                     ▼                                                │
│                OUTPUT bbox                                           │
└──────────────────────────────────────────────────────────────────────┘
The Four SensorsLayerMethodRoleFFT Phase CorrelatorDFT cross-power spectrum + PSR confidence   Primary tracker — fast, frequency-domain lock   ORB-KF FusionORB keypoints → median location → Kalman correct   Re-acquisition after target loss   4-State Kinematic Kalman FilterConstant-velocity state model $(x, y, v_x, v_y)$   Physics prediction + blind coasting   Topological Data AnalysisConnected component homology proxy   Distractor rejection when tracker merges onto wrong object   The Novel Re-Acquisition MethodWhen the FFT correlator loses confidence (PSR drops below dynamic threshold $\tau$), the system triggers ORB global search:   Detect ORB keypoints across the entire frame   Match against stored target descriptor   Take median of matched keypoint locations — not centroid, not RANSAC homography   Feed that median point directly into the Kalman Filter as a noisy measurement   FFT re-seeds from the corrected Kalman position   Why median? It is robust to outlier matches. A handful of wrong matches don't corrupt the re-acquisition — the median suppresses them naturally.   This is structurally identical to how a robotics state estimator fuses GPS (low frequency, noisy) with IMU (high frequency, drifts) — the lower-confidence signal corrects the higher-level estimator when the primary fails. Two different domains. Same underlying principle. Derived independently.   Key Engineering DecisionsLogarithmic Image CompressionPythongray = (np.log1p(gray * 5.0) / np.log1p(5.0)).astype(np.float32)
Suppresses streetlight glare and high-contrast distractors before feature extraction. Called "Digital Sunglasses" in the codebase.   Subpixel Parabolic Peak Fitting
1D parabola fit on the FFT response surface — eliminates bounding box jitter at sub-pixel precision without any smoothing filter.   Dynamic Tau (Desperation Fix)Pythondynamic_tau = max(2.0, self.base_tau - (coast_frames * 0.1))
As coasting duration increases, the acceptance threshold lowers — the tracker becomes increasingly willing to re-lock on a weaker signal rather than coast forever.   2D Sparse Morphing Space
Tests 3–5 aspect ratio variants per frame. Handles target deformation (skateboarder crouching, standing, spinning) without unbounded drift — geometric anti-spaghetti clamps prevent runaway morphing.   Cubic-Gated Template LearningPythondynamic_lr = self.learning_rate * (gamma ** 3)
Learning rate is zero when confidence is low (avoids learning distractors), accelerates cubically when confidence is high. Only updates the rolling template — pristine template is frozen at init.   PerformanceTested on real skateboarder video footage (outdoor, variable lighting, partial and full occlusion):   ScenarioStandard CSRTAQEFPartial occlusionDrifts to background   Holds lock via CoM gravity   Complete occlusion (>10 frames)Permanently lost   Re-acquires via ORB-KF   High-contrast distractor enters frameMerges onto distractor   TDA surgery rejects distractor   Target changes aspect ratioLoses lock   Morphing space adapts   Streetlight / glarePSR collapses   Log compression suppresses   UsageBashpip install -r requirements.txt
python demo.py --source path/to/video.mp4
Pass --source path/to/video.mp4 (or --source 0 for live webcam)Click and drag to draw bounding box around target   Press ENTER to begin tracking   Press Q or ESC to quit   Live display shows:Bounding box (colour = tracker state)   Status: FFT LOCKED / ORB EKF FUSED / BLIND COASTING / TDA SURGERY   Gamma value and coast frame count   Phase-field heatmap (top-right corner) — real-time FFT response visualisation   Tracker State ColoursColourStateMeaning🟢 GreenFFT LOCKEDHigh confidence, primary tracker active   🟣 MagentaORB EKF FUSEDRe-acquiring via keypoint fusion   🟠 OrangeBLIND COASTINGKalman momentum, waiting for signal   —TDA SURGERYDistractor rejected, confidence forced low   Related WorkThis tracker draws architectural inspiration from[cite: 3]:MOSSE (Bolme et al., 2010) — FFT correlation filter baseline[cite: 3]ECO (Danelljan et al., 2017) — discriminative correlation with sample memory[cite: 3]Kalman & Multi-Sensor Fusion in robotics (Thrun, Burgard, Fox) — the structural parallel for re-acquisition[cite: 3]Topological Data Analysis (Carlsson, 2009) — homology proxy for distractor detection[cite: 3]AQEF combines elements from all four in a single real-time system — without requiring any neural network or GPU[cite: 3].AuthorPraveen S — Mechatronics Engineering, Bannari Amman Institute of Technology   Built as part of an autonomous drone perception and target tracking project.LinkedIn · GitHub   "Most engineers solve the problem in front of them. This tracker hunts the problem down."[cite: 3]
EOF
Use the **Copy code** button at the top right of the box, paste it into your terminal, and push with `git add README.md && git commit -m "docs: refine architecture flow and links" && git push`!
