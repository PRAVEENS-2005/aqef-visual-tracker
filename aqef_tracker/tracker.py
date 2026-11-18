import cv2
import numpy as np

class UltimateAQEFTracker:
    """
    Adaptive Quad-Sensor Energy Field (AQEF) Tracker
    Combines FFT, ORB, Kalman, and Topological Data Analysis.
    """
    def __init__(self, learning_rate=0.04): 
        # Base Hyperparameters
        self.learning_rate = learning_rate
        self.base_tau = 3.5
        
        # Geometry & Memory
        self.pos = (0.0, 0.0)
        self.base_sz = (0.0, 0.0)
        self.internal_sz = (0, 0)
        self.hann_window = None
        
        self.pristine_fft = None  
        self.rolling_fft = None   
        self.target_snapshot = None 
        
        self.coast_frames = 0.0 
        
        # The ORB Engine (Global Fallback)
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.orb_target_des = None
        
        # The Physics Engine (Extended Kalman Filter)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        
        # Topology Engine (TDA)
        self.target_topology = {'components': 1}

    def _get_features(self, img, pos, sz, scale_w=1.0, scale_h=1.0):
        cx, cy = int(pos[0]), int(pos[1])
        w, h = int(sz[0] * scale_w), int(sz[1] * scale_h)
        
        # Pad image to prevent crashes at screen edges
        x1, y1 = cx - w//2, cy - h//2
        x2, y2 = x1 + w, y1 + h
        pad_t, pad_b = max(0, -y1), max(0, y2 - img.shape[0])
        pad_l, pad_r = max(0, -x1), max(0, x2 - img.shape[1])
        
        if any([pad_t, pad_b, pad_l, pad_r]):
            img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
            
        patch = img[y1+pad_t : y2+pad_t, x1+pad_l : x2+pad_l]
        if patch.shape[1] > 0 and patch.shape[0] > 0:
            patch = cv2.resize(patch, self.internal_sz)
            
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # LOGARITHMIC COMPRESSION (The "Digital Sunglasses" fix for streetlights)
        # We cast back to np.float32 to prevent OpenCV Sobel from crashing
        gray = (np.log1p(gray * 5.0) / np.log1p(5.0)).astype(np.float32)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        
        blended = (gray * 0.5) + (magnitude * 0.5)
        blended = (blended - np.mean(blended)) / (np.std(blended) + 1e-5)
        return blended * self.hann_window

    def _subpixel_peak(self, response, max_loc):
        """1D Parabola fit for zero-jitter precision."""
        x, y = max_loc 
        h, w = response.shape
        if x == 0 or x == w - 1 or y == 0 or y == h - 1:
            return float(x), float(y)

        val_c = response[y, x]
        val_l = response[y, x-1]
        val_r = response[y, x+1]
        val_u = response[y-1, x]
        val_d = response[y+1, x]

        dx = 0.5 * (val_r - val_l) / (2 * val_c - val_l - val_r + 1e-5)
        dy = 0.5 * (val_d - val_u) / (2 * val_c - val_u - val_d + 1e-5)
        return float(x + dx), float(y + dy)

    def _apply_com_gravity(self, img, pos, sz):
        """Wasserstein-inspired Center of Mass pull."""
        cx, cy = int(pos[0]), int(pos[1])
        w, h = int(sz[0]), int(sz[1])
        
        x1, y1 = max(0, cx - w//2), max(0, cy - h//2)
        patch = img[y1:y1+h, x1:x1+w]
        
        if patch.shape[0] == 0 or patch.shape[1] == 0: return pos
            
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        M = cv2.moments(edges)
        
        if M["m00"] > 0:
            drift_x = int(M["m10"] / M["m00"]) - (w / 2.0)
            drift_y = int(M["m01"] / M["m00"]) - (h / 2.0)
            return (pos[0] + (drift_x * 0.15), pos[1] + (drift_y * 0.15))
        return pos

    def _topological_gate(self, features):
        """TDA Proxy: Calculates homology (connected components) to prevent distractor merging."""
        binary = (features > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)

    def init(self, frame, bbox):
        x, y, w, h = bbox
        self.base_sz = [float(w), float(h)] # Use list to allow independent morphing
        self.target_snapshot = frame[y:y+h, x:x+w].copy()
        
        opt_w = cv2.getOptimalDFTSize(int(w))
        opt_h = cv2.getOptimalDFTSize(int(h))
        self.internal_sz = (opt_w, opt_h)
        
        self.pos = (x + w/2.0, y + h/2.0)
        self.hann_window = cv2.createHanningWindow(self.internal_sz, cv2.CV_32F)
        
        self.kalman.statePre = np.array([[self.pos[0]], [self.pos[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[self.pos[0]], [self.pos[1]], [0], [0]], np.float32)
        
        features = self._get_features(frame, self.pos, self.base_sz)
        self.pristine_fft = np.fft.fft2(features)
        self.rolling_fft = self.pristine_fft.copy()
        
        target_gray = cv2.cvtColor(self.target_snapshot, cv2.COLOR_BGR2GRAY)
        _, self.orb_target_des = self.orb.detectAndCompute(target_gray, None)
        
        self.target_topology['components'] = self._topological_gate(features)

    def update(self, frame):
        # 1. PHYSICS PREDICT
        pred = self.kalman.predict()
        pred_pos = (float(pred[0, 0]), float(pred[1, 0]))
        
        status_msg = ""
        fused_template = (0.15 * self.pristine_fft) + (0.85 * self.rolling_fft)
        
        # THE DESPERATION FIX (Dynamic Tau)
        dynamic_tau = max(2.0, self.base_tau - (self.coast_frames * 0.1))
        
        # 2D SPARSE MORPHING SPACE (Wipeout Recovery)
        morph_factors = [
            (1.0, 1.0),   (0.98, 0.98), (1.02, 1.02) # Standard scaling
        ]
        if self.coast_frames > 2:
            morph_factors += [(1.30, 0.70), (0.70, 1.30)] # Horizontal Wipeout / Vertical Standup
            
        best_psr = -1
        best_morph = (1.0, 1.0)
        best_pos = pred_pos
        best_features = None
        best_response = None
        best_dx, best_dy = 0, 0
        
        for scale_w, scale_h in morph_factors:
            features = self._get_features(frame, pred_pos, self.base_sz, scale_w, scale_h)
            search_fft = np.fft.fft2(features)
            
            cross_power = search_fft * np.conj(fused_template)
            response = np.fft.ifft2(cross_power).real
            response = np.fft.fftshift(response)
            
            _, max_val, _, max_loc = cv2.minMaxLoc(response)
            sub_x, sub_y = self._subpixel_peak(response, max_loc)
            
            h, w = response.shape
            mask = np.ones_like(response, dtype=bool)
            y_min, y_max = max(0, max_loc[1]-4), min(h, max_loc[1]+5)
            x_min, x_max = max(0, max_loc[0]-4), min(w, max_loc[0]+5)
            mask[y_min:y_max, x_min:x_max] = False
            
            sidelobes = response[mask]
            psr = (max_val - np.mean(sidelobes)) / (np.std(sidelobes) + 1e-5)
            
            # Mathematical Penalty to heavily discourage blind morphing
            penalty = 1.0 if (scale_w == 1.0 and scale_h == 1.0) else 0.985
            psr *= penalty
            
            if psr > best_psr:
                best_psr = psr
                best_morph = (scale_w, scale_h)
                best_features = features
                best_response = response
                
                actual_w_search, actual_h_search = self.base_sz[0] * scale_w, self.base_sz[1] * scale_h
                best_dx = (sub_x - w/2.0) * (actual_w_search / w)
                best_dy = (sub_y - h/2.0) * (actual_h_search / h)
                best_pos = (pred_pos[0] + best_dx, pred_pos[1] + best_dy)

        # CONTINUOUS ARBITRATION MODEL
        gamma = 1.0 / (1.0 + np.exp(-2.0 * (best_psr - dynamic_tau)))
        
        # TOPOLOGICAL SURGERY
        if gamma > 0.40:
            current_comp = self._topological_gate(best_features)
            if current_comp > self.target_topology['components'] + 1:
                gamma = 0.10 # Surgery Triggered: Force rejection of the distractor
                status_msg = f"TDA SURGERY (Comp: {current_comp})"
                
        jump_dist = np.hypot(best_dx, best_dy)
        max_allowed_jump = max(self.base_sz) * (0.40 + (self.coast_frames * 0.25))
        omega_penalty = 999.0 if jump_dist > max_allowed_jump else 0.0
        
        if omega_penalty > 0: gamma = 0.0 # Kinematic override

        fusion_active = False
        status_color = (0, 0, 0)
        
        # GLOBAL FALLBACK (ORB EKF Fusion)
        if gamma < 0.40 and self.orb_target_des is not None:
            live_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            live_kp, live_des = self.orb.detectAndCompute(live_gray, None)
            
            if live_des is not None:
                matches = self.matcher.match(self.orb_target_des, live_des)
                matches = sorted(matches, key=lambda x: x.distance)
                robust_matches = [m for m in matches[:40] if m.distance < 60]
                
                if len(robust_matches) > 6:
                    live_pts = np.float32([live_kp[m.trainIdx].pt for m in robust_matches])
                    orb_x, orb_y = float(np.median(live_pts[:, 0])), float(np.median(live_pts[:, 1]))
                    
                    self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 50.0 
                    self.kalman.correct(np.array([[np.float32(orb_x)], [np.float32(orb_y)]]))
                    self.pos = (float(self.kalman.statePost[0, 0]), float(self.kalman.statePost[1, 0]))
                    
                    fusion_active = True
                    self.coast_frames = max(0, self.coast_frames - 2) 
                    gamma = 0.50 # Recovering
                    status_msg = f"ORB EKF FUSED ({len(robust_matches)} pts)"
                    status_color = (255, 0, 255) 

        if fusion_active:
            pass
        elif gamma >= 0.40:
            # RAPID COAST RECOVERY
            self.coast_frames = max(0, self.coast_frames - 2.0)
            
            self.pos = self._apply_com_gravity(frame, best_pos, self.base_sz)
            
            # EXPONENTIAL DEFORMATION (Learning the shape)
            if best_morph != (1.0, 1.0) and gamma > 0.60:
                self.base_sz[0] = self.base_sz[0] * (0.85 + 0.15 * best_morph[0])
                self.base_sz[1] = self.base_sz[1] * (0.85 + 0.15 * best_morph[1])
            
            # GEOMETRIC PRIORS (The Anti-Spaghetti Clamp)
            # 1. Aspect Ratio limit (Max 2.5 : 1)
            aspect_ratio = self.base_sz[0] / max(1.0, self.base_sz[1])
            if aspect_ratio > 2.5: self.base_sz[0] = self.base_sz[1] * 2.5
            elif aspect_ratio < 0.4: self.base_sz[1] = self.base_sz[0] * 2.5
            
            # 2. Maximum Screen Limits (60%)
            max_w, max_h = frame.shape[1] * 0.6, frame.shape[0] * 0.6
            self.base_sz[0] = min(self.base_sz[0], max_w)
            self.base_sz[1] = min(self.base_sz[1], max_h)
            
            self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
            self.kalman.correct(np.array([[np.float32(self.pos[0])], [np.float32(self.pos[1])]]))
            
            # SMART LEARNING (Gated by Gamma)
            if gamma > 0.80:
                new_fft = np.fft.fft2(best_features)
                dynamic_lr = self.learning_rate * (gamma ** 3) # Cubic acceleration
                self.rolling_fft = (1 - dynamic_lr) * self.rolling_fft + dynamic_lr * new_fft
            
            if not status_msg:
                status_msg = f"FFT LOCKED | Tau: {dynamic_tau:.1f}"
                status_color = (0, 255, 0)
        else:
            # BLIND COASTING (Momentum persistence for Glare)
            self.coast_frames = min(30.0, self.coast_frames + 1)
            self.pos = pred_pos
            
            # Coast seamlessly for 4 frames, then hit the exponential brakes
            friction = 1.0 if self.coast_frames < 4 else max(0.0, np.exp(-0.8 * (self.coast_frames - 4)))
            self.kalman.statePost[2, 0] *= friction
            self.kalman.statePost[3, 0] *= friction
            
            if not status_msg:
                status_msg = f"BLIND COASTING ({self.coast_frames})"
                status_color = (0, 165, 255)
            
        self.pos = (np.clip(self.pos[0], 0, frame.shape[1]), np.clip(self.pos[1], 0, frame.shape[0]))
            
        # Visualization: Generate the Phase-Field Heatmap for output
        vis_heatmap = None
        if best_response is not None:
            vis_heatmap = cv2.normalize(best_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_heatmap = cv2.applyColorMap(vis_heatmap, cv2.COLORMAP_JET)
        
        out_w, out_h = int(self.base_sz[0]), int(self.base_sz[1])
        out_x, out_y = int(self.pos[0] - out_w / 2.0), int(self.pos[1] - out_h / 2.0)
        
        info_string = f"Gamma: {gamma:.2f} | Coast: {self.coast_frames:.1f}"
        
        return (out_x, out_y, out_w, out_h), status_msg, status_color, info_string, vis_heatmap

def get_custom_roi(frame, window_name="Ultimate AQEF Tracker"):
    drawing = False
    ix, iy = -1, -1
    bbox = (0, 0, 0, 0)
    roi_selected = False
    temp_frame = frame.copy()
    
    def draw_roi(event, x, y, flags, param):
        nonlocal drawing, ix, iy, bbox, roi_selected, temp_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            roi_selected = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi_selected = True
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2)
            bbox = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_roi)
    
    print("\n--- AQEF FUSION INITIALIZED ---")
    print("1. Click and drag to draw a box around the target.")
    print("2. Press ENTER to track.")
    
    while True:
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [13, 32]: 
            if roi_selected and bbox[2] > 1 and bbox[3] > 1: return bbox
        elif key in [27, ord('q')]: return None

if __name__ == "__main__":
    video_source = "/home/praveens/testfile1.webm" # Replace with your video
    cap = cv2.VideoCapture(video_source) 
    ret, frame = cap.read()
    
    if not ret:
        print("[ERROR] Could not read video feed.")
        exit()

    bbox = get_custom_roi(frame)
    if bbox is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    
    tracker = UltimateAQEFTracker()
    tracker.init(frame, bbox)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        box, msg, color, info, heatmap = tracker.update(frame)
        x, y, w, h = box
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display Phase-Field Heatmap in the top right corner
        if heatmap is not None:
            hm_h, hm_w = heatmap.shape[:2]
            frame[10:10+hm_h, frame.shape[1]-hm_w-10:frame.shape[1]-10] = heatmap
        
        cv2.imshow("Ultimate AQEF Tracker", frame)
        if cv2.waitKey(30) & 0xFF in [27, ord('q')]: break
            
    cap.release()
    cv2.destroyAllWindows()
