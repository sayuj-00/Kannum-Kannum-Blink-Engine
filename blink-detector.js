/**
 * blink-detector.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Self-contained blink-detection engine.
 * Uses MediaPipe FaceLandmarker (@mediapipe/tasks-vision) for real-time
 * facial landmark detection and Eye Aspect Ratio (EAR) for blink detection.
 *
 * USAGE
 * ─────
 * import BlinkDetector from './blink-detector.js';
 *
 * const detector = new BlinkDetector({
 *   videoElement:   document.getElementById('webcam'),   // REQUIRED
 *   onBlink:        () => console.log('blink!'),          // fires once per blink
 *   onCountChange:  (n) => updateUI(n),                  // fires every count change
 *   onStatusChange: (msg) => showStatus(msg),            // optional status text
 *   onError:        (err) => showError(err),             // optional error handler
 *   earThreshold:   0.20,    // eyes-closed threshold  (0.15-0.25 works well)
 *   consecutiveFrames: 2,    // min frames below threshold to register "closed"
 * });
 *
 * await detector.startCamera();   // request cam + start detection
 * detector.stopCamera();           // stop cam + detection
 * detector.resetCount();           // reset blinkCount to 0
 *
 * Read:  detector.blinkCount       // current integer count
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * MediaPipe FaceLandmarker produces 478 3-D landmarks (the canonical face mesh).
 * Eye landmarks used for EAR (right/left from the model frame = user view):
 *
 *  Right eye:
 *    Vertical pairs: [159,145], [160,144], [158,153]
 *    Horizontal:      [33, 133]
 *
 *  Left eye:
 *    Vertical pairs: [386,374], [385,380], [387,373]
 *    Horizontal:      [362, 263]
 *
 *  EAR = sum(vertical_dists) / (N_pairs * horizontal_dist)
 *        Adapted from Soukupova & Cech (2016)
 *
 * Blink state machine:
 *   OPEN  --(EAR < threshold for N frames)--> CLOSED
 *   CLOSED--(EAR >= threshold)             --> OPEN + blinkCount++
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ─── MediaPipe CDN constants ──────────────────────────────────────────────────
const MEDIAPIPE_CDN =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const FACE_LANDMARKER_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// ─── Eye landmark index maps (MediaPipe 478-point mesh) ──────────────────────
const RIGHT_EYE = {
  // [top, bottom] vertical pairs
  vertical: [
    [159, 145],
    [160, 144],
    [158, 153],
  ],
  // [left corner, right corner] horizontal
  horizontal: [33, 133],
};

const LEFT_EYE = {
  vertical: [
    [386, 374],
    [385, 380],
    [387, 373],
  ],
  horizontal: [362, 263],
};

// ─── Blink state enum ─────────────────────────────────────────────────────────
const BlinkState = Object.freeze({ OPEN: 'OPEN', CLOSED: 'CLOSED' });

// ─── Utility: Euclidean distance between two landmarks ────────────────────────
function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate Eye Aspect Ratio for one eye.
 * @param {Array} lm   - full landmark array from MediaPipe
 * @param {Object} eye - { vertical: [[top,bot],...], horizontal: [l,r] }
 * @returns {number} EAR value (higher = more open)
 */
function computeEAR(lm, eye) {
  const [h1, h2] = eye.horizontal;
  const horizontal = dist(lm[h1], lm[h2]);
  if (horizontal === 0) return 1; // degenerate guard

  let verticalSum = 0;
  for (const [top, bot] of eye.vertical) {
    verticalSum += dist(lm[top], lm[bot]);
  }

  return verticalSum / (eye.vertical.length * horizontal);
}

// ─── BlinkDetector class ──────────────────────────────────────────────────────
export default class BlinkDetector {
  /**
   * @param {Object} opts
   * @param {HTMLVideoElement} opts.videoElement       - video el for the stream
   * @param {Function}        [opts.onBlink]           - called once per blink (count)
   * @param {Function}        [opts.onCountChange]     - called with new count
   * @param {Function}        [opts.onStatusChange]    - called with status string
   * @param {Function}        [opts.onError]           - called with Error object
   * @param {number}          [opts.earThreshold]      - EAR below = closed (default 0.20)
   * @param {number}          [opts.consecutiveFrames] - frames below threshold needed (default 2)
   */
  constructor(opts = {}) {
    if (!opts.videoElement) {
      throw new Error('[BlinkDetector] opts.videoElement is required.');
    }

    // ── Public config ──────────────────────────────────────────────────────
    this.videoElement      = opts.videoElement;
    this.onBlink           = opts.onBlink           || null;
    this.onCountChange     = opts.onCountChange     || null;
    this.onStatusChange    = opts.onStatusChange    || null;
    this.onError           = opts.onError           || null;
    this.earThreshold      = opts.earThreshold      ?? 0.20;
    this.consecutiveFrames = opts.consecutiveFrames ?? 2;

    // ── Public state ───────────────────────────────────────────────────────
    this.blinkCount = 0;
    this.isRunning  = false;

    // ── Private state ──────────────────────────────────────────────────────
    this._faceLandmarker   = null;
    this._stream           = null;
    this._animFrameId      = null;
    this._blinkState       = BlinkState.OPEN;
    this._closedFrameCount = 0;   // consecutive frames with EAR < threshold
    this._lastVideoTime    = -1;  // guards against processing the same frame twice

    // ── Blink-speed tracking ───────────────────────────────────────────────
    // Used to distinguish a slow single blink (+2) from a fast double blink (+1 pair).
    this._lastBlinkTime    = 0;     // timestamp (ms) of most recent blink completion
    this._lastBlinkWasFast = false; // whether the last blink was already the 2nd of a fast pair
  }

  // ──────────────────────────────────────────────────────────────────────────
  // PUBLIC API
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Request camera permission, start stream, load MediaPipe, begin detection.
   * Resolves when detection loop has started.
   */
  async startCamera() {
    if (this.isRunning) return;

    try {
      this._emitStatus('Requesting camera permission...');
      await this._initCamera();
      this._emitStatus('Loading MediaPipe model...');
      await this._initMediaPipe();
      this._emitStatus('Detecting blinks...');
      this.isRunning = true;
      this._startDetectionLoop();
    } catch (err) {
      this._handleError(err);
      throw err;
    }
  }

  /**
   * Stop all tracks, cancel animation frame, release MediaPipe resources.
   * Preserves blinkCount.
   */
  stopCamera() {
    this.isRunning = false;

    // Cancel detection loop
    if (this._animFrameId !== null) {
      cancelAnimationFrame(this._animFrameId);
      this._animFrameId = null;
    }

    // Stop camera tracks
    if (this._stream) {
      this._stream.getTracks().forEach((t) => t.stop());
      this._stream = null;
    }

    // Clear video source
    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }

    // Release MediaPipe resources
    if (this._faceLandmarker) {
      try { this._faceLandmarker.close(); } catch (_) {}
      this._faceLandmarker = null;
    }

    // Reset per-frame state (but NOT blinkCount)
    this._blinkState       = BlinkState.OPEN;
    this._closedFrameCount = 0;
    this._lastVideoTime    = -1;
    this._lastBlinkTime    = 0;
    this._lastBlinkWasFast = false;

    this._emitStatus('Camera stopped.');
  }

  /**
   * Reset blinkCount to 0 and fire onCountChange callback.
   */
  resetCount() {
    this.blinkCount      = 0;
    this._lastBlinkTime  = 0;
    this._lastBlinkWasFast = false;
    this._emitCountChange();
    this._emitStatus('Counter reset.');
  }

  // ──────────────────────────────────────────────────────────────────────────
  // PRIVATE - Initialisation
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Request getUserMedia and attach stream to videoElement.
   */
  async _initCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        'Camera API (getUserMedia) is not supported in this browser.'
      );
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width:      { ideal: 640 },
          height:     { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });
    } catch (err) {
      if (
        err.name === 'NotAllowedError' ||
        err.name === 'PermissionDeniedError'
      ) {
        throw new Error(
          'Camera permission is required for blink detection.'
        );
      }
      if (
        err.name === 'NotFoundError' ||
        err.name === 'DevicesNotFoundError'
      ) {
        throw new Error('No webcam detected.');
      }
      throw err;
    }

    this._stream = stream;
    this.videoElement.srcObject = stream;
    this.videoElement.setAttribute('playsinline', '');

    // Wait for metadata before playing
    await new Promise((resolve, reject) => {
      this.videoElement.onloadedmetadata = resolve;
      this.videoElement.onerror = () =>
        reject(new Error('Video element failed to load stream.'));
    });

    await this.videoElement.play();
  }

  /**
   * Dynamically import FaceLandmarker from CDN and create the task instance.
   */
  async _initMediaPipe() {
    let FaceLandmarker, FilesetResolver;

    try {
      const vision = await import(
        /* webpackIgnore: true */
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
      );
      FaceLandmarker  = vision.FaceLandmarker;
      FilesetResolver = vision.FilesetResolver;
    } catch (importErr) {
      throw new Error(
        'Failed to load MediaPipe. Check your internet connection. ' +
          importErr.message
      );
    }

    const filesetResolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_CDN);

    this._faceLandmarker = await FaceLandmarker.createFromOptions(
      filesetResolver,
      {
        baseOptions: {
          modelAssetPath: FACE_LANDMARKER_MODEL,
          delegate:       'GPU',  // falls back to CPU automatically
        },
        runningMode:                       'VIDEO',
        numFaces:                          1,
        minFaceDetectionConfidence:        0.5,
        minFacePresenceConfidence:         0.5,
        minTrackingConfidence:             0.5,
        outputFaceBlendshapes:             false,
        outputFacialTransformationMatrices: false,
      }
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // PRIVATE - Detection loop
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * rAF-based detection loop. Runs until stopCamera() is called.
   */
  _startDetectionLoop() {
    const loop = (nowMs) => {
      if (!this.isRunning) return;
      this._processFrame(nowMs);
      this._animFrameId = requestAnimationFrame(loop);
    };
    this._animFrameId = requestAnimationFrame(loop);
  }

  /**
   * Process a single video frame through MediaPipe and update blink state.
   * @param {number} nowMs - timestamp from rAF (milliseconds)
   */
  _processFrame(nowMs) {
    const video = this.videoElement;

    // Guard: video must be playing and have data
    if (
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      video.paused ||
      video.ended
    ) {
      return;
    }

    // Guard: skip if this exact frame was already processed
    if (video.currentTime === this._lastVideoTime) return;
    this._lastVideoTime = video.currentTime;

    // Run inference
    let result;
    try {
      result = this._faceLandmarker.detectForVideo(video, nowMs);
    } catch (_) {
      // Skip this frame silently on inference errors
      return;
    }

    // ── 1. Validate exactly one face ─────────────────────────────────────
    if (
      !result ||
      !result.faceLandmarks ||
      result.faceLandmarks.length !== 1
    ) {
      // No face, or multiple faces: reset closed-frame counter, don't count
      this._closedFrameCount = 0;
      return;
    }

    const landmarks = result.faceLandmarks[0]; // 478 normalised points

    // ── 2. Compute average EAR ────────────────────────────────────────────
    const earRight = computeEAR(landmarks, RIGHT_EYE);
    const earLeft  = computeEAR(landmarks, LEFT_EYE);
    const earAvg   = (earRight + earLeft) / 2;

    // ── 3. Blink state machine ────────────────────────────────────────────
    if (earAvg < this.earThreshold) {
      this._closedFrameCount++;

      if (
        this._blinkState === BlinkState.OPEN &&
        this._closedFrameCount >= this.consecutiveFrames
      ) {
        // Confirmed eye closure: transition OPEN -> CLOSED
        this._blinkState = BlinkState.CLOSED;
      }
    } else {
      // Eyes open this frame
      if (this._blinkState === BlinkState.CLOSED) {
        // Complete cycle OPEN->CLOSED->OPEN: register one blink
        this._registerBlink();
      }
      this._blinkState       = BlinkState.OPEN;
      this._closedFrameCount = 0;
    }
  }

  // ──────────────────────────────────────────────────────────────────────────
  // PRIVATE - Blink registration & callbacks
  // ──────────────────────────────────────────────────────────────────────────

  _registerBlink() {
    // ── Slow-vs-fast blink logic ──────────────────────────────────────────
    //
    //  • Slow / single blink  →  +2  (default)
    //  • Fast double blink    →  the PAIR counts as +1 total:
    //      blink 1 (fast)  adds +2  (treated as slow until proven otherwise)
    //      blink 2 (fast, within FAST_MS of blink 1) subtracts 1 → net +1
    //      blink 3+ resets the cycle
    //
    const FAST_MS = 450; // ms window that defines a "fast double blink"
    const now     = Date.now();
    const gap     = now - this._lastBlinkTime;

    if (gap < FAST_MS && !this._lastBlinkWasFast) {
      // ── 2nd blink of a fast pair ──────────────────────────────────────
      // Blink 1 already added +2; we want the pair total to be +1,
      // so subtract 1 (keeping count non-negative).
      this.blinkCount      = Math.max(0, this.blinkCount - 1);
      this._lastBlinkWasFast = true;   // mark pair consumed; next blink resets
    } else {
      // ── Slow / single blink (or 3rd+ after a pair) ───────────────────
      this.blinkCount     += 2;
      this._lastBlinkWasFast = false;
    }

    this._lastBlinkTime = now;
    this._emitCountChange();
    if (typeof this.onBlink === 'function') {
      this.onBlink(this.blinkCount);
    }
  }

  _emitCountChange() {
    if (typeof this.onCountChange === 'function') {
      this.onCountChange(this.blinkCount);
    }
  }

  _emitStatus(message) {
    if (typeof this.onStatusChange === 'function') {
      this.onStatusChange(message);
    }
  }

  _handleError(err) {
    if (typeof this.onError === 'function') {
      this.onError(err);
    } else {
      console.error('[BlinkDetector]', err);
    }
    this._emitStatus('Error: ' + err.message);
  }
}
