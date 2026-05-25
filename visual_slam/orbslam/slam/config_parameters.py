"""
Central runtime parameters for the RGB-D SLAM pipeline.
This module collects feature, tracking, mapping, loop-closing, and optimization constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Collect the global constants used by tracking, mapping, and optimization.
class Parameters:
    # ================================================================
    # C++ core / runtime selection
    # ================================================================
    USE_CPP_CORE = False

    # ================================================================
    # Sparse SLAM threading
    # ================================================================
    kLocalMappingOnSeparateThread = False
    kTrackingWaitForLocalMappingToGetIdle = False
    kWaitForLocalMappingTimeout = 0.5
    kParallelLBAWaitIdleTimeout = 0.3

    # ================================================================
    # Feature management
    # ================================================================
    kNumFeatures = 2000
    kUseDynamicDesDistanceTh = True
    kUseDescriptorSigmaMadv2 = False

    kSigmaLevel0 = 1.0
    kFeatureMatchDefaultRatioTest = 0.7
    kKdtNmsRadius = 3
    kCheckFeaturesOrientation = True

    kORBNumLevels = 8
    kORBScaleFactor = 1.2
    kORBDeterministic = True
    kDescriptorSize = 32

    # ================================================================
    # Point triangulation / visibility
    # ================================================================
    kCosMaxParallaxInitializer = 0.99998
    kCosMaxParallax = 0.9998
    kMinRatioBaselineDepth = 0.01

    kViewingCosLimitForPoint = 0.5
    kScaleConsistencyFactor = 1.5
    kMaxDistanceToleranceFactor = 1.2
    kMinDistanceToleranceFactor = 0.8

    # ================================================================
    # Initializer
    # ================================================================
    kInitializerDesiredMedianDepth = 1
    kInitializerMinRatioDepthBaseline = 100
    kInitializerNumMinFeatures = 100
    kInitializerNumMinFeaturesStereo = 500
    kInitializerNumMinTriangulatedPoints = 150
    kInitializerNumMinTriangulatedPointsStereo = 100
    kInitializerFeatureMatchRatioTest = 0.9
    kInitializerNumMinNumPointsForPnPWithDepth = 15
    kInitializerUseCellCoverageCheck = True
    kInitializerUseMinFrameDistanceCheck = True

    # ================================================================
    # Tracking
    # ================================================================
    kUseMotionModel = True
    kUseSearchFrameByProjection = True
    kMinNumMatchedFeaturesSearchFrameByProjection = 20
    kUseEssentialMatrixFitting = False
    kMinNumMatchedFeaturesSearchReferenceFrame = 15
    kMaxNumOfKeyframesInLocalMap = 80
    kNumBestCovisibilityKeyFrames = 10
    kNumBestCovisibilityKeyFramesTracking = kNumBestCovisibilityKeyFrames
    kExpandLocalMapWithParent = True
    kExpandLocalMapWithChildren = True
    kUseVisualOdometryPoints = True
    kMaxNumVisualOdometryPoints = 100
    kMaxNumStereoPointsOnNewKeyframe = 150   # phase1_lab: bumped from 100 to compensate for sparser KFs
    kUseInterruptLocalMapping = False

    kMaxOutliersRatioInPoseOptimization = 0.9

    kUseMotionBlurDection = True
    kMotionBlurDetectionLalacianVarianceThreshold = 100.0
    kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography = 30

    # ================================================================
    # Keyframe generation
    # ================================================================
    kNumMinPointsForNewKf = 15
    kNumMinTrackedClosePointsForNewKfNonMonocular = 100
    kNumMaxNonTrackedClosePointsForNewKfNonMonocular = 70
    kThNewKfRefRatioMonocular = 0.9
    kThNewKfRefRatioStereo = 0.85   # phase1_lab: was 0.90, eased slightly since c1a hard override disabled
    kThNewKfRefRatioNonMonocular = 0.25
    kUseFeatureCoverageControlForNewKf = False
    kUseFovCentersBasedKfGeneration = False
    kMaxFovCentersDistanceForKfGeneration = 0.2
    kMinFramesBetweenKeyframesSequentialRgbd = 10   # phase1_lab: was 5, matches slow-rover lab dynamics
    kMinFramesBetweenKeyframesThreadedRgbd = 0
    kMaxFramesBetweenKeyframesRgbd = -1         # phase1_lab: disabled c1a hard override (was 10); -1 = use fps default = 30
    kUseFpsAwareKeyframeSpacing = True
    kMinKeyframeSpacingSeconds = 0.30   # phase1_lab: 0.30s floor (was 0.10) for slow-rover
    kLocalMappingMaxQueueForForcedInsert = 3
    kNewKeyframeRefMinObs = -1

    # ================================================================
    # Keyframe culling
    # ================================================================
    kKeyframeCullingRedundantObsRatio = 0.60   # phase1_lab: was 0.45, more aggressive culling for sparser KFs
    kKeyframeMaxTimeDistanceInSecForCulling = 0.10
    kKeyframeCullingMinNumPoints = 0

    # ================================================================
    # Stereo / RGB-D matching
    # ================================================================
    kStereoMatchingMaxRowDistance = 1.1
    kStereoMatchingShowMatchedPoints = False

    # ================================================================
    # Search matches by projection
    # ================================================================
    kMaxReprojectionDistanceFrame = 7
    kMaxReprojectionDistanceFrameNonStereo = 15
    kMaxReprojectionDistanceMap = 3
    kMaxReprojectionDistanceMapRgbd = 3
    kMaxReprojectionDistanceMapReloc = 5
    kMaxReprojectionDistanceFuse = 3
    kMaxReprojectionDistanceSim3 = 30.0

    kMatchRatioTestFrameByProjection = 0.9
    kMatchRatioTestMap = 0.8
    kMatchRatioTestEpipolarLine = 0.8

    kMaxDescriptorDistance = 0
    kMinDistanceFromEpipole = 10

    # ================================================================
    # Local Mapping
    # ================================================================
    kLocalMappingParallelKpsMatching = True
    kLocalMappingParallelKpsMatchingNumWorkers = 2
    kLocalMappingParallelFusePointsNumWorkers = 2
    kLocalMappingDebugAndPrintToFile = True
    kLocalMappingNumNeighborKeyFramesStereo = 10
    kLocalMappingNumNeighborKeyFramesMonocular = 20
    kLocalMappingTimeoutPopKeyframe = 0.5

    # ================================================================
    # Covisibility graph
    # ================================================================
    kMinNumOfCovisiblePointsForCreatingConnection = 15

    # ================================================================
    # Optimization engine
    # ================================================================
    kOptimizationAllUseGtsam = False
    kOptimizationFrontEndUseGtsam = False
    kOptimizationBundleAdjustUseGtsam = False
    kOptimizationLoopClosingUseGtsam = False

    # ================================================================
    # Bundle Adjustment
    # ================================================================
    kLocalBAWindowSize = 20
    kUseLargeWindowBA = False
    kEveryNumFramesLargeWindowBA = 10
    kLargeBAWindowSize = 20
    kUseParallelProcessLBA = False
    kEnableLocalBAStarvationGuard = True
    kMaxKeyframesWithoutLocalBA = 5
    kMaxConsecutiveLocalBAAborts = 3
    kForceLocalBAWhenStarved = True

    # ================================================================
    # Global Bundle Adjustment
    # ================================================================
    kUseGBA = False
    kGBADebugAndPrintToFile = True
    kGBAUseRobustKernel = True
    kGlobalBAIterations = 10
    kGlobalBAMinInlierEdges = 10

    # ================================================================
    # Loop closing
    # ================================================================
    kUseLoopClosing = True
    kLoopClosingCommonWordRatioThreshold = 0.55   # phase1_lab: was 0.50, slightly tightened (sparser KFs give consistency filter more room)
    kMinDeltaFrameForMeaningfulLoopClosure = 10
    kMaxResultsForLoopClosure = 10   # baseline 2_35V value
    kLoopCandidateSource = "auto"
    kLoopDbowDetectorTopK = 10       # baseline 2_35V value
    kLoopOracleMaxGtTimeDiffSec = 0.05
    kLoopDetectingTimeoutPopKeyframe = 0.5
    kLoopClosingDebugWithLoopDetectionImages = False
    kLoopClosingDebugWithSimmetryMatrix = True
    kLoopClosingDebugAndPrintToFile = True
    kLoopClosingDebugWithLoopConsistencyCheckImages = True
    kLoopClosingDebugShowLoopMatchedPoints = False
    kLoopClosingParallelKpsMatching = True
    kLoopClosingParallelKpsMatchingNumWorkers = 2
    kLoopClosingGeometryCheckerMinKpsMatches = 9
    kLoopClosingSE3GuidedMinSeedInliers = 4
    kLoopClosingMaxEstimatedPoseDistanceForGuidedSE3 = 0.0
    kLoopClosingMaxEstimatedPoseRotationDegForGuidedSE3 = 0.0
    kLoopClosingSE3RansacMaxError = 0.25
    kLoopClosingSE3RansacIterations = 300
    kLoopClosingTh2 = 20
    kLoopClosingMaxReprojectionDistanceMapSearch = 20
    kLoopClosingMinNumMatchedMapPoints = 40
    kLoopClosingMaxReprojectionDistanceFuse = 4
    kLoopClosingFeatureMatchRatioTest = 0.75
    kLoopClosingSim3DriftSigmaFactor = 100.0

    # [PHASE2-CONNECTED-TEMPORAL-WINDOW] (added 2026-05-18)
    # Restrict the connected/covisibility filter used during loop candidate
    # retrieval to a temporal window. Only KFs whose KID is within +/- N
    # temporal steps of the query are treated as "connected" for the purpose
    # of being filtered out as loop candidates. Long-range covisibility
    # connections induced by low-drift revisits (e.g. lab out-and-back) are
    # therefore allowed to participate as loop candidates.
    #   0  -> legacy behavior: all KFs in covisibility graph are filtered.
    #   >0 -> only temporally near covisibility neighbors are filtered.
    # Default 30 matches typical sliding-window KF coverage; set to 0 to
    # revert without code changes.
    kLoopConnectedFilterTemporalWindowKf = 30

    # [PHASE2-MIN-SCORE-RELAX] (added 2026-05-18)
    # Multiplicative relaxation applied to the dynamic min_score gate.
    # min_score is otherwise the lowest BoW score among the query's
    # covisibility neighbors (pyslam loop_detector_base.py:322). For lab
    # out-and-back trajectories the neighbors share strong views and yield
    # min_score ~0.25; genuine spatial revisits viewed from opposite headings
    # land at ~0.04 BoW. Scaling the gate down lets those revisits through.
    #   1.0 -> legacy behavior (min_score gate unchanged).
    #   <1  -> more permissive (admits lower-scoring candidates).
    # Default 0.15 was sized from observed neighbor:revisit score ratios in
    # the Phase 1 lab run (revisit ~0.04 / neighbor ~0.25 = 0.16). Increase
    # toward 1.0 if false-positive loops appear.
    kLoopClosingMinScoreRelaxFactor = 0.15

    # [PHASE2-SIM3-INLIER-RELAX] (added 2026-05-18)
    # Override applied AT THE LOOP-CLOSING SOLVER CALL only — the geometry
    # checker's BoW-match floor still uses kLoopClosingGeometryCheckerMinKpsMatches.
    # Lowers the Sim3Solver seed-inlier RANSAC threshold so candidates with
    # 2-3 high-quality 3D correspondences can still produce a Sim3 hypothesis
    # for optimize_sim3 to refine. In the Phase 1 lab run, max RANSAC inliers
    # was 6 / mean 2.94 (out of 40+ correspondences) — too few to survive the
    # original 4-inlier seed threshold.
    #   <=0 -> use kLoopClosingSE3GuidedMinSeedInliers unchanged.
    #   >0  -> override (use this value as the Sim3Solver seed threshold).
    # Default 2 is the absolute minimum for RANSAC to produce a transform.
    # Raise back toward 4 if false-positive loops appear.
    # Phase 3 (2026-05-18): tightened 2 -> 3 after Phase 2 produced 1 FP
    # (KF234<->KF23, GBA outlier 38.7%) plus 1 marginal (KF108<->KF162,
    # 748 mm spatial). 3-inlier hypotheses should suppress the marginals
    # while keeping clearly-genuine 4+ inlier revisits intact.
    kLoopClosingSim3SeedInliersOverride = 3

    # information; this SE3 port keeps conservative non-identity weights to
    # distinguish structural, covisible, and loop constraints.
    kEssentialGraphSpanningTreeWeight = 1.0
    kEssentialGraphCovisibilityWeightScale = 0.01
    kEssentialGraphCovisibilityWeightMin = 0.5
    kEssentialGraphCovisibilityWeightMax = 5.0
    kEssentialGraphLoopEdgeWeight = 10.0

    # ================================================================
    # Relocalization
    # ================================================================
    kRelocalizationDebugAndPrintToFile = True
    kRelocalizationMinKpsMatches = 15
    kRelocalizationParallelKpsMatching = True
    kRelocalizationParallelKpsMatchingNumWorkers = 2
    kRelocalizationFeatureMatchRatioTest = 0.75
    kRelocalizationFeatureMatchRatioTestLarge = 0.9
    kRelocalizationPoseOpt1MinMatches = 10
    kRelocalizationDoPoseOpt2NumInliers = 50
    kRelocalizationMaxReprojectionDistanceMapSearchCoarse = 10
    kRelocalizationMaxReprojectionDistanceMapSearchFine = 3

    # ================================================================
    # Common reprojection thresholds
    # ================================================================
    kChi2Mono = 5.991
    kChi2Stereo = 7.815
    kHuberMono = math.sqrt(kChi2Mono)
    kHuberStereo = math.sqrt(kChi2Stereo)
    kMinDepth = 1e-2

    # ================================================================
    # Memory Policy
    # ================================================================
    kMaxLenFrameDeque = 20
    kEnableFrameEvictionCleanup = True
    kEnableFrameViewPruning = True
    kFrameViewPruneEveryNFrames = 20
    kFrameViewRetention = 20

    kStoreNormalFrameImages = False
    kStoreKeyFrameImages = True
    kStoreKeyFrameDepthImages = False
    kReleaseNormalFrameImagesAfterUse = True
    kReleaseEvictedFrameFeatureCache = False

    # ================================================================
    # RGB-D helper defaults
    # ================================================================
    kDefaultRgbdBaselineMeters = 0.08


# Hold optional runtime overrides layered on top of the global parameters.
@dataclass
class OrbSlamSettings:
    """
    Small instance-level settings wrapper for runners.

    is only for future runner-level overrides.
    """

    sensor_type_name: str = "rgbd"
    num_features: int = Parameters.kNumFeatures
    num_levels: int = Parameters.kORBNumLevels
    scale_factor: float = Parameters.kORBScaleFactor
    deterministic: bool = Parameters.kORBDeterministic
    use_loop_closing: bool = Parameters.kUseLoopClosing
    use_local_mapping: bool = True
    use_relocalization: bool = True
