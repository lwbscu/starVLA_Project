"""H200 shared cross-dataset mixtures discovered under public/three/dataset.

The data configs for the robot types live in the existing LIBERO and
SimplerEnv registries. This file only adds dataset-name mixtures that match the
actual H200 shared directory layout from the dataset scan.
"""


ROBOT_TYPE_CONFIG_MAP = {}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}


DATASET_NAMED_MIXTURES = {
    # Stable default for cross-dataset pretraining:
    # all four datasets are LeRobot v2.1, franka, and include modality.json.
    "h200_three_libero_only": [
        ("libero_object", 1.0, "libero_franka"),
        ("libero_goal", 1.0, "libero_franka"),
        ("libero_spatial", 1.0, "libero_franka"),
        ("libero_10", 1.0, "libero_franka"),
    ],
    # Same semantic LIBERO split, but using the alternative no_noops directory
    # names if that copy is preferred on the cluster.
    "h200_three_libero_no_noops": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    # Experimental: the current H200 scan found bridge/fractal as LeRobot v2.0
    # roots without modality.json. The launcher preflight intentionally rejects
    # these until their modality metadata is supplied.
    "h200_three_oxe_experimental": [
        ("bridge", 1.0, "oxe_bridge"),
        ("fractal", 1.0, "oxe_rt1"),
    ],
    "h200_three_all_experimental": [
        ("libero_object", 1.0, "libero_franka"),
        ("libero_goal", 1.0, "libero_franka"),
        ("libero_spatial", 1.0, "libero_franka"),
        ("libero_10", 1.0, "libero_franka"),
        ("bridge", 1.0, "oxe_bridge"),
        ("fractal", 1.0, "oxe_rt1"),
    ],
}
