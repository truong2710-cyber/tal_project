dataset_info = dict(
    dataset_name='custom_coco_20',
    paper_info=dict(
        author='Custom Dataset for Pose Estimation',
        title='Pose Estimation 20 Keypoints',
        container='Internal',
        year='2025',
        homepage='N/A',
    ),
    # -----------------------------------------------------------
    # KEYPOINT INFO (20 Keypoints)
    # -----------------------------------------------------------
    keypoint_info={
        # Bắt đầu từ ID 0
        0: dict(name='R Ear', id=0, color=[255, 128, 0], type='upper', swap='L Ear'),
        1: dict(name='L Ear', id=1, color=[0, 255, 0], type='upper', swap='R Ear'),
        2: dict(name='Nose', id=2, color=[51, 153, 255], type='upper', swap=''),
        3: dict(name='R Shoulder', id=3, color=[255, 128, 0], type='upper', swap='L Shoulder'),
        4: dict(name='L Shoulder', id=4, color=[0, 255, 0], type='upper', swap='R Shoulder'),
        5: dict(name='R Hip', id=5, color=[255, 128, 0], type='lower', swap='L Hip'),
        6: dict(name='L Hip', id=6, color=[0, 255, 0], type='lower', swap='R Hip'),
        7: dict(name='R Elbow', id=7, color=[255, 128, 0], type='upper', swap='L Elbow'),
        8: dict(name='L Elbow', id=8, color=[0, 255, 0], type='upper', swap='R Elbow'),
        9: dict(name='L Wrist', id=9, color=[0, 255, 0], type='upper', swap='R Wrist'),
        10: dict(name='R Wrist', id=10, color=[255, 128, 0], type='upper', swap='L Wrist'),
        11: dict(name='R Knee', id=11, color=[255, 128, 0], type='lower', swap='L Knee'),
        12: dict(name='L Knee', id=12, color=[0, 255, 0], type='lower', swap='R Knee'),
        13: dict(name='L Ankle', id=13, color=[0, 255, 0], type='lower', swap='R Ankle'),
        14: dict(name='R Ankle', id=14, color=[255, 128, 0], type='lower', swap='L Ankle'),
        15: dict(name='R Foot Tip', id=15, color=[255, 128, 0], type='lower', swap='L Foot Tip'),
        16: dict(name='L Foot Tip', id=16, color=[0, 255, 0], type='lower', swap='R Foot Tip'),
        17: dict(name='Stick Top', id=17, color=[128, 128, 128], type='other', swap=''),
        18: dict(name='Stick Bottom', id=18, color=[128, 128, 128], type='other', swap=''),
        19: dict(name='Stick Tip', id=19, color=[128, 128, 128], type='other', swap=''),
    },
    skeleton_info={
        # [6, 12] -> indices (5, 11) -> (R Hip, R Knee)
        0: dict(link=('R Hip', 'R Knee'), id=0, color=[255, 128, 0]),
        # [3, 1] -> indices (2, 0) -> (Nose, R Ear)
        1: dict(link=('Nose', 'R Ear'), id=1, color=[51, 153, 255]),
        # [4, 6] -> indices (3, 5) -> (R Shoulder, R Hip)
        2: dict(link=('R Shoulder', 'R Hip'), id=2, color=[255, 128, 0]),
        # [13, 14] -> indices (12, 13) -> (L Knee, L Ankle)
        3: dict(link=('L Knee', 'L Ankle'), id=3, color=[51, 153, 255]),
        # [7, 13] -> indices (6, 12) -> (L Hip, L Knee)
        4: dict(link=('L Hip', 'L Knee'), id=4, color=[255, 128, 0]),
        # [18, 19] -> indices (17, 18) -> (Stick Top, Stick Bottom)
        5: dict(link=('Stick Top', 'Stick Bottom'), id=5, color=[128, 128, 128]),
        # [4, 5] -> indices (3, 4) -> (R Shoulder, L Shoulder)
        6: dict(link=('R Shoulder', 'L Shoulder'), id=6, color=[0, 255, 0]),
        # [4, 8] -> indices (3, 7) -> (R Shoulder, R Elbow)
        7: dict(link=('R Shoulder', 'R Elbow'), id=7, color=[255, 128, 0]),
        # [12, 15] -> indices (11, 14) -> (R Knee, R Ankle)
        8: dict(link=('R Knee', 'R Ankle'), id=8, color=[255, 128, 0]),
        # [5, 9] -> indices (4, 8) -> (L Shoulder, L Elbow)
        9: dict(link=('L Shoulder', 'L Elbow'), id=9, color=[0, 255, 0]),
        # [8, 11] -> indices (7, 10) -> (R Elbow, R Wrist)
        10: dict(link=('R Elbow', 'R Wrist'), id=10, color=[0, 255, 0]),
        # [9, 10] -> indices (8, 9) -> (L Elbow, L Wrist)
        11: dict(link=('L Elbow', 'L Wrist'), id=11, color=[51, 153, 255]),
        # [1, 2] -> indices (0, 1) -> (R Ear, L Ear)
        12: dict(link=('R Ear', 'L Ear'), id=12, color=[51, 153, 255]),
        # [19, 20] -> indices (18, 19) -> (Stick Bottom, Stick Tip)
        13: dict(link=('Stick Bottom', 'Stick Tip'), id=13, color=[128, 128, 128]),
        # [15, 16] -> indices (14, 15) -> (R Ankle, R Foot Tip)
        14: dict(link=('R Ankle', 'R Foot Tip'), id=14, color=[255, 128, 0]),
        # [6, 7] -> indices (5, 6) -> (R Hip, L Hip)
        15: dict(link=('R Hip', 'L Hip'), id=15, color=[51, 153, 255]),
        # [14, 17] -> indices (13, 16) -> (L Ankle, L Foot Tip)
        16: dict(link=('L Ankle', 'L Foot Tip'), id=16, color=[128, 128, 128]),
        # [2, 3] -> indices (1, 2) -> (L Ear, Nose)
        17: dict(link=('L Ear', 'Nose'), id=17, color=[51, 153, 255]),
        # [7, 5] -> indices (6, 4) -> (L Hip, L Shoulder)
        18: dict(link=('L Hip', 'L Shoulder'), id=18, color=[255, 128, 0]),
    },
        joint_weights=[
        1.0,
        1.0,
        1.3,
        1.2,
        1.2,
        1.1,
        1.1, 
        1.2,
        1.2,
        1.5,
        1.5,
        1.1,
        1.1,
        1.3,
        1.3,
        1.5,
        1.5,
        1.2,
        1.2,
        1.2,
    ],
    sigmas=[
        0.026,
        0.026,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.062,
        0.062,
        0.080,
        0.080,
        0.070,
    ]
)