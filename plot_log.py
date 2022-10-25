from util.plot_utils import plot_logs
from pathlib import Path

# log_directory = [Path('outputs/R50_scene_test/trial_2')]
log_directory = [Path('/mnt/workspace/users/ymjian/test/exps/fine_tune_test/image_based/dn_d_detr_from_scratch')]

# solid lines are training results,
# dashed lines are validation results.

# ## 0
fields_of_interest = (
    'loss',
    'mAP',
    )

## 1
fields_of_interest = (
    'loss_ce',
    'loss_bbox',
    'loss_giou',
    )

## 2
fields_of_interest = (
    'class_error',
    'cardinality_error_unscaled',
    )

###################
plot_logs(log_directory,
          fields_of_interest)   
