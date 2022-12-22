from util.plot_utils import plot_logs
from pathlib import Path

log_directory = [Path('/path/to/log/file')]

# solid lines are training results,
# dashed lines are validation results.

# ## 0
fields_of_interest = (
    'loss',
    'mAP',
    )

## 1
# fields_of_interest = (
#     'loss_ce',
#     'loss_bbox',
#     'loss_giou',
#     )

# # ## 2
# fields_of_interest = (
#     'class_error',
#     'cardinality_error_unscaled',
#     )


plot_logs(log_directory,
          fields_of_interest)   
