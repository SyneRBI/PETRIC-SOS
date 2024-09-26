import numpy
import cil

from sirf.STIR import AcquisitionData
AcquisitionData.set_storage_scheme('memory')

from main import Submission, submission_callbacks  # your submission (`main.py`)
from petric import data, metrics, QualityMetrics  # our data & evaluation
assert issubclass(Submission, cil.optimisation.algorithms.Algorithm)

metrics_with_timeout = metrics[0]
if data.reference_image is not None:
    metrics_with_timeout.callbacks.append(
        QualityMetrics(data.reference_image, data.whole_object_mask, data.background_mask,
                        tb_summary_writer=metrics_with_timeout.tb, voi_mask_dict=data.voi_masks))
metrics_with_timeout.reset() # timeout from now

Submission(data).run(numpy.inf, callbacks=metrics + submission_callbacks)