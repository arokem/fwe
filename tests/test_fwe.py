import logging
from bids import BIDSLayout
import nibabel as nib
import numpy as np
from generate_dataset import create_dummy_bids_dataset
from pathlib import Path
from fwe.free_water_elimination import eliminate_water
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
import tempfile

logger = logging.getLogger(__name__)


def test_eliminate_water():
    n_sessions = 1
    n_subjects = 1
    with tempfile.TemporaryDirectory(prefix="test-fwe_") as bids_path:
        logger.info("Created temporary directory: %s", bids_path)
        bids_path = Path(bids_path)
        water_diffusivity = 0.0003
        tensor_og = [
            np.random.randint(4, 10) * water_diffusivity,
            water_diffusivity,
            water_diffusivity,
        ]
        logger.info("Original tensors: %s", tensor_og)
        create_dummy_bids_dataset(
            bids_path, n_subjects, n_sessions, kwargs=dict(wm_tensor=tensor_og)
        )
        fw_dwi_fp = eliminate_water(bids_path, n_iterations=100, lr=0.0005)

        # Load data
        layout = BIDSLayout(bids_path, derivatives=True)
        subject = layout.get_subjects()[0]
        session = layout.get_sessions()[0]
        context = {
            "subject": subject,
            "session": session,
            "scope": "synthetic",
            "return_type": "filename",
        }
        bval_fp = layout.get(**context, extension="bval")[0]
        bvec_fp = layout.get(**context, extension="bvec")[0]
        gtab = gradient_table(bval_fp, bvec_fp)
        brain_mask_np = nib.load(layout.get(**context, suffix="mask")[0]).get_fdata()
        tissue_seg_np = nib.load(layout.get(**context, suffix="dseg")[0]).get_fdata()
        fw_dwi_np = nib.load(fw_dwi_fp).get_fdata()

        # Fit tensor model on water free data
        dtimodel = dti.TensorModel(gtab)
        dti_fit = dtimodel.fit(fw_dwi_np, mask=brain_mask_np)

        # Compare with original tensors
        white_matter_mask = tissue_seg_np == 3
        tensor_recon = dti_fit.evals[white_matter_mask]  # evals=eigenvalues
        # We're assuming the same tensor to all voxels in WM
        tensor_recon = tensor_recon.mean(axis=0)
        print(f"Reconstructed tensors in WM: {tensor_recon}")
        print(f"Original tensors in WM:      {tensor_og}")
        print(f"Original free water in WM:   {[0.0003, 0.0003, 0.0003]}")
        assert np.allclose(tensor_og, tensor_recon, rtol=0.3, atol=0.0002)


# test_eliminate_water()
