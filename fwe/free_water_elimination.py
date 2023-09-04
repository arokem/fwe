"""Main workflow for the free water elimination pipeline.
"""
# %%
import logging
from pathlib import Path
from bids.layout import BIDSLayout
import nibabel as nib
import numpy as np
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# %%
def eliminate_water(bids_path, n_iterations=100, lr=0.0005):
    layout = BIDSLayout(bids_path, derivatives=True)
    subject = layout.get_subjects()[0]
    session = layout.get_sessions()[0]

    context = {
        "subject": subject,
        "session": session,
        "scope": "synthetic",
        "return_type": "filename",
    }
    bval_fp = layout.get(**context, extension="bval")
    bvec_fp = layout.get(**context, extension="bvec")
    img_fp = layout.get(**context, suffix="dwi", extension="nii.gz")[0]
    if not bval_fp or not bvec_fp:
        logger.error("bval file not found for the specified subject and session.")

    logger.debug(f"Found bval file at: {bval_fp[0]}")
    bvals, bvecs = read_bvals_bvecs(str(bval_fp[0]), str(bvec_fp[0]))
    unique_bvals = np.unique(bvals[bvals != 0])

    if len(unique_bvals) == 0:
        logger.error("No non-zero b-values found. Data might be incorrect.")
    elif len(unique_bvals) == 1:
        logger.warning("This is single shell data. Using Beltrami model.")
        from fwe import beltrami

        gtab1 = gradient_table(bvals, bvecs, b0_threshold=0)
        gtab2 = gradient_table(bvals * 10**-3, bvecs, b0_threshold=0)
        img = nib.load(img_fp)
        img_np = img.get_fdata()
        affine = img.affine
        b0_data = img_np[:, :, :, gtab1.b0s_mask]
        mean_b0 = np.mean(b0_data, axis=-1)

        # Load the brain mask
        brain_mask_fp = layout.get(**context, suffix="mask", extension="nii.gz")[0]
        if not brain_mask_fp:
            logger.warning(
                "Brain mask not found for the specified subject and session."
            )
        else:
            logger.debug(f"Found brain mask at: {brain_mask_fp}")
            brain_mask = nib.load(brain_mask_fp).get_fdata()
        # load tissue segmentation mask
        tissue_seg_fp = layout.get(**context, suffix="dseg", extension="nii.gz")[0]
        logger.debug(f"Found tissue segmentation mask at: {tissue_seg_fp}")
        tissue_seg_np = nib.load(tissue_seg_fp).get_fdata()
        wm_mask = tissue_seg_np == 3
        csf_mask = tissue_seg_np == 1
        St = np.round(np.percentile(mean_b0[wm_mask], 95))
        Sw = np.round(np.percentile(mean_b0[csf_mask], 95))
        logger.info(f"Stissue: {St}, Swater: {Sw}")
        logger.info("Running Beltrami model...")
        bmodel = beltrami.BeltramiModel(
            gtab2,
            init_method="hybrid",
            Stissue=St,
            Swater=Sw,
            iterations=n_iterations,
            learning_rate=lr,
        )
        bfit = bmodel.fit(img_np, mask=brain_mask)
        # Predict back the free water corrected signal:
        we_signal = beltrami.fw_model_prediction(
            bfit.model_params, gtab2, S0=mean_b0, Diso=0.0003
        )
        # Remove the free water signal from the original signal:
        wf_dwi = img_np - we_signal

        wf_dwi_fp = "_rec-wf_dwi".join(img_fp.split("_dwi"))
        wf_dwi_nii = nib.Nifti1Image(wf_dwi, affine, header=img.header)
        nib.save(wf_dwi_nii, wf_dwi_fp)

        return wf_dwi_fp

    else:
        logger.info("This is multi-shell data. Not yet implemented.")


# %%
# Check eigen values of diffusion tensor on water free
