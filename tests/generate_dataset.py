import json
from pathlib import Path
import os
import os.path as op
import numpy as np
import nibabel as nib
from bids.layout import BIDSLayout


def touch(fname, times=None):
    with open(fname, "a"):
        os.utime(fname, times)


def to_bids_description(
    path, fname="dataset_description.json", BIDSVersion="1.4.0", **kwargs
):
    """Dumps a dict into a bids description at the given location"""
    kwargs.update({"BIDSVersion": BIDSVersion})
    if (
        "GeneratedBy" in kwargs or "PipelineDescription" in kwargs
    ) and "DatasetType" not in kwargs:
        kwargs["DatasetType"] = "derivative"
    desc_file = op.join(path, fname)
    with open(desc_file, "w") as outfile:
        json.dump(kwargs, outfile)


def create_dummy_data(
    dir: Path,
    subject: str,
    session: str = None,
    shape=(100, 100, 100),
    ndirs: int = 8,
    simulate_kwargs=None,
):
    """Create dummy dwi data for a given subject and session

    Parameters
    ----------
    dir : Path
        directory to save dummy data
    subject : str
        subjectID
    session : str, optional
        sessionID, by default None
    shape : tuple, optional
        3d shape of image, by default (100, 100, 100)
    ndirs : int, optional
        number of diffusion directions, by default 8
    simulate_kwargs : dict, optional
        kwargs for multitensor.simulate, by default None

    Example
    -------
    >>> n_sessions = 2
    >>> n_subjects = 3
    >>> bids_path = "/tmp/data"
    >>> create_dummy_bids_dataset(bids_path, n_subjects, n_sessions)
    >>> layout = BIDSLayout(bids_path, derivatives=True)
    """
    dir = Path(dir)
    aff = np.eye(4)
    data_t1 = np.ones(shape)
    # Dummy brain mask
    data_brain_mask = np.zeros(shape)
    data_brain_mask[20:80, 20:80, 20:80] = 1  # Assuming brain occupies this region

    # Dummy tissue segmentation
    data_tissue_seg = np.zeros(shape)
    data_tissue_seg[25:75, 25:75, 25:75] = 2  # Gray matter
    data_tissue_seg[30:70, 30:70, 30:70] = 3  # White matter
    data_tissue_seg[35:65, 35:65, 35:65] = 1  # CSF

    import multitensor as mt

    data_dwi, gtab = mt.simulate(
        dseg=data_tissue_seg,
        n_dirs=ndirs,
        bvals=[1000],
        plot=False,
        **simulate_kwargs,
    )
    bvecs = gtab.bvecs
    bvals = gtab.bvals

    if session is None:
        data_dir = dir / subject
    else:
        data_dir = dir / subject / session
    data_dir.mkdir(parents=True, exist_ok=True)

    sub_ses = f"{subject}_{session}"

    np.savetxt((data_dir / "dwi" / f"{sub_ses}_dwi.bval"), bvals)
    np.savetxt((data_dir / "dwi" / f"{sub_ses}_dwi.bvec"), bvecs)
    nib.save(
        nib.Nifti1Image(data_dwi, aff), (data_dir / "dwi" / f"{sub_ses}_dwi.nii.gz")
    )
    nib.save(
        nib.Nifti1Image(data_t1, aff), (data_dir / "anat" / f"{sub_ses}_T1w.nii.gz")
    )
    # Save brain mask
    brain_mask_file = data_dir / "anat" / f"{sub_ses}_desc-brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(data_brain_mask, aff), brain_mask_file)

    # Save tissue segmentation
    tissue_seg_file = data_dir / "anat" / f"{sub_ses}_dseg.nii.gz"
    nib.save(nib.Nifti1Image(data_tissue_seg, aff), tissue_seg_file)


def create_dummy_bids_dataset(path, n_subjects, n_sessions, kwargs=None):
    path.mkdir(parents=True, exist_ok=True)
    subjects = ["sub-0%s" % (d + 1) for d in range(n_subjects)]
    sessions = ["ses-0%s" % (d + 1) for d in range(n_sessions)]
    to_bids_description(
        path, **{"Name": "Dummy", "Subjects": subjects, "Sessions": sessions}
    )
    pipeline_description = {"Name": "Dummy", "GeneratedBy": [{"Name": "synthetic"}]}
    deriv_dir = path / "derivatives/synthetic"
    deriv_dir.mkdir(parents=True, exist_ok=True)
    to_bids_description(deriv_dir, **pipeline_description)

    for subject in subjects:
        for session in sessions:
            for modality in ["anat", "dwi"]:
                (deriv_dir / subject / session / modality).mkdir(
                    parents=True, exist_ok=True
                )
                (path / subject / session / modality).mkdir(parents=True, exist_ok=True)
            # Make some dummy data:
            create_dummy_data(deriv_dir, subject, session, simulate_kwargs=kwargs)
            # create_dummy_data(path, subject, session)
