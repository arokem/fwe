# Adapted from https://dipy.org/documentation/1.7.0/examples_built/26_simulations/simulate_multi_tensor/#sphx-glr-examples-built-26-simulations-simulate-multi-tensor-py

import numpy as np
from dipy.core.sphere import disperse_charges, HemiSphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor, multi_tensor_odf


def simulate(
    dseg, n_dirs=64, bvals=[1000, 2500], wm_tensor=[0.0015, 0.0003, 0.0003], plot=False
):
    """Generate simulation dwi data with free water in white matter, size of dseg

    Args:
        dseg (np.array,): tissue segmentation file
        n_dirs (int, optional): number of directions. Defaults to 64.
        bvals (list, optional): b values. Defaults to [1000, 2500].
        wm_tensor (list, optional): white matter tensor.
            Defaults to [0.0015, 0.0003, 0.0003].
        plot (bool, optional): whether to plot the signal. Defaults to False.

    Returns:
        tuple: dwi data, gradient table

    Example:
        signal, signal_noisy = simulate(n_dirs=64, bvals=[1000], plot=True)
        data_tissue_seg = np.zeros((100,100,100))
        data_tissue_seg[25:75, 25:75, 25:75] = 2  # Gray matter
        data_tissue_seg[30:70, 30:70, 30:70] = 3  # White matter
        data_tissue_seg[35:65, 35:65, 35:65] = 1  # CSF
        img_np, gtab = simulate(n_dirs=64, bvals=[1000], dseg=data_tissue_seg, plot=True)
    """
    n_dirs = n_dirs - 2  # we're adding 2 b0s at the end
    # For the simulation we will need a GradientTable with the b-values and
    # b-vectors. To create one, we can first create some random points on a
    # ``HemiSphere`` using spherical polar coordinates.
    theta = np.pi * np.random.rand(n_dirs)
    phi = 2 * np.pi * np.random.rand(n_dirs)
    hsph_initial = HemiSphere(theta=theta, phi=phi)

    # Next, we call ``disperse_charges`` which will iteratively move the points so
    # that the electrostatic potential energy is minimized.
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)

    # We need n stacks of ``vertices``, one for every shell, and we need n sets
    # of b-values and b-vectors as well
    vertices = hsph_updated.vertices
    values = np.ones(vertices.shape[0])

    bvecs = np.vstack([vertices] * len(bvals))
    bvals = np.hstack([bv * values for bv in bvals])

    # We can also add some b0s. Let's add one at the beginning and one at the end.
    bvecs = np.insert(bvecs, (0, bvecs.shape[0]), np.array([0, 0, 0]), axis=0)
    bvals = np.insert(bvals, (0, bvals.shape[0]), 0)

    # Let's now create the ``GradientTable``.
    gtab = gradient_table(bvals, bvecs)

    meta_dict = {
        1: {
            "type": "CSF",
            "mevals": np.array([[0.0003, 0.0003, 0.0003], [0.0003, 0.0003, 0.0003]]),
            "angles": [(0, 0), (60, 0)],
            "fractions": [50, 50],
            "S0": 5,
        },
        2: {
            "type": "GM",
            "mevals": np.array([[0.0005, 0.0003, 0.0003], [0.0003, 0.0003, 0.0003]]),
            "angles": [(0, 0), (60, 0)],
            "fractions": [10, 90],
            "S0": 80,
        },
        3: {
            "type": "WM",
            "mevals": np.array(
                [
                    wm_tensor,  # TODO: we can even add more tensors here
                    [0.0003, 0.0003, 0.0003],
                ]
            ),
            "angles": [(0, 0), (60, 0)],
            "fractions": [90, 10],
            "S0": 100,
        },
    }
    img_np = np.zeros((*dseg.shape, n_dirs + 2))
    assert len(np.unique(dseg[dseg != 0])) == 3, "only 3 tissue types supported"
    for key, params in meta_dict.items():
        tissue_type = params.pop("type")
        signal, _ = multi_tensor(gtab, snr=20, **params)
        img_np[dseg == key] = signal
        if plot:
            import matplotlib.pyplot as plt

            plt.plot(signal, label="noiseless")
            plt.legend()
            # plt.show()
            plt.savefig(f"../figures/simulated_signal_{tissue_type}.png")
    return img_np, gtab
