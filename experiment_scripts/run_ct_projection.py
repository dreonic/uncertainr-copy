# Enable import from parent package
from collections import defaultdict
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, training, loss_functions, modules
import _utils as utils

from torch.utils.data import DataLoader
from functools import partial
import wandb
import pytorch_lightning as pl
import hydra
import numpy as np
import torch
import tree
from copy import deepcopy


@hydra.main(config_path="../conf", config_name="config")
def main(args):
    assert args.model.name == "mlp"
    root_path = os.getcwd()
    args.seed = (
        args.run if args.seed == 0 else int(torch.randint(0, 2**32 - 1, (1,)).item())
    )
    print(f"Setting seed to {args.seed}.")
    pl.seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sidelength = args.data.sidelength

    if args.data.zero_pad:
        padding = int(((sidelength**2 + sidelength**2) ** 0.5 - sidelength) / 2) + 1
    else:
        padding = 0
    padded_sidelength = sidelength + 2 * padding

    # AAPM validation scan_ids: [0,1,2,3,4,5,6,7]
    # AAPM test scan_ids: [0,1,2,3,4,5,6,7]
    # Shepp-Logan validation scan_ids: [1,2,3,4,5]
    # Shepp-Logan test scan_ids: [6,7,8,9,10]
    scan_ids = [args.data.scan_id]
    for scan_id in scan_ids:
        # wandb.init(
        #     name=f"uncertainr_projection_{args.data.name}_{scan_id}_{args.run}",
        #     project="uncertainr"  # Set a short project name
        # )
        if args.data.name == "shepp_2d":
            img_data = dataio.Shepp_2d(scan_id=scan_id)
            img_no_pad = img_data
        elif args.data.name == "aapm_2d":
            img_data = dataio.AAPM_2d(
                scan_id=scan_id,
                test_set=args.data.test_set,
                PADDING=padding,
            )
            img_no_pad = dataio.AAPM_2d(
                scan_id=scan_id, test_set=args.data.test_set, PADDING=0
            )
        else:
            raise ValueError(f"Dataset {args.data.name} not implemented.")

        coord_data = dataio.Implicit2DWrapper(
            img_data, sidelength=padded_sidelength, preload_cuda=True
        )
        zero_pad_mask = dataio.zero_pad_mask(padding, padded_sidelength, coord_data)
        zero_pad_mask = torch.unsqueeze(torch.unsqueeze(zero_pad_mask, 0), 2)
        zero_pad_mask = zero_pad_mask.cuda()

        dataloader = DataLoader(
            coord_data, shuffle=True, batch_size=args.data.batch_size, num_workers=0
        )

        coord_no_pad = dataio.Implicit2DWrapper(
            img_no_pad,
            sidelength=sidelength,
            preload_cuda=True,
        )
        dataloader_no_pad = DataLoader(
            coord_no_pad, shuffle=True, batch_size=args.data.batch_size, num_workers=0
        )

        # Define the model.
        # model_type = args.model.model_type
        # activation_type = args.model.activation_type
        # omega_0 = args.model.omega_0

        # net = partial(modules.UncertaINR, args.uncertainty)

        ###################
        ### TRAIN MODEL ###
        ###################

        ## Ensemble training, args.uncertainty.num_baselearners=1 corresponds to single model
        # bslrs = []
        # for BASELEARN_ID in range(0, args.uncertainty.num_baselearners):
        #     args.seed = int(100 * args.run + BASELEARN_ID)
        #     print(f"Setting seed to {args.seed}.")
        #     pl.seed_everything(args.seed)
        #     if activation_type in [
        #         "sine",
        #         "relu",
        #         "tanh",
        #         "selu",
        #         "elu",
        #         "softplus",
        #         "silu",
        #         "sigmoid",
        #     ]:
        #         if model_type in ["rbf", "nerf", "rff_enc", "mlp"]:
        #             model = net(
        #                 type=activation_type,
        #                 mode=model_type,
        #                 omega_0=omega_0,
        #                 hidden_features=args.model.width,
        #                 out_features=args.data.out_dim,
        #                 num_hidden_layers=args.model.depth,
        #                 in_features=args.data.in_dim,
        #                 outermost_linear=args.model.outerlin,
        #                 embed_width=args.model.embed_width,
        #                 bias=args.model.bias,
        #                 zero_pad=padding,
        #             )
        #         else:
        #             raise NotImplementedError
        #     else:
        #         raise NotImplementedError

        #     model.cuda()

        #     # Define the loss
        #     thetas = np.linspace(0.0, np.pi, args.data.n_views + 1)[:-1]
        #     for data in dataloader:
        #         gt_coords = data[0]["coords"]
        #         gt_img = data[1]["img"]
        #         noise_sigma = args.data.noise_sigma
        #         if args.data.noise_snr_dB not in ["None", 0]:
        #             noise_sigma = loss_functions.get_SNR_stdev(
        #                 args.data.noise_snr_dB, gt_coords, gt_img, list(thetas)
        #             )
        #         loss_fn = loss_functions.single_image_mse_project(
        #             gt_coords,
        #             gt_img,
        #             list(thetas),
        #             args.data.view_batch_size,
        #             projection_length=padded_sidelength,
        #             noise_sigma=noise_sigma,
        #             regularize=args.reg.type,
        #             reg_coeff=args.reg.coeff,
        #         )

        #     model = training.train(
        #         model=model,
        #         train_dataloader=dataloader_no_pad,
        #         epochs=args.num_epochs,
        #         lr=args.opt.lr,
        #         steps_til_summary=args.steps_til_summary,
        #         epochs_til_checkpoint=args.epochs_til_ckpt,
        #         save_last_ckpt=True,
        #         model_dir=root_path,
        #         loss_fn=loss_fn,
        #         opt_cfg=args.opt,
        #         data_train_number=scan_id,
        #         bslr_id=BASELEARN_ID,
        #         SWA_mult=args.model.swa_mult,
        #         zero_pad_mask=zero_pad_mask,
        #     )
        #     model = model.cpu()
        #     bslrs.append(model)
        #     del model
        # model = modules.Ensemble(bslrs)

        #########################
        ### SINOGRAM GENERATION ###
        #########################
        
        # Define projection angles
        thetas = np.linspace(0.0, np.pi, args.data.n_views + 1)[:-1]

        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)

        with torch.no_grad():
            model_input, gt_data = next(iter(dataloader_no_pad))
            
            # Get ground truth image
            gt_coords = model_input["coords"]
            gt_img = gt_data["img"]

            # Generate sinogram using radon transform
            print(f"Generating sinogram for {len(thetas)} projection angles...")
            sinogram_data = []
            
            for i, theta in enumerate(thetas):
                projection = loss_functions.ct_project(gt_coords, gt_img, theta)
                sinogram_data.append(projection.cpu().squeeze())
                if i % 5 == 0:
                    print(f"Processed {i+1}/{len(thetas)} projections")
            
            # Stack projections to form sinogram
            sinogram = torch.stack(sinogram_data, dim=0)  # Shape: [n_views, projection_length]
            
            # Save sinogram data
            sinogram_np = sinogram.numpy()
            np.save(os.path.join(results_dir, f"sinogram_{scan_id}.npy"), sinogram_np)
            
            print(f"Sinogram shape: {sinogram_np.shape}")
            print(f"Sinogram range: [{sinogram_np.min():.4f}, {sinogram_np.max():.4f}]")

        # Save sinogram visualization


        summary_fn = partial(
            utils.write_uncertain_experiment_vals, (sidelength, sidelength)
        )

        summary_fn(
            gt=gt_img,
            model_output={"mean_out": gt_img, "var_out": torch.zeros_like(gt_img)},
            args=args,
            # suffix=f"_scan{scan_id}",
            scan_id=scan_id,
            save_imgs=True,
            save_model_out=False,
        )
        
        # Create sinogram plot
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        gt_img_2d = gt_img.cpu().numpy().reshape(sidelength, sidelength)
        plt.imshow(gt_img_2d, cmap='gray')
        plt.title(f'Original Image (Scan {scan_id})')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(sinogram_np, cmap='gray', aspect='auto')
        plt.title(f'Sinogram ({len(thetas)} projections)')
        plt.xlabel('Detector Position')
        plt.ylabel('Projection Angle')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'sinogram_comparison_{scan_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Sinogram saved: {os.path.join(results_dir, f'sinogram_{scan_id}.npy')}")
        print(f"Visualization saved: {os.path.join(results_dir, f'sinogram_comparison_{scan_id}.png')}")

        torch.cuda.empty_cache()

    print("Sinogram generation completed for all scans!")


if __name__ == "__main__":
    main()
