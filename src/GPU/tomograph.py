import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
import astra
import torch
from skimage.metrics import structural_similarity as ssim
from ALNS.ALNS import ALNS
from RoundToNearest import FindNearest
from alns.stop import *
from DART import *
import time  
import warnings
warnings.filterwarnings("ignore")

def FindNearest(weights: torch.tensor, nQuantized, device: torch.device):
    num_levels = 2 ** nQuantized
    wMin = 0
    wMax = MAX_RANGE
    step = (wMax - wMin) / (2 ** nQuantized - 1)
    quantization_levels = wMin + torch.arange(num_levels, device=device) * step
    indices = torch.abs(weights - quantization_levels[:, None]).argmin(dim=0)
    wq = quantization_levels[indices]
    return wq, indices.int()

def mae(true, pred):
    return np.mean(np.abs(true - pred))

def mse(true, pred):
    return np.mean((true - pred) ** 2)

def rmse(img, ref):
    
    return np.sqrt(mse(img, ref))

def psnr(img, ref, data_range=255.0):
    """Peak-Signal-to-Noise Ratio (dB)."""
    err = mse(img, ref)
    if err == 0:
        return np.inf
    return 20 * np.log10(data_range / np.sqrt(err))

def L_inf(A, x, b):
    residual = A @ x - b
    return np.max(np.abs(residual))

def compute_ssim_flat(flat_img1, flat_img2, data_range=255.0):

    img_shape=(int(np.sqrt(flat_img1.shape[0])), int(np.sqrt(flat_img1.shape[0])))
    img1 = flat_img1.reshape(img_shape)
    img2 = flat_img2.reshape(img_shape)
    return ssim(img1, img2, data_range=data_range)


def save_image(array, path):
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    Image.fromarray(array).save(path)

def run_ALNS(A, p, rec_sart, nQuantized, device):
    x0 = torch.from_numpy(rec_sart).float().flatten().to(device)
    nearWq, nearQ = FindNearest(x0, nQuantized, device)
    W = torch.from_numpy(A).float().to(device)
    y = torch.from_numpy(p).float().to(device)
    alns_obj = ALNS(nearQ, x0, W, nQuantized, debug=False, use_gptq=False)
    alns_obj.set_stopping_criteria(stopping_criteria=MaxIterations(ALNS_ITERS))
    alns_obj.set_LS_operator('S')
    alns_obj.set_B_k(y)
    alns_obj.set_torch_device(device)
    alns_start = time.perf_counter()
    solution = alns_obj.solve()
    alns_time = time.perf_counter() - alns_start
    alns_rec = solution.quantized_weights.detach().cpu().numpy().flatten()
    return alns_rec, alns_time

def process_folder(image_folder, n_angles=8, n_d=512, iters=1000, noise_factor= None):
    results = []
    image_paths = glob.glob(os.path.join(image_folder, '*.png'))
    device = 'cuda'
    nQuantized = NQ
    num_levels = 2 ** nQuantized
    wMin = 0
    wMax = MAX_RANGE
    step = (wMax - wMin) / (2 ** nQuantized - 1)
    gray_levels = wMin + np.arange(num_levels) * step
    print(gray_levels)
    num_images = 0
    for img_path in image_paths:
        num_images +=1
        print(f"Processing {img_path}...", num_images,"/",len(image_paths))
        img = np.array(Image.open(img_path), dtype=np.float32)
        print("images gray levl :", np.unique(img).astype(np.float32))
        Nx, Ny = img.shape
        angles = np.linspace(0, np.pi, n_angles, False)
        proj_geom = astra.create_proj_geom('parallel', 1.0, n_d, angles)
        vol_geom = astra.create_vol_geom(Nx, Ny)
        phantom_id = astra.data2d.create('-vol', vol_geom, data=img)
        proj_id = astra.create_projector('linear', proj_geom, vol_geom)
       # proj_id2 = astra.create_projector('cuda', proj_geom, vol_geom)
        
        matrix_id = astra.projector.matrix(proj_id)
        A = astra.matrix.get(matrix_id)
        A = A.todense()
        A = np.array(A)  # projection matrix
        x_true = img.flatten()
        sino_id, sinogram = astra.creators.create_sino(phantom_id, proj_id)
        if noise_factor != None:
            sinogram += np.random.uniform(low=-noise_factor, high=noise_factor, size=sinogram.shape)
            sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
        p = sinogram.flatten()
        # --- SART ---
        sart_recon_start = time.perf_counter()
        rec_sart = SART(vol_geom, 0, proj_id, sino_id, iters=iters, use_gpu=False)
        sart_total_time = time.perf_counter() - sart_recon_start

        # --- SIRT ---
        sirt_recon_start = time.perf_counter()
        rec_sirt = SIRT(vol_geom, 0, proj_id, sino_id, iters=iters, use_gpu=False)
        sirt_total_time = time.perf_counter() - sirt_recon_start



        # --- DART  ---
        dart_start = time.perf_counter()
        dart = DART(gray_levels=gray_levels, p=0.5, rec_shape=img.shape,
                    proj_geom=proj_geom, projector_id=proj_id, sinogram=sinogram)
        dart.gray_levels = gray_levels
        dart.thresholds = dart.update_gray_thresholds()
        dart_rec = dart.run(iters=DART_ITERS, rec_alg="SART", rec_iter=1000)
        dart_rec = dart.segment(dart_rec)
        dart_time = time.perf_counter() - dart_start

        # --- ALNS ---
        alns_rec, alns_time = run_ALNS(A, p, rec_sart, nQuantized, device)
        cpu_alns_rec, cpu_alns_time = run_ALNS(A, p, rec_sart, nQuantized, "cpu")
        
        print("SIRT : ", len(np.unique(rec_sirt.flatten())))
        print("SART : ", len(np.unique(rec_sart.flatten())))
        print("DART : ", len(np.unique(dart_rec.flatten())))
        print("ALNS : ", len(np.unique(alns_rec)))
        print("ALNS VS GT : ", mae(alns_rec, x_true) )
        print("DART VS GT : ", mae(dart_rec.flatten(), x_true) )
        print("SART VS GT : ", mae(dart.segment(rec_sart).flatten(), x_true))
        # --- Metrics ---

        mae_sart = mae(x_true, rec_sart.flatten())
        mae_sirt = mae(x_true, rec_sirt.flatten())
        mae_dart = mae(x_true, dart_rec.flatten())
        mae_alns = mae(x_true, alns_rec)
        mae_alns_cpu = mae(x_true, cpu_alns_rec)

        psnr_sart = compute_ssim_flat(x_true, rec_sart.flatten(), MAX_RANGE)
        psnr_sirt = compute_ssim_flat(x_true, rec_sirt.flatten(), MAX_RANGE)
        psnr_dart = compute_ssim_flat(x_true, dart_rec.flatten(), MAX_RANGE)
        psnr_alns = compute_ssim_flat(x_true, alns_rec, MAX_RANGE)
        psnr_alns_cpu = compute_ssim_flat(x_true, cpu_alns_rec, MAX_RANGE)
        
        linf_sart = L_inf(A, rec_sart.flatten(), p)
        linf_sirt = L_inf(A, rec_sirt.flatten(), p)
        linf_dart = L_inf(A, dart_rec.flatten(), p)
        linf_alns = L_inf(A, alns_rec, p)
        linf_alns_cpu = L_inf(A, cpu_alns_rec, p)

        results.append({
            "image": os.path.basename(img_path),
             "mae_sart": mae_sart, "mae_sirt": mae_sirt, "mae_dart": mae_dart, "mae_alns": mae_alns,  "mae_alns_cpu": mae_alns_cpu,
            "psnr_sart": psnr_sart, "psnr_sirt": psnr_sirt, "psnr_dart": psnr_dart, "psnr_alns": psnr_alns, "psnr_alns_cpu": psnr_alns_cpu,
            "linf_sart": linf_sart, "linf_sirt": linf_sirt, "linf_dart": linf_dart, "linf_alns": linf_alns, "linf_alns_cpu": linf_alns_cpu,
            "time_sart": sart_total_time, "time_sirt": sirt_total_time, "time_dart": dart_time, "time_alns": alns_time, "time_alns_cpu": cpu_alns_time,
        })

        # Clean up ASTRA objects
        astra.data2d.delete(phantom_id)
        astra.data2d.delete(sino_id)
        astra.projector.delete(proj_id)
        
        basename = os.path.basename(img_path)
        save_image(rec_sart, os.path.join(folders["SART_raw"], basename))
        save_image(rec_sirt, os.path.join(folders["SIRT_raw"], basename))
        save_image(dart_rec, os.path.join(folders["DART"], basename))
        alns_img = alns_rec.reshape(Nx, Ny)
        save_image(alns_img, os.path.join(folders["ALNS"], basename))
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(PATH_SAVE, index=False)
    print(f"Results saved to {PATH_SAVE}")

if __name__ == "__main__":
    image_folder = "./binary_images"    
    output_base = os.path.abspath(image_folder)
    NQ = 1
    MAX_RANGE = 255
    ALNS_ITERS = 10
    DART_ITERS = 100
    PATH_SAVE = "reconstruction_comparison.csv"
    n_angles = 64
    n_det = 128
    iters = 1000
    noise_factor = 1000

    folders = {
        "SART_raw": os.path.join(output_base, "SART_raw"),
        "SIRT_raw": os.path.join(output_base, "SIRT_raw"),
        "DART": os.path.join(output_base, "DART"),
        "ALNS": os.path.join(output_base, "ALNS"),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    process_folder(image_folder, n_angles, n_det, iters, noise_factor)
