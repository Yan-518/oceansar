import numpy as np
from oceansar import ocs_io as tpio
from oceansar import constants as const
from oceansar.utils import geometry as geo
#create a function
def raw_reconstr(raw_output_file, reconstr_output_file, channel_ids = [0, 1, 2, 3, 4]):
    # Load config and raw data
    raw_file = tpio.RawFile(raw_output_file, 'r')
    raw_data = raw_file.get('raw_data*')  # Shape: [num_bursts, num_ch, az_size, rg_size]
    sr0 = raw_file.get('sr0')
    az0 = raw_file.get('az0')
    inc_angle = raw_file.get('inc_angle')
    b_ati = raw_file.get('b_ati')
    ant_L = raw_file.get('ant_L')

    print(f"Raw data shape: {raw_data.shape}")

    # Extract parameters
    # Load the processed data
    f0 = raw_file.get('f0')
    prf = raw_file.get('prf')
    num_ch = raw_file.get('num_ch')
    rg_bw = raw_file.get('rg_bw')
    rg_sampling = raw_file.get('rg_sampling')
    v_ground = raw_file.get('v_ground')
    alt = raw_file.get('orbit_alt')

    v_orbit = geo.orbit_to_vel(alt, inc=np.deg2rad(inc_angle))

    l0 = const.c / f0

    print(f"v_orbit: {v_orbit}, prf: {prf}")

    # let's start from following the paper
    b_ati = b_ati[channel_ids]
    raw_data = raw_data[:, channel_ids, :, :]
    N_ch = raw_data.shape[1] # number of channels
    N_az = raw_data.shape[2]
    N_rg = raw_data.shape[3]
    # # construct transfer function in frequency domain, eq13 (Krieger et al, 2004)
    # # let's start from following the paper
    f_dop = np.fft.fftshift(np.fft.fftfreq(raw_data.shape[2], d=1./prf))
    # f_matrix = f_dop[:, None] + np.arange(int(-N_ch/2), int(N_ch/2)+1) * prf # (az_size * prf_band * N_ch)
    k_start = np.ceil(-N_ch / 2 - f_dop / prf).astype(int)
    k_alias = k_start[:, None] + np.arange(N_ch)[None, :]
    f_matrix = f_dop[:, None] + k_alias * prf
    H_vec = np.exp(-1j * np.pi * (b_ati**2 / (2 * l0 * sr0) + b_ati * f_matrix[:,:, None] / v_orbit)) 
    # H_vec = np.exp(-1j * (v_ground/v_orbit) *np.pi * (b_ati**2 / (2 * l0 * sr0) + b_ati * f_matrix[:,:, None] / v_orbit)) 
    P_vec = N_ch * np.linalg.inv(H_vec)
    raw_data_fft = np.fft.fftshift(np.fft.fft(raw_data[0, :, :, :], axis=1), axes = 1) # FFT along azimuth
    reconstr_signal = np.einsum('car,acb->bar', raw_data_fft, P_vec)
    # upsample_signal = N_ch * np.fft.ifft(np.fft.ifftshift(reconstr_signal.reshape(N_ch * raw_data.shape[2], raw_data.shape[3]), axes = 0),axis = 0)
    N_out = N_ch * N_az
    df = prf / N_az

    f_out = np.fft.fftshift(
        np.fft.fftfreq(N_out, d=1.0 / (N_ch * prf))
    )

    # Map each recovered component to its actual output frequency bin
    indices = np.rint(
        (f_matrix - f_out[0]) / df
    ).astype(int)

    assert np.all((indices >= 0) & (indices < N_out)), (
        "Some reconstructed frequencies lie outside the output grid."
    )

    assert np.unique(indices).size == N_out, (
        "Frequency mapping is not one-to-one; check alias selection."
    )

    full_spectrum = np.zeros((N_out, N_rg), dtype=complex)

    for az_idx in range(N_az):
        for band_idx in range(N_ch):
            out_idx = indices[az_idx, band_idx]
            full_spectrum[out_idx, :] = reconstr_signal[band_idx, az_idx, :]

    # Convert to the azimuth domain
    upsample_signal = np.fft.ifft(np.fft.ifftshift(full_spectrum, axes=0), axis=0)
    # add the dimension of polarization
    upsample_signal = upsample_signal[None, :, :]
    print('Upsampling completed!')
    # save the recontructed raw data to a new file
    # better to put together with the raw_data to reduce the volume in the future
    reconstr_file = tpio.ReconstructedRawFile(reconstr_output_file, 'w', upsample_signal.shape)
    reconstr_file.set('inc_angle', inc_angle)
    reconstr_file.set('f0', f0)
    reconstr_file.set('num_ch', num_ch)
    reconstr_file.set('ant_L', ant_L)
    reconstr_file.set('prf', prf)
    reconstr_file.set('v_ground', v_ground)
    reconstr_file.set('az0', az0)
    reconstr_file.set('orbit_alt', alt)
    reconstr_file.set('sr0', sr0)
    reconstr_file.set('rg_sampling', rg_sampling)
    reconstr_file.set('rg_bw', rg_bw)
    reconstr_file.set('raw_data*', upsample_signal)
    reconstr_file.close()
