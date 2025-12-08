import h5py
import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.special import wofz


# --- Model Functions for Fitting ---

def gaussian(x, amplitude, center, sigma):
    """Gaussian lineshape."""
    return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))


def lorentzian(x, amplitude, center, sigma):
    """Lorentzian lineshape. Sigma corresponds to HWHM."""
    return amplitude * sigma ** 2 / ((x - center) ** 2 + sigma ** 2)


def voigt(x, amplitude, center, sigma, gamma):
    """
    Voigt profile (convolution of Gaussian and Lorentzian).
    sigma: Gaussian standard deviation
    gamma: Lorentzian half-width at half-maximum (HWHM)
    """
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * wofz(z).real / (sigma * np.sqrt(2 * np.pi))


def pseudo_voigt(x, amplitude, center, sigma, fraction):
    """
    Pseudo-Voigt profile (linear combination of Gaussian and Lorentzian).
    sigma: HWHM (approximated for Gaussian part to match Lorentzian width definition if needed,
           but here typically we treat it as a width parameter)
    fraction: Lorentz fraction (0 to 1)
    """
    # Note: To consistently compare, sigma in both should represent similar width metric.
    # Usually Pseudo-Voigt is defined with FWHM. Here we stick to a simple mixing.
    return fraction * lorentzian(x, amplitude, center, sigma) + \
        (1 - fraction) * gaussian(x, amplitude, center, sigma)


def background_constant(x, c):
    return c


def background_linear(x, m, c):
    return m * x + c


# --- Main GID Class ---

class GID:
    def __init__(self, file, scans, alpha_i_name='chi', detector_name='mythen2', monitor_name='mon',
                 transmission_name='autof_eh1_transm', att_name='autof_eh1_curratt', cnttime_name='sec',
                 PX0=50, mythen_gap=120, PPD=198.5, pixel_size_qxz=0.055, angle_name='delta', energy_name='monoe',
                 I0=1e12, *args, **kwargs):
        self.file = file
        self.scans = np.array(scans)
        self.alpha_i_name = alpha_i_name
        self.detector_name = detector_name
        self.monitor_name = monitor_name
        self.transmission_name = transmission_name
        self.att_name = att_name
        self.energy_name = energy_name
        self.cnttime_name = cnttime_name
        self.angle_name = angle_name

        self.PX0 = PX0
        self.mythen_gap = mythen_gap
        self.PPD = PPD
        self.I0 = I0

        # Initialize attributes to store data
        self.data = np.empty((0, 0))  # Placeholder
        self.angle = np.array([])
        self.alpha_i = np.array([])
        self.monitor = np.array([])
        self.transmission = np.array([])
        self.attenuator = np.array([])
        self.cnttime = np.array([])
        self.energy = 0.0
        self.sample_name = ""
        self.Pi = ''

        # Processed data containers
        self.data_gap = None
        self.data_gap_e = None
        self.qz = None
        self.qxy = None
        self.Qz_map = None
        self.Qx_map = None

        self.__load_data__()
        self.__process_2D_data__()

    def __load_single_scan__(self, ScanN):
        print('Loading scan #{}'.format(ScanN))
        try:
            with h5py.File(self.file, "r") as f:
                # Using [()] to read dataset into numpy array immediately
                self.data = f.get(f"{ScanN}.1/measurement/{self.detector_name}")[()]

                self.angle = f.get(f"{ScanN}.1/measurement/{self.angle_name}")[()]
                self.alpha_i = f.get(f"{ScanN}.1/instrument/positioners/{self.alpha_i_name}")[()]
                self.monitor = f.get(f"{ScanN}.1/measurement/{self.monitor_name}")[()]
                self.transmission = f.get(f"{ScanN}.1/measurement/{self.transmission_name}")[()]
                self.attenuator = f.get(f"{ScanN}.1/measurement/{self.att_name}")[()]
                self.cnttime = f.get(f"{ScanN}.1/measurement/{self.cnttime_name}")[()]

                energy = f.get(f"{ScanN}.1/instrument/positioners/{self.energy_name}")[()]
                self.energy = float(energy)

                sample_name_ds = f.get(f"{ScanN}.1/sample/name/")
                if sample_name_ds:
                    # Handle string decoding if necessary
                    self.sample_name = str(sample_name_ds[()])[2:-1:1]

                pi_ds = f.get(f"{ScanN}.1/measurement/fb_Pi")
                if pi_ds:
                    Pi = np.mean(pi_ds[()])
                    if Pi < 90:
                        self.Pi = int(np.round(Pi, 0))
                    else:
                        self.Pi = ''
        except Exception as e:
            print(f"Error loading scan {ScanN}: {e}")
            raise

        print('Loaded scan #{}'.format(ScanN))

    def __load_data__(self, skip_points=1):
        t0 = time.time()
        print("Start loading data.")

        if len(self.scans) == 1:
            ScanN = str(self.scans[0])
            self.__load_single_scan__(ScanN)
        else:
            # Load first scan to initialize arrays
            first_ScanN = str(self.scans[0])
            self.__load_single_scan__(first_ScanN)

            # Append subsequent scans
            with h5py.File(self.file, "r") as f:
                for each in self.scans[1:]:
                    ScanN = str(each)
                    print(f'Loading scan {ScanN}')

                    try:
                        data = f.get(f"{ScanN}.1/measurement/{self.detector_name}")[skip_points:]
                        self.data = np.append(self.data, data, axis=0)

                        data_x = f.get(f"{ScanN}.1/measurement/{self.angle_name}")[
                            ()]  # Assuming angle matches data points, adjusting skip_points if needed
                        # Note: original code didn't slice data_x with skip_points, but did for others.
                        # Assuming data_x (angle) aligns with detector data.
                        # If data_x has same length as data, it should be sliced too?
                        # Original code: self.angle = np.append(self.angle, data_x) (no slicing)
                        # But typically measurement arrays are same length.
                        # I will keep original behavior but it looks suspicious if skip_points > 0
                        self.angle = np.append(self.angle, data_x)

                        data_mon = f.get(f"{ScanN}.1/measurement/{self.monitor_name}")[skip_points:]
                        self.monitor = np.append(self.monitor, data_mon)

                        data_transm = f.get(f"{ScanN}.1/measurement/{self.transmission_name}")[skip_points:]
                        self.transmission = np.append(self.transmission, data_transm)

                        data_att = f.get(f"{ScanN}.1/measurement/{self.att_name}")[skip_points:]
                        self.attenuator = np.append(self.attenuator, data_att)

                        cnttime = f.get(f"{ScanN}.1/measurement/{self.cnttime_name}")[skip_points:]
                        self.cnttime = np.append(self.cnttime, cnttime)

                        print(f'Loaded scan #{ScanN}')
                    except Exception as e:
                        print(f"Error appending scan {ScanN}: {e}")

        print("Loading completed. Reading time %3.3f sec" % (time.time() - t0))

    def get_qz(self, pixels):
        # Calculate qz. Assuming alpha_i is scalar or matches dimensions if it varies.
        # Here we treat alpha_i as a scalar (mean value) if it's an array to produce a 1D qz array for pixels.
        alpha_i = np.mean(self.alpha_i) if np.size(self.alpha_i) > 1 else self.alpha_i

        wavelength = 12.398 / self.energy
        k0 = 2 * np.pi / wavelength

        # pixels can be an array
        qz = k0 * (np.sin(np.deg2rad(alpha_i)) + np.sin(np.deg2rad((pixels - self.PX0) / self.PPD)))
        return qz

    def get_qxy(self, angle):
        wavelength = 12.398 / self.energy
        k0 = 2 * np.pi / wavelength
        qxy = 2 * k0 * np.sin(np.deg2rad(angle / 2))
        return qxy

    def __process_2D_data__(self):
        t0 = time.time()
        print("Start processing 2D data.")
        nx, ny = np.shape(self.data)

        # Handle gaps in detector modules (Mythen detector specific)
        # Assuming data is split into two halves with a gap
        map2Dm = np.ones((nx, ny + self.mythen_gap))

        # Specific slicing for Mythen detector (1280 pixels per module?)
        # Original code had hardcoded indices: 0:1279 and 1280:2559
        # 1280 is module size.
        if ny >= 2559:
            map2Dm[:, 0:1279] = self.data[:, 0:1279]
            map2Dm[:, (1280 + self.mythen_gap):(2559 + self.mythen_gap)] = self.data[:, 1280:2559]
        else:
            # Fallback if dimensions don't match expectation
            print(f"Warning: Data shape {self.data.shape} does not match expected Mythen format. Using raw data.")
            map2Dm = self.data

        nxm, nym = np.shape(map2Dm)

        # Normalize by monitor and transmission
        # Ensure monitor and transmission are properly broadcasted
        # self.monitor is 1D (nx,), map2Dm is 2D (nx, nym)

        norm_factor = self.transmission * self.monitor / self.monitor[0]
        # Avoid division by zero
        norm_factor = np.where(norm_factor == 0, 1.0, norm_factor)

        self.data_gap = map2Dm / norm_factor[:, np.newaxis] / self.I0
        self.data_gap_e = np.sqrt(map2Dm) / norm_factor[:, np.newaxis] / self.I0

        self.qz = self.get_qz(np.arange(nym))
        self.qxy = self.get_qxy(self.angle)

        print("Processing completed. Processing time %3.3f sec \n\n" % (time.time() - t0))

    def plot_2D_image(self, ax=None, save=False, **kwargs):
        if ax is None:
            fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), layout='tight')
        else:
            ax0 = ax

        # Determine limits for better contrast
        mean_val = np.mean(self.data_gap)
        std_val = np.std(self.data_gap)
        _vmin = np.log10(max(mean_val - std_val, 1e-12))  # Avoid log of negative/zero
        _vmax = np.log10(mean_val + 3 * std_val)

        # extent=[xmin, xmax, ymin, ymax]
        # data_gap is (angles, pixels). Rot90 rotates it.
        # qxy corresponds to angles (rows originally), qz corresponds to pixels (cols originally).
        # np.rot90(m, k=1) rotates 90 degrees counter-clockwise.
        # Original: rows=angles(x), cols=pixels(z).
        # Rotated: rows=pixels(z), cols=angles(x) (reversed).

        # Let's check extent logic from original code:
        # extent=[np.min(self.qxy),np.max(self.qxy),np.min(self.qz),np.max(self.qz)]
        # This implies x-axis is qxy, y-axis is qz.

        im = ax0.imshow(np.log10(np.rot90(self.data_gap)), aspect='auto', vmin=_vmin, vmax=_vmax,
                        extent=[np.min(self.qxy), np.max(self.qxy), np.min(self.qz), np.max(self.qz)], **kwargs)

        ax0.set_xlabel('$q_{xy}, \\AA^{-1}$')
        ax0.set_ylabel('$q_{z}, \\AA^{-1}$')

        if save:
            print('Saving 2D GID map.')
            self._save_figure(plt.gcf(), '2Dmap')

        return ax0

    def get_qxy_cut(self, qz_min, qz_max):
        # Find indices for qz range
        qz_indices = np.where((self.qz > qz_min) & (self.qz < qz_max))[0]
        if len(qz_indices) == 0:
            print(f"Warning: No data in qz range [{qz_min}, {qz_max}]")
            return [self.qxy, np.zeros_like(self.qxy)]

        cut_qxy = np.sum(self.data_gap[:, qz_indices], axis=1)
        return [self.qxy, cut_qxy]

    def save_qxy_cut(self, qz_min, qz_max, **kwargs):
        qxy, cut_qxy = self.get_qxy_cut(qz_min, qz_max)
        out = np.array([qxy, cut_qxy])

        self._ensure_sample_dir()

        filename = './{}/GID_{}_scan_{}_qxy_cut_{}_{}_A.dat'.format(
            self.sample_name, self.sample_name, self.scans, qz_min, qz_max)

        np.savetxt(filename, out.T)
        print('GID cut saved as: {}'.format(filename))

    def plot_qxy_cut(self, qz_min, qz_max, ax=None, save=False, **kwargs):
        if ax is None:
            fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), layout='tight')
        else:
            ax0 = ax

        qxy, cut_qxy = self.get_qxy_cut(qz_min, qz_max)

        ax0.plot(qxy, cut_qxy, 'o', markersize=5, label='$Cut\\: {:.2f}<q_z<{:.2f}$'.format(qz_min, qz_max))
        ax0.set_xlabel('$q_{xy}, \\AA^{-1}$')
        ax0.set_ylabel('Intensity')
        ax0.legend()

        if save:
            print('Saving qxy cut plot.')
            self._save_figure(plt.gcf(), f'qxy_cut_{qz_min}_{qz_max}_A')

    def get_qz_cut(self, qxy_min, qxy_max):
        qxy_indices = np.where((self.qxy > qxy_min) & (self.qxy < qxy_max))[0]
        if len(qxy_indices) == 0:
            print(f"Warning: No data in qxy range [{qxy_min}, {qxy_max}]")
            return [self.qz, np.zeros_like(self.qz)]

        cut_qz = np.sum(self.data_gap[qxy_indices, :], axis=0)
        return [self.qz, cut_qz]

    def save_qz_cut(self, qxy_min, qxy_max, **kwargs):
        qz, cut_qz = self.get_qz_cut(qxy_min, qxy_max)
        out = np.array([qz, cut_qz])

        self._ensure_sample_dir()

        filename = './{}/GID_{}_scan_{}_qz_cut_{}_{}_A.dat'.format(
            self.sample_name, self.sample_name, self.scans, qxy_min, qxy_max)

        np.savetxt(filename, out.T)
        print('GID cut saved as: {}'.format(filename))

    def plot_qz_cut(self, qxy_min, qxy_max, ax=None, save=False, **kwargs):
        if ax is None:
            fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), layout='tight')
        else:
            ax0 = ax

        qz, cut_qz = self.get_qz_cut(qxy_min, qxy_max)

        ax0.plot(qz, cut_qz, '.', label='$Cut\\: {:.1f}<q_{{xy}}<{:.1f}$'.format(qxy_min, qxy_max))
        ax0.set_xlabel('$q_{z}, \\AA^{-1}$')
        ax0.set_ylabel('Intensity')
        ax0.legend()

        if save:
            print('Saving qz cut plot.')
            self._save_figure(plt.gcf(), f'qz_cut_{qxy_min}_{qxy_max}_A')

    def plot_quick_analysis(self, save=False):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), layout='tight')

        self.plot_2D_image(ax=ax[0])

        # Default limits, might need adjustment based on data
        ax[0].set_ylim(0, 1.5)
        ax[0].set_xlim(np.min(self.qxy), np.max(self.qxy))

        # Example lines, these should probably be configurable
        # ax[0].hlines([0.01, 0.3], 1.2, 1.6, linestyle='--', alpha=0.8, color='C0')
        # ax[0].hlines([0.5, 0.95], 1.2, 1.6, linestyle='--', alpha=0.8, color='C1')

        self.plot_qxy_cut(0.01, 0.3, ax=ax[1])
        self.plot_qxy_cut(0.5, 0.95, ax=ax[1])

        plt.suptitle('GID : Sample {}, Scan {}, Pi = {} mN/m'.format(self.sample_name, self.scans, self.Pi))

        if save:
            print('Saving standard GID plot.')
            self._save_figure(fig, 'quick_analysis')

    def _ensure_sample_dir(self):
        try:
            os.mkdir(self.sample_name)
        except OSError:
            pass  # Directory likely exists

    def _save_figure(self, fig, suffix):
        self._ensure_sample_dir()
        filename = './{}/GID_{}_scan_{}_{}.png'.format(
            self.sample_name, self.sample_name, self.scans, suffix)
        fig.savefig(filename, dpi=200)

    # --- New Features ---

    def fit_profile(self, x, y, model='gaussian', background='linear', limits=None, p0=None):
        """
        Fit a profile to the specified model with background.

        Parameters:
        - x, y: data arrays
        - model: 'gaussian', 'lorentzian', 'voigt', 'pseudo_voigt'
        - background: 'constant', 'linear', None
        - limits: tuple (min, max) to restrict fitting range
        - p0: initial guess for parameters [amp, center, width, (gamma/fraction), (bg_params...)]

        Returns:
        - popt: optimal parameters
        - pcov: covariance of parameters
        - fit_func: the callable function used for fitting
        """

        # Apply limits
        if limits:
            mask = (x >= limits[0]) & (x <= limits[1])
            x_fit = x[mask]
            y_fit = y[mask]
        else:
            x_fit = x
            y_fit = y

        # Construct model function
        def fit_func(x, *args):
            # Unpack parameters based on model choice
            # Basic params: amp, cen, wid
            current_idx = 0

            amp = args[current_idx];
            current_idx += 1
            cen = args[current_idx];
            current_idx += 1
            wid = args[current_idx];
            current_idx += 1

            extra_param = None
            if model in ['voigt', 'pseudo_voigt']:
                extra_param = args[current_idx];
                current_idx += 1

            # Calculate signal
            if model == 'gaussian':
                y_sig = gaussian(x, amp, cen, wid)
            elif model == 'lorentzian':
                y_sig = lorentzian(x, amp, cen, wid)
            elif model == 'voigt':
                y_sig = voigt(x, amp, cen, wid, extra_param)
            elif model == 'pseudo_voigt':
                y_sig = pseudo_voigt(x, amp, cen, wid, extra_param)
            else:
                raise ValueError(f"Unknown model: {model}")

            # Calculate background
            y_bg = 0
            if background == 'constant':
                c = args[current_idx]
                y_bg = background_constant(x, c)
            elif background == 'linear':
                m = args[current_idx];
                current_idx += 1
                c = args[current_idx]
                y_bg = background_linear(x, m, c)

            return y_sig + y_bg

        # Initial guess generation if not provided
        if p0 is None:
            p0 = []
            # Signal guess
            amp_guess = np.max(y_fit) - np.min(y_fit)
            cen_guess = x_fit[np.argmax(y_fit)]
            wid_guess = (np.max(x_fit) - np.min(x_fit)) / 10.0  # Rough guess
            p0.extend([amp_guess, cen_guess, wid_guess])

            if model == 'voigt':
                p0.append(wid_guess / 2.0)  # Gamma guess
            elif model == 'pseudo_voigt':
                p0.append(0.5)  # Fraction guess

            # Background guess
            if background == 'constant':
                p0.append(np.min(y_fit))
            elif background == 'linear':
                slope = (y_fit[-1] - y_fit[0]) / (x_fit[-1] - x_fit[0])
                intercept = y_fit[0] - slope * x_fit[0]
                p0.extend([slope, intercept])

        try:
            popt, pcov = curve_fit(fit_func, x_fit, y_fit, p0=p0)
            return popt, pcov, fit_func, x_fit, y_fit
        except RuntimeError as e:
            print(f"Fitting failed: {e}")
            return None, None, None, x_fit, y_fit

    def save_image_h5(self, filename=None):
        """
        Save the 2D image data to an HDF5 file in q-coordinates.
        Saves: Intensity map, Qxy axis, Qz axis.
        """
        if filename is None:
            self._ensure_sample_dir()
            filename = './{}/GID_{}_scan_{}_2D.h5'.format(
                self.sample_name, self.sample_name, self.scans)

        try:
            with h5py.File(filename, 'w') as hf:
                hf.create_dataset('intensity', data=self.data_gap)
                hf.create_dataset('qxy', data=self.qxy)
                hf.create_dataset('qz', data=self.qz)

                # Add metadata
                hf.attrs['sample_name'] = self.sample_name
                hf.attrs['scans'] = self.scans
                hf.attrs['energy'] = self.energy
            print(f"2D image saved to {filename}")
        except Exception as e:
            print(f"Error saving H5 file: {e}")