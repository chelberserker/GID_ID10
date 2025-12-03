import h5py
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from math import sin, cos, pi
import scipy as sc
import copy


class GID:
    def __init__(self, file, scans, alpha_i_name='chi', detector_name='mythen2', monitor_name='mon',
                 transmission_name='autof_eh1_transm', att_name='autof_eh1_curratt', cnttime_name='sec',
                 PX0=50, mythen_gap=120, PPD = 198.5, pixel_size_qxz=0.055, angle_name = 'delta', energy_name='monoe', I0=1e12, *args,**kwargs):
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

        self.__load_data__()
        self.__process_2D_data__()
    def __load_single_scan__(self, ScanN):
        print('Loading scan #{}'.format(ScanN))
        f = h5py.File(self.file, "r")
        self.data = f.get(ScanN + '.1/measurement/' + self.detector_name)

        data_x = f.get(ScanN + '.1' + '/measurement/' + self.angle_name)
        self.angle = np.array(data_x)

        data_a_i = f.get(ScanN + '.1' + '/instrument/positioners/' + self.alpha_i_name)
        self.alpha_i = np.array(data_a_i)

        data_mon = f.get(ScanN + '.1' + '/measurement/' + self.monitor_name)
        self.monitor = np.array(data_mon)

        data_transm = f.get(ScanN + '.1' + '/measurement/' + self.transmission_name)
        self.transmission = np.array(data_transm)

        data_att = f.get(ScanN + '.1' + '/measurement/' + self.att_name)
        self.attenuator = np.array(data_att)

        cnttime = f.get(ScanN + '.1' + '/measurement/' + self.cnttime_name)
        self.cnttime = np.array(cnttime)

        energy = f.get(ScanN + '.1' + '/instrument/positioners/' + self.energy_name)
        self.energy = float(np.array(energy))

        self.sample_name = str(f.get(ScanN + '.1' + '/sample/name/')[()])[2:-1:1]

        Pi = np.mean(f.get(ScanN + '.1' + '/measurement/' + 'fb_Pi'))
        if Pi<90:
            self.Pi = int(np.round(Pi, 0))
        else:
            self.Pi = ''

        print('Loaded scan #{}'.format(ScanN))

    def __load_data__(self, skip_points=1):
        t0 = time.time()
        print("Start loading data.")
        if len(self.scans) == 1:
            ScanN = str(self.scans[0])
            self.__load_single_scan__(ScanN)
        else:
            first_ScanN = str(self.scans[0])
            self.__load_single_scan__(first_ScanN)
            for each in self.scans[1:]:
                print('Loading scan ', each)
                f = h5py.File(self.file, "r")
                ScanN = str(each)

                data = f.get(ScanN + '.1/measurement/' + self.detector_name)[skip_points:]
                self.data = np.append(self.data, data, axis=0)

                data_x = f.get(ScanN + '.1' + '/measurement/' + self.angle_name)
                self.angle = np.append(self.angle, data_x)

                data_mon = f.get(ScanN + '.1' + '/measurement/' + self.monitor_name)[skip_points:]
                self.monitor = np.append(self.monitor, data_mon)

                data_transm = f.get(ScanN + '.1' + '/measurement/' + self.transmission_name)[skip_points:]
                self.transmission = np.append(self.transmission, data_transm)

                data_att = f.get(ScanN + '.1' + '/measurement/' + self.att_name)[skip_points:]
                self.attenuator = np.append(self.attenuator, data_att)

                cnttime = f.get(ScanN + '.1' + '/measurement/' + self.cnttime_name)[skip_points:]
                self.cnttime = np.append(self.cnttime, cnttime)


                print('Loaded scan #{}'.format(ScanN))

        print("Loading completed. Reading time %3.3f sec" % (time.time() - t0))

    @staticmethod
    def get_qz(self, pixels):
        qz = 2 * np.pi * (np.sin(np.deg2rad(self.alpha_i)) + np.sin(np.deg2rad((pixels - self.PX0) / self.PPD))) / (12.398 / self.energy)
        return qz
    @staticmethod
    def get_qxy(self, angle):
        qxy = 4 * np.pi * np.sin(np.deg2rad(angle / 2)) / (12.398 / self.energy)
        return qxy

    def __process_2D_data__(self):
        t0 = time.time()
        print("Start processing 2D data.")
        nx, ny = np.shape(self.data)
        map2Dm = np.ones((nx, ny + self.mythen_gap))
        nxm, nym = np.shape(map2Dm)
        map2Dm[:, 0:1279] = self.data[:, 0:1279]
        map2Dm[:, (1280 + self.mythen_gap):(2559 + self.mythen_gap)] = self.data[:, 1280:2559]

        self.data_gap = map2Dm/np.outer(self.transmission*self.monitor/self.monitor[0], np.ones(nym))/self.I0
        self.data_gap_e = np.sqrt(map2Dm)/np.outer(self.transmission*self.monitor/self.monitor[0], np.ones(nym))/self.I0

        self.qz = self.get_qz(self,np.arange(nym))
        self.qxy = self.get_qxy(self, self.angle)

        print("Processing completed. Processing time %3.3f sec \n\n" % (time.time() - t0))

    def plot_2D_image(self, ax=None, save=False, **kwargs):
        if ax is None:
            fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), layout='tight')
        else:
            ax0 = ax

        _vmin = np.log10(np.mean(self.data_gap)-np.std(self.data_gap))
        _vmax = np.log10(np.mean(self.data_gap)+3*np.std(self.data_gap))
        ax0.imshow(np.log10(np.rot90(self.data_gap)), aspect='equal', vmin=_vmin, vmax=_vmax,
                    extent=[np.min(self.qxy),np.max(self.qxy),np.min(self.qz),np.max(self.qz)], **kwargs)
        ax0.set_xlabel('$q_{xy}, \\AA^{-1}$')
        ax0.set_ylabel('$q_{z}, \\AA^{-1}$')

        if save:
            print('Saving 2D GID map.')
            try:
                os.mkdir(self.sample_name)
            except:
                pass
            plt.savefig('./{}/GID_{}_scan_{}_2Dmap.png'.format(self.sample_name,self.sample_name, self.scans), dpi=200)

    def get_qxy_cut(self, qz_min, qz_max):
        cut_qxy = np.sum(self.data_gap[:,np.min(np.where(self.qz>qz_min)):np.max(np.where(self.qz<qz_max))], axis=1)
        return [self.qxy, cut_qxy]

    def save_qxy_cut(self, qz_min, qz_max, **kwargs):
        qxy, cut_qxy = self.get_qxy_cut(qz_min, qz_max)
        out = np.array([qxy, cut_qxy])
        try:
            os.mkdir(self.sample_name)
        except:
            pass
        filename = './{}/GID_{}_scan_{}_qxy_cut_{}_{}_A.dat'.format(self.sample_name, self.sample_name, self.scans,
                                                                   qz_min, qz_max)

        np.savetxt(filename, out.T)

        print('GID cut saved as: {}'.format(filename))

    def plot_qxy_cut(self,qz_min, qz_max, ax=None, save= False, **kwargs):
        if ax is None:
            fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), layout='tight')
        else:
            ax0 = ax

        qxy, cut_qxy = self.get_qxy_cut(qz_min, qz_max)

        ax0.plot(qxy.T, cut_qxy, 'o', markersize=5, label = '$Cut\\: {:.2f}<q_z<{:.2f}$'.format( qz_min, qz_max))
        ax0.set_xlabel('$q_{xy}, \\AA^{-1}$')
        ax0.set_ylabel('Intensity')
        ax0.legend()

        if save:
            print('Saving qxy cut plot.')
            try:
                os.mkdir(self.sample_name)
            except:
                pass
            plt.savefig(
                './{}/GID_{}_scan_{}_qxy_cut_{}_{}_A.png'.format(self.sample_name, self.sample_name, self.scans, qz_min,
                                                                qz_max), dpi=200)

    def get_qz_cut(self, qxy_min, qxy_max):
        cut_qz = np.sum(self.data_gap[np.min(np.where(self.qxy>qxy_min)):np.max(np.where(self.qxy<qxy_max)),:], axis=0)
        return [self.qz, cut_qz]

    def save_qz_cut(self, qxy_min, qxy_max, **kwargs):
        qz, cut_qz = self.get_qz_cut(qxy_min, qxy_max)
        out = np.array([qz, cut_qz])
        try:
            os.mkdir(self.sample_name)
        except:
            pass
        filename = './{}/GID_{}_scan_{}_qz_cut_{}_{}_A.dat'.format(self.sample_name, self.sample_name, self.scans, qxy_min, qxy_max)

        np.savetxt(filename, out.T)

        print('GID cut saved as: {}'.format(filename))


    def plot_qz_cut(self,qxy_min, qxy_max, ax=None, save = False, **kwargs):
        if ax is None:
            fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), layout='tight')
        else:
            ax0 = ax

        qz, cut_qz = self.get_qz_cut(qxy_min, qxy_max)

        ax0.plot(qz.T, cut_qz, '.', label = '$Cut\\: {:.1f}<q_{}<{:.1f}$'.format(qxy_min, 'xy', qxy_max))
        ax0.set_xlabel('$q_{z}, \\AA^{-1}$')
        ax0.set_ylabel('Intensity')
        ax0.legend()

        if save:
            print('Saving qz cut plot.')
            try:
                os.mkdir(self.sample_name)
            except:
                pass
            plt.savefig('./{}/GID_{}_scan_{}_qz_cut_{}_{}_A.png'.format(self.sample_name,self.sample_name, self.scans, qxy_min, qxy_max), dpi=200)


    def plot_quick_analysis(self, save=False):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 6), layout='tight')

        self.plot_2D_image(ax=ax[0])

        ax[0].set_ylim(0, 1)
        ax[0].set_xlim(1.2, 1.6)

        ax[0].hlines([0.01, 0.3], 1.2, 1.6, linestyle='--', alpha=0.8, color='C0')
        ax[0].hlines([0.5, 0.95], 1.2, 1.6, linestyle='--', alpha=0.8, color='C1')

        self.plot_qxy_cut(0.01, 0.3, ax=ax[1])
        self.plot_qxy_cut(0.5, 0.95, ax=ax[1])

        plt.suptitle('GID : Sample {}, Scan {}, Pi = {} mN/m'.format(self.sample_name, self.scans, self.Pi))

        if save:
            print('Saving standard GID plot.')
            try:
                os.mkdir(self.sample_name)
            except:
                pass
            plt.savefig('./{}/GID_{}_scan_{}.png'.format(self.sample_name,self.sample_name, self.scans), dpi=200)





