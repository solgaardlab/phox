from tqdm import tqdm
import time
import numpy as np
import os
import pickle
import pdb

"""
class initialization: 
chip: AFM chip handle
backward: whether it is backward or not

usage:
"reconstruct_field" function output a normalized
reconstructed field with 4 dimensions, phase reference
is the 5th output port

Ground truth:
if set input to v, matrix to u (with gamma)
for forward pass, GT field is normalize[(u @ v) * np.exp(-1j * gamma)]
for backward pass, GT field is normalize[u.T @ (v * np.exp(-1j * gamma))]
"""

class Coherent_meas():
    def __init__(self, chip, backward = False):
        self.chip = chip
        self.backward = backward
        self.out_mzi_idx_forward = {10:4,11:4,12:2,13:3,14:2,15:2,16:1,17:1}
        self.out_mzi_idx_backward = {1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4}
        
    def get_phi_down(self, powers):
        """
        get phase measurement result for single phase sensor
        """
        pu = np.arctan2((powers[1,0] - powers[3,0]), \
                              (powers[0,0] - powers[2,0])) - np.pi
        pd = np.arctan2((powers[1,1] - powers[3,1]), \
                              (powers[0,1] - powers[2,1])) - np.pi
        return pd
    
    def set_bar(self, c):
        """
        set output MZI at column c to bar state
        """
        if self.backward:
            row = self.out_mzi_idx_backward[c]
        else:
            row = self.out_mzi_idx_forward[c]
        self.chip.ps[(c, row)].phase = np.pi
    
    def get_input_cmeas(self, c, avg_num):
        """
        measure input power at column c
        """
        ## set all columns after c to bar state
        if self.backward:
            for cc in np.arange(1, c):
                self.set_bar(cc)
        else:
            for cc in np.arange(c, 18):
                self.set_bar(cc)
        time.sleep(0.05)

        ## measure power at end of mesh
        inp_power = np.zeros(6)
        for ii in range(avg_num):
            if self.backward:
                inp_power += self.chip.fractional_left
            else:
                inp_power += self.chip.fractional_right
        inp_power /= avg_num
        return inp_power


    def config_layer(self, col, avg_num=1):
        """
        nullify layer (col, col+1). The MZI starts at left side of col,
        ends at left side of col+2

        avg_num: each power measurement can be repeated several times
        and averaged to reduce noise, not needed if signal is strong
        """
        if self.backward:
            col1 = col
            col2 = col-1
            row1 = self.out_mzi_idx_backward[col1]
            row2 = self.out_mzi_idx_backward[col2]
        else:
            col1 = col
            col2 = col+1
            row1 = self.out_mzi_idx_forward[col1]
            row2 = self.out_mzi_idx_forward[col2]

        ## measure input power at column col
        inp_power = self.get_input_cmeas(col, avg_num)

        """
        set theta = pi/2, set phi to 0, pi/2,
        pi, 3pi/2 to measure phase
        """
        self.chip.ps[(col2, row2)].phase = np.pi/2
        time.sleep(0.05)

        powers = []
        for jj in range(4):
            self.chip.ps[(col1,row1)].phase = np.pi/2*jj
            time.sleep(0.05)

            pp = np.zeros(6)
            for ii in range(avg_num):
                if self.backward:
                    pp += self.chip.fractional_left
                else:
                    pp += self.chip.fractional_right
            pp /= avg_num
            powers.append(pp)
        powers = np.asarray(powers)

        ## compute phase and set theta, phi to nullify
        phi = self.get_phi_down(powers[:,row2-1:row2+1]) 
        self.chip.ps[(col1, row1)].phase = phi + np.pi
        time.sleep(0.05)
        theta = 2*np.arctan2(inp_power[row2], inp_power[row2-1])
        self.chip.ps[(col2, row2)].phase = theta + np.pi
        time.sleep(0.05)

        ## measure output power, can be used as sanity check
        out_power = np.zeros(6)
        for ii in range(avg_num):
            if self.backward:
                out_power += self.chip.fractional_left
            else:
                out_power += self.chip.fractional_right
        out_power /= avg_num
        return phi, theta, out_power, powers


    def back_prop_layer(self, inp_field, phase_save, col):
        """
        after self-config, back propagate through column col
        to reconstruct the input field
        """
        if self.backward:
            col1 = col
            col2 = col-1
            row1 = self.out_mzi_idx_backward[col1]
            row2 = self.out_mzi_idx_backward[col2]
            phi = phase_save[(col1, row1)]
            theta = phase_save[(col2, row2)]
        else:
            col1 = col
            col2 = col+1
            row1 = self.out_mzi_idx_forward[col1]
            row2 = self.out_mzi_idx_forward[col2]
            phi = phase_save[(col1, row1)]
            theta = phase_save[(col2, row2)]

        if col == 12:
            H1 = np.asarray([[np.exp(-1j*phi),0],[0,1]])
        else:
            H1 = np.asarray([[1,0],[0,np.exp(-1j*phi)]])
        H2 = np.asarray([[1,0],[0,np.exp(-1j*theta)]])
        B = np.asarray([[1,-1j],[-1j,1]])/np.sqrt(2)
        M = H1 @ B @ H2 @ B
        out_field = M@np.asarray([inp_field,0]).astype(np.complex64)
        return out_field


    def meas_inp_power(self, avg_num):
        ## set all output MZI to bar state
        if self.backward:
            for cc in np.arange(1, 9):
                self.set_bar(cc)
        else:
            for cc in np.arange(10, 18):
                self.set_bar(cc)
        time.sleep(0.05)

        ## measure field power (matrix output)
        meas_vec = np.zeros(6)
        for nn in range(avg_num):
            if self.backward:
                meas_vec += self.chip.fractional_left
            else:
                meas_vec += self.chip.fractional_right
        meas_vec = meas_vec/avg_num
        return meas_vec


    def reconstruct_field(self, log_data = False, log_dir=None, avg_num=1):
        """
        log_data: whether log data
        log_dir: to save all intermediate results and measurement results

        return: normalized field reconstruction, 4 dimensions
        """
        if self.backward:
            self.chip.to_layer(0)
            time.sleep(1.0)
        else:
            self.chip.to_layer(16)
            time.sleep(1.0)
        
        ## measure input power
        inp_power_all = self.meas_inp_power(avg_num)
        
        ## self-config to measure phase
        if self.backward:
            phi, theta, out_power, meas_power = self.config_layer(8, avg_num)
            phi, theta, out_power, meas_power = self.config_layer(6, avg_num)
            phi, theta, out_power, meas_power = self.config_layer(4, avg_num)
            phi, theta, out_power, meas_power = self.config_layer(2, avg_num)
        else:
            phi, theta, out_power, meas_power = self.config_layer(10, avg_num)
            phi, theta, out_power, meas_power = self.config_layer(12, avg_num)
            phi, theta, out_power, meas_power = self.config_layer(14, avg_num)
            phi, theta, out_power, meas_power = self.config_layer(16, avg_num)

        phase_save = {}
        if self.backward:
            for cc in range(1,9):
                kk = (cc, self.out_mzi_idx_backward[cc])
                phase_save[kk] = self.chip.ps[kk].phase
        else:
            for cc in range(10,18):
                kk = (cc, self.out_mzi_idx_forward[cc])
                phase_save[kk] = self.chip.ps[kk].phase
        
        # pdb.set_trace()
        inp_field = 1
        rec_field = []
        if self.backward:
            for col in [2,4,6,8]:
                out_field = self.back_prop_layer(inp_field, phase_save, col)
                rec_field.append(out_field[0])
                inp_field = out_field[1]
            rec_field.append(out_field[1])
        else:
            for col in [16,14,12,10]:
                out_field = self.back_prop_layer(inp_field, phase_save, col)
                rec_field.append(out_field[0])
                inp_field = out_field[1]
            rec_field.append(out_field[1])

        if log_data and (log_dir is not None):
            np.save(os.path.join(log_dir, 'inp_power.npy'), inp_power_all)
            with open(os.path.join(log_dir, 'mesh_phase.pickle'), 'wb') as f:
                pickle.dump(phase_save, f)
            np.save(os.path.join(log_dir, 'out_power.npy'), out_power)
            np.save(os.path.join(log_dir, 'rec_field.npy'), rec_field)

        inp_phase = np.angle(rec_field) - np.angle(rec_field)[-1] + np.pi
        inp_phase = np.mod(inp_phase[:4], 2*np.pi)
        inp_mag = np.sqrt(np.abs(inp_power_all[:4]))
        rec_field_norm = inp_mag * np.exp(1j*inp_phase)
        rec_field_norm = rec_field_norm / np.linalg.norm(rec_field_norm)
        return rec_field_norm    
    