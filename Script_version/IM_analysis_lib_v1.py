# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:29:05 2023

@author: Yicheng Liao (YLI), Energinet

Immitance analysis library - v1

Update history:
    21 Mar 2025 - v1 - first public version
    
"""
import cmath
import math
import numpy as np
from numpy.fft import fft, ifft
import numpy.linalg as lnlg 
import pandas as pd


"""
===================================================================================
-------------------------- Time-domain tool library ------------------------------
===================================================================================
"""

# FFT analysis of a waveform y
# Inputs: y - waveform vector
#         fs - sampling frequency
# Return: f_oneside - frequency vector (>= 0 Hz)
#         Y_oneside - Fourier coefficient vector (complex-valued)
# Update: 21 Jul 2023 - YLI - first version
def get_FFT(y,fs):
    
    y = y*np.hamming(len(y))
    # FFT
    Y = fft(y)
    N = len(y)
    n = np.arange(N)
    T = N/fs
    freq = n/T 
        
    # Convert to the one-sided specturm
    n_oneside = N//2
    # get the one side frequency vector
    f_oneside = freq[:n_oneside]
        
    # normalize the amplitude
    Y_oneside =Y[:n_oneside]/n_oneside

    return f_oneside,Y_oneside



# FFT analysis at a selected frequency
# Inputs: f_oneside - frequency vector (>= 0 Hz)
#         Y_oneside - Fourier coefficient vector (complex-valued)
#         f - selected frequency, Hz
# Return: Y_f - Fourier coefficient at f (a complex value)
# Update: 06 Dec 2023 - YLI - second version
#         16 Feb 2024 - YLI - added consideration of negative f for MIMO impedance calculation 
def get_FFT_at_f_v2(f_oneside,Y_oneside,f):
    if f == 0:
        if f_oneside[0] == 0:
            Y_f = Y_oneside[0]/2
        else:
            print('-------- First frequency of the FFT result not at 0Hz --------')
            Y_f = 0
    elif f < 0:
        f_step = f_oneside[1]-f_oneside[0]
        f_idx = round((-f)/f_step)
        Y_f = np.conj(Y_oneside[f_idx])
    else:
        f_step = f_oneside[1]-f_oneside[0]
        # find the index for the exact frequency
        f_idx = round(f/f_step)
        # print(f/f_step)
        # print(f_idx)
        # print(f)
        Y_f = Y_oneside[f_idx]
    return Y_f


# DFT analysis at a single selected frequency
# Inputs: signal - signal vector 
#         fs - sampling frequency, Hz
#         f - selected frequency, Hz, must be positive
# Return: X - Fourier coefficient at f (a complex value)
# Update: 25 Jul 2024 - AUF - updated DFT algorithm
#         30 Jul 2024 - YLI - updated for negative frequency
def DFT_1f(signal, fs, f):
    # helping variables
    N = len(signal) # NUmber of samples
    T_signal = N/fs # Period of samples signal
    f_nyquist = fs/2 # Nyquist frequency

    # check if f outside f_nyquist and positive -> and return 0
    if f >= f_nyquist:
        print("Requested frequency is outside possible frequency range for the signal")
        return 0
    # check if f = 0 and do faster procedure
    elif f == 0:
        Amp = sum(signal)/N
        # Phase = 0
        return Amp
    elif f < 0:
        fpos = -f
        if (T_signal*fpos) % 1 == 0:
            # helping values and arrays
            delta_t = 1/fs
            t = np.linspace(delta_t,T_signal,N)
            # creating complex wave for the time
            s_m = np.exp(-1j*2*np.pi*fpos*t)
            # creating np array our of signal
            x = np.array(signal) 
            # multiplying with singal and summing
            X = np.sum(x*s_m)/N
            return np.conj(X)
        
    # check if f is inside of the signal and fs spectrum for DFT
    elif (T_signal*f) % 1 == 0:
        # helping values and arrays
        delta_t = 1/fs
        t = np.linspace(delta_t,T_signal,N)
        # creating complex wave for the time
        s_m = np.exp(-1j*2*np.pi*f*t)
        # creating np array our of signal
        x = np.array(signal) 
        # multiplying with singal and summing
        X = np.sum(x*s_m)/N
        return X

    # frequency cannot be calculated with this signal
    else:
        print("Wrong number of samples to calculate required DFT")
        return 0


# Clarke transformation: abc->alpha,beta,zero
# Inputs: ya,yb,yc - vectors
#         tf_type - transformation type, 1 - amplitude invariant, 2 - power invariant
# Return: yalpha,ybeta,yzero - 1-D numpy arrays
# Update: 28 Jul 2023 - YLI - first version
def abc2alpha_beta_gamma(ya,yb,yc,tf_type=2):
    yalpha = np.zeros(len(ya))
    ybeta = np.zeros(len(ya))
    ygamma = np.zeros(len(ya))
    # check waveform data format
    if len(ya) == len(yb) and len(ya) == len(yc):

        
        for k in range(len(ya)):
            yin = np.array([ya[k],yb[k],yc[k]])
            if tf_type == 1:
                T = 2/3*np.array([[1,-1/2,-1/2],[0,math.sqrt(3)/2,-math.sqrt(3)/2],[1/2,1/2,1/2]])                
                yout = np.matmul(T,yin)
            elif tf_type == 2:
                T = math.sqrt(2/3)*np.array([[1,-1/2,-1/2],[0,math.sqrt(3)/2,-math.sqrt(3)/2],[1/2,1/2,1/2]])
                yout = np.matmul(T,yin)
            else:
                print('-------- Transformation type NOT defined --------')
                yout = np.zeros(yin.shape)
            yalpha[k] = yout[0]
            ybeta[k] = yout[1]
            ygamma[k] = yout[2]
    else:
        print('-------- Data formats of inputs are NOT correct --------')

    return yalpha,ybeta,ygamma
     
# Complex transformation: real vector -> complex vector
# Inputs: ya,yb - real vectors
# Return: yab1,yab2 - complex vectors
# Update: 28 Jul 2023 - YLI - first version
def realvec2complexvec(ya,yb):
    yab1 = np.zeros(len(ya),dtype=complex)
    yab2 = np.zeros(len(ya),dtype=complex)
    # check waveform data format
    if len(ya) == len(yb):       
        for k in range(len(ya)):
            yin = np.array([ya[k],yb[k]])
            T = np.array([[1,1j],[1,-1j]])                
            yout = np.matmul(T,yin)

            yab1[k] = yout[0]
            yab2[k] = yout[1]
    else:
        print('-------- Data formats of inputs are NOT correct --------')

    return yab1,yab2    


"""
===================================================================================
-------------------------- Frequency-domain tool library ------------------------------
===================================================================================
"""

# Define a TF object for a transfer funtion or transfer function matrix
# f - frequency vector
# values - frequency-domain response (complex-valued) at the correponding frequencies
# Update: 21 Jul 2023 - YLI - first version
#         17 Jul 2024 - YLI - 2nd version, defining f and values type
class TF:
    f: list
    values: list
    def __init__(self,f,values):
        self.f = f
        self.values = values

# Check SISO or MIMO model
# Return: True - MIMO model
#         False - SISO model
def TF_check(TF_model):
    if TF_model.values[0].shape:
        return True
    else:
        return False
# Calculate TF inverse 
# Note: TF shall be a n-by-n matrix, so it can be inverted
# Update: 21 Jul 2023 - YLI - first version
#         19 Jul 2024 - YLI - 2nd version, updtated based on TF type, and applicable for both SISO and MIMO TF models
def TF_inv(TF_mat):
    if TF_check(TF_mat):
        if TF_mat.values[0].shape[0] == TF_mat.values[0].shape[1]:
            n_f = len(TF_mat.f)
            Mat_vec = []
            for k in range(n_f):
                Mat_vec.append(lnlg.inv(TF_mat.values[k]))
        
            TF_inv = TF(TF_mat.f,Mat_vec) 
        else: 
            print('-------- TF_mat is NOT a square matrix --------')
            TF_inv = 'None'
    else:
        n_f = len(TF_mat.f)
        Mat_vec = []
        for k in range(n_f):
            Mat_vec.append(1/(TF_mat.values[k]))
        TF_inv = TF(TF_mat.f,Mat_vec) 

    return TF_inv

# Caclulate TF sum, element-wise
# Note: TF_mat1 and TF_mat2 shall have the same matrix dimension
# Update: 21 Jul 2023 - YLI - first version
#         19 Jul 2024 - YLI - 2nd version, updtated based on TF type, and applicable for both SISO and MIMO TF models

def TF_sum(TF_mat1,TF_mat2):
    if TF_check(TF_mat1) and TF_check(TF_mat2):
        if len(TF_mat1.f) == len(TF_mat2.f):
            if TF_mat1.values[0].shape[0] == TF_mat2.values[0].shape[0] and TF_mat1.values[0].shape[1] == TF_mat2.values[0].shape[1]:
                n_f = len(TF_mat1.f)
                Mat_vec = []
                for k in range(n_f):
                    Mat_vec.append(np.add(TF_mat1.values[k],TF_mat2.values[k]))
            
                TF_sum = TF(TF_mat1.f,Mat_vec) 
                
            else: 
                print('-------- TF_mat1 and TF_mat2 do NOT have same matrix dimensions --------')
                TF_sum = 'None'
        else:
            print('-------- TF_mat1 and TF_mat2 do NOT have same frequency points --------')
            TF_sum = 'None'
    else:
        if TF_check(TF_mat1) or TF_check(TF_mat2): 
            print('-------- TF_mat1 and TF_mat2 do NOT have same model form (SISO or MIMO) --------')
            TF_sum = 'None'
        else:
            if len(TF_mat1.f) == len(TF_mat2.f):
                n_f = len(TF_mat1.f)
                Mat_vec = []
                for k in range(n_f):
                    Mat_vec.append(TF_mat1.values[k]+TF_mat2.values[k])
            
                TF_sum = TF(TF_mat1.f,Mat_vec) 
            else:
                print('-------- TF_mat1 and TF_mat2 do NOT have same frequency points --------')
                TF_sum = 'None'
        
    
    return TF_sum
 

# Calculate TF product    
# Note: TF_mat1 and TF_mat2 shall be capable of multiplication
# Update: 21 Jul 2023 - YLI - first version
#         19 Jul 2024 - YLI - 2nd version, updtated based on TF type, and applicable for both SISO and MIMO TF models
def TF_product(TF_mat1,TF_mat2):
    if TF_check(TF_mat1) and TF_check(TF_mat2):
        if len(TF_mat1.f) == len(TF_mat2.f):
            if TF_mat1.values[0].shape[1] == TF_mat2.values[0].shape[0]:
                n_f = len(TF_mat1.f)
                Mat_vec = []
                for k in range(n_f):
                    Mat_vec.append(np.matmul(TF_mat1.values[k],TF_mat2.values[k]))
            
                TF_product = TF(TF_mat1.f,Mat_vec) 
            else: 
                print('-------- TF_mat1 and TF_mat2 are NOT capable of multiplication --------')
                TF_product= 'None'
        else:
            print('-------- TF_mat1 and TF_mat2 do NOT have same frequency points --------')
            TF_product = 'None'
    else:
        if TF_check(TF_mat1) or TF_check(TF_mat2): 
            print('-------- TF_mat1 and TF_mat2 do NOT have same model form (SISO or MIMO) --------')
            TF_product = 'None'
        else:
            if len(TF_mat1.f) == len(TF_mat2.f):
                n_f = len(TF_mat1.f)
                Mat_vec = []
                for k in range(n_f):
                    Mat_vec.append(TF_mat1.values[k]*TF_mat2.values[k])
            
                TF_product = TF(TF_mat1.f,Mat_vec) 
            else:
                print('-------- TF_mat1 and TF_mat2 do NOT have same frequency points --------')
                TF_product = 'None'
    
    return TF_product

# Calculate TF scaling 
# Note: TF_mat shall be a square tansfer funtion matrix or a SISO model
# Update: 21 Jul 2023 - YLI - first version
#         19 Jul 2024 - YLI - 2nd version, updtated based on TF type, and applicable for both SISO and MIMO TF models

def TF_scale(scaler,TF_mat):
    if TF_check(TF_mat):
        if TF_mat.values[0].shape[0] == TF_mat.values[0].shape[1]:
            n_f = len(TF_mat.f)
            n_dim = TF_mat.values[0].shape[0]
            
            Mat_vec = []
            Mat_scaler = scaler*np.identity(n_dim)
            
            for k in range(n_f):
                Mat_vec.append(np.matmul(Mat_scaler,TF_mat.values[k]))
            
            TF_product = TF(TF_mat.f,Mat_vec) 
        else: 
            print('-------- TF_mat is NOT a square matrix --------')
            TF_product = 'None'
    else:
        n_f = len(TF_mat.f)
        
        Mat_vec = []
        
        for k in range(n_f):
            Mat_vec.append(Mat_scaler*TF_mat.values[k])
        
        TF_product = TF(TF_mat.f,Mat_vec) 
    
    return TF_product

# Calculate TF added with constant matrix
# Update: 21 Jul 2023 - YLI - first version
def TF_add_constant(const,TF_mat):
    if TF_check(TF_mat):
        n_f = len(TF_mat.f)
        n_row = TF_mat.values[0].shape[0]
        n_col = TF_mat.values[0].shape[1]
        
        Mat_vec = []                          
        Mat_adder = const*np.ones((n_row,n_col),dtype=np.complex)

        for k in range(n_f):
            Mat_vec.append(np.add(Mat_adder,TF_mat.values[k]))

        TF_sum = TF(TF_mat.f,Mat_vec) 
    else:
        n_f = len(TF_mat.f)
        
        Mat_vec = []                          

        for k in range(n_f):
            Mat_vec.append(const + TF_mat.values[k])

        TF_sum = TF(TF_mat.f,Mat_vec) 
      
   
    return TF_sum

# Eigenvalue sort to obtain index
# Note: used in TF_eig function
# Update: 21 Jul 2023 - YLI - first version
def eigval_sort_index(eigval_last,eigval_current):
    n_rank = len(eigval_last)
    idx_vec = []
    for m in range(n_rank): # for m-th eigenvalue
        eig_last = eigval_last[m]
        for l in range(n_rank):
            eig_current = eigval_current[l]
            eig_error = np.abs(eig_current - eig_last)
            if l == 0:
                eig_error_min = eig_error
                idx = l
            else:
                if eig_error < eig_error_min:
                    eig_error_min = eig_error
                    idx = l
        idx_vec.append(idx)            
    return idx_vec

# Directly sort eigenvalues and eigenvectors
# Note: used in TF_eig function
# Update: 21 Jul 2023 - YLI - first version
def eig_sort(eigval_last,eigval_current,eigvec_current):
    n_rank = len(eigval_last)
    for m in range(n_rank): # for m-th eigenvalue
        eig_last = eigval_last[m]
        for l in range(m,n_rank):
            eig_current = eigval_current[l]
            eig_error = np.abs(eig_current - eig_last)
            if l == m:
                eig_error_min = eig_error
            else:
                if eig_error < eig_error_min:
                    eig_error_min = eig_error

                    eigval_tmp = eigval_current[l]
                    eigval_current[l] = eigval_current[m]
                    eigval_current[m] = eigval_tmp
                    
                    eigvec_tmp = eigvec_current[:,l]
                    eigvec_current[:,l] = eigvec_current[:,m]
                    eigvec_current[:,m] = eigvec_tmp                    
                    
    return eigval_current, eigvec_current

# Calculate TF eigenvalues and eigenvectors
# Note: TF_mat shall be a square matrix
# Return: TF_eigval - eigenvalues of the input TF matrix 
#         TF_eigvec - eigenvectors of the input TF matrix, i.e., right eigenvector matrix
# Update: 21 Jul 2023 - YLI - first version
def TF_eig(TF_mat):
    n_f = len(TF_mat.f) 
    n_rank = TF_mat.values[0].shape[0]
                            
    eigval_vec = []
    eigvec_vec = []


    for k in range(n_f):
        eigval_current,eigvec_current = lnlg.eig(TF_mat.values[k])
        if k ==0:
            
            eigval_sorted = eigval_current
            eigvec_sorted = eigvec_current
            eigval_last = eigval_sorted
        else:
            
            # using eigval_sort_index
            # idx = eigval_sort_index(eigval_last, eigval_current)
            # eigval_sorted = []
            # eigvec_sorted = np.empty((n_rank,n_rank),dtype = complex)
            # for m in range(n_rank):
            #     eigval_sorted.append(eigval_current[idx[m]])
            #     eigvec_sorted[:,m]=eigvec_current[:,idx[m]]
            # eigval_last = eigval_sorted
            
            
            # using eig_sort
            eigval_sorted,eigvec_sorted = eig_sort(eigval_last,eigval_current,eigvec_current)
            eigval_current = eigval_sorted
            eigvec_current = eigvec_sorted
            eigval_last = eigval_sorted
            
          
            
        eigval_vec.append(eigval_sorted)
        eigvec_vec.append(eigvec_sorted)

    TF_eigval = TF(TF_mat.f,eigval_vec) 
    TF_eigvec = TF(TF_mat.f,eigvec_vec)
    
    return TF_eigval, TF_eigvec

# Calculate TF determinant
# Note: TF_mat shall be a square matrix
# Update: 21 Jul 2023 - YLI - first version
def TF_detF(TF_mat):
    n_f = len(TF_mat.f)
    n_rank = TF_mat.values[0].shape[0]

    detF_vec = []

    for k in range(n_f):
        detF_vec.append(lnlg.det(np.identity(n_rank)+TF_mat.values[k]))

    TF_detF = TF(TF_mat.f,detF_vec) 
    
    return TF_detF

# Calculate frequency-domain participation functions
# Note: Used after using TF_eig function
# Update: 21 Jul 2023 - YLI - first version
def TF_eig_participation(TF_eigval,TF_eigvec):
    n_f = len(TF_eigval.f)
    n_rank = TF_eigvec.values[0].shape[0]
                    
  
    Participation_mat = []
    

    for k in range(n_f):
        V = TF_eigvec.values[k]       # right eigenvector matrix
        W = lnlg.inv(V)               # left eigenvector matrix
        
        Participation = np.empty((n_rank,n_rank),dtype=np.complex)
        for l in range(n_rank):
            for m in range(n_rank):
                Participation[m,l]=W[l,m] * V[m,l]
                
        Participation_mat.append(Participation)
    
    TF_participation = TF(TF_eigval.f,Participation_mat)
    return TF_participation

# Convert Zdq to P (Jacobian matrix)
# Update: 19 April 2024 - YLI - first version
def Zdq2P(Zdq,Pss,Qss,Vss): 
    # Pss, Qss defined as current flowing into the component
    Pt = -Pss
    Qt = -Qss
    Av = 1/Vss*np.array([[Pt,-Qt],[Qt,Pt]])
    Ai = np.array([[Vss,0],[0,-Vss]])
    T = np.array([[0,1],[Vss,0]])
    
    n_f = len(Zdq.f) 
    # n_rank = Zdq.values[0].shape[0]
    Kn = []
       
    for k in range(n_f):
        Z = Zdq.values[k]
        Kn.append(np.matmul((Av-np.matmul(Ai,np.linalg.inv(Z))),T))
        

   
    Kn_TF = TF(Zdq.f, Kn)
    return Kn_TF   

"""
===================================================================================
-------------------------- File saving and reading library ------------------------------
===================================================================================
"""


# IMTB_AC immittance saved to CSV file
# Inputs: filename - CSV file name including path
#         str_list - name list of the data to be saved
#         data_list - data list to be saved
# Return: data - data in a dict object
# Update: 28 Jul 2023 - YLI - first version
#         30 Jul 2024 - YLI - 2nd version, adding magnitude and phase results, saved in EXCEL file

def IMTB_AC_immittace_toCSV(filename,str_list,data_list):
    data = {}
    for k in range(len(str_list)):
        if 'complex' in data_list[k][0].dtype.name:
            # Complex number
            data[str_list[k]+'_re'] = np.real(data_list[k])
            data[str_list[k]+'_im'] = np.imag(data_list[k])
            
            # Magnitude and phase
            data[str_list[k]+'_abs'] = np.abs(data_list[k])
            data[str_list[k]+'_dB'] = 20*np.log10(np.abs(data_list[k]))
            data[str_list[k]+'_pha_rad'] = np.angle(data_list[k],deg=False)
            data[str_list[k]+'_pha_deg'] = np.angle(data_list[k],deg=True)
            
        else:
            data[str_list[k]] = data_list[k]
    
    df = pd.DataFrame.from_dict(data)
    
    
    # save in csv file
    df.to_csv(filename,index = False)
    
    # save in Excel file
    if ".csv" in filename:
        excelfile = filename.replace(".csv",".xlsx")         
    elif ".xlsx" in filename:
        excelfile = filename
    else:
        excelfile = filename + ".xlsx"
        
    df.to_excel(excelfile,index = False)
    
    return data


# Read immittance data from CSV file into TF objects
# Update: 28 Jul 2023 - YLI - first version
def IM_read_CSV(filename):    
    df = pd.read_csv(filename)
    h1 = [column.split('_', 1)[0] for column in df.columns]
    print(h1)
    h2 = [column.split('_', 1)[-1] for column in df.columns]
    print(h2)
    h1[0] = 'f'
    h2[0] = 'Hz'
    df.columns = [h1,h2]
    # print(h1)
    # print(h2)
    
    var_set = []     
    Z_TF = {}
    
    
    
    if 'IM_Zab_MIMO_' in filename:
        
        for idx,h in enumerate(h1):
            if idx == 0:
                freq = np.asarray(pd.DataFrame(df[h,h2[idx]]))
                flist = []
                for n in range(len(freq)):
                    flist.append(freq[n][0])
            elif h not in var_set:
                
                Z_re = np.asarray(pd.DataFrame(df[h,'ab_11_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'ab_11_im']))
                Z11 = Z_re + 1j*Z_im
                # print(Z11[1][0])

                
                Z_re = np.asarray(pd.DataFrame(df[h,'ab_12_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'ab_12_im']))
                Z12 = Z_re + 1j*Z_im
                
                Z_re = np.asarray(pd.DataFrame(df[h,'ab_21_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'ab_21_im']))
                Z21 = Z_re + 1j*Z_im
                
                Z_re = np.asarray(pd.DataFrame(df[h,'ab_22_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'ab_22_im']))
                Z22 = Z_re + 1j*Z_im
                Zlist = []
                for n in range(len(flist)):
                    Zlist.append(np.asarray([[Z11[n][0],Z12[n][0]],[Z21[n][0],Z22[n][0]]]))
                                                   
                Z_TF[h] = TF(flist,Zlist)        
                var_set.append(h)
        return Z_TF
    elif 'IM_Zdq_MIMO_' in filename:
        
        for idx,h in enumerate(h1):
            if idx == 0:
                freq = np.asarray(pd.DataFrame(df[h,h2[idx]]))
                flist = []
                for n in range(len(freq)):
                    flist.append(freq[n][0])
            elif h not in var_set:
                
                Z_re = np.asarray(pd.DataFrame(df[h,'dq_11_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'dq_11_im']))
                Z11 = Z_re + 1j*Z_im
                # print(Z11[1][0])

                
                Z_re = np.asarray(pd.DataFrame(df[h,'dq_12_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'dq_12_im']))
                Z12 = Z_re + 1j*Z_im
                
                Z_re = np.asarray(pd.DataFrame(df[h,'dq_21_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'dq_21_im']))
                Z21 = Z_re + 1j*Z_im
                
                Z_re = np.asarray(pd.DataFrame(df[h,'dq_22_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'dq_22_im']))
                Z22 = Z_re + 1j*Z_im
                Zlist = []
                for n in range(len(flist)):
                    Zlist.append(np.asarray([[Z11[n][0],Z12[n][0]],[Z21[n][0],Z22[n][0]]]))
                                                   
                Z_TF[h] = TF(flist,Zlist)        
                var_set.append(h)
        return Z_TF
    elif 'IM_pos_' in filename:
        
        for idx,h in enumerate(h1):
            if idx == 0:
                freq = np.asarray(pd.DataFrame(df[h,h2[idx]]))
                flist = []
                for n in range(len(freq)):
                    flist.append(freq[n][0])
            elif h not in var_set:
                Z_re = np.asarray(pd.DataFrame(df[h,'re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'im']))
                Z = Z_re + 1j*Z_im
                # print(Z)
                Zlist = []
                for n in range(len(flist)):
                    Zlist.append(Z[n][0])
                
                
                Z_TF[h] = TF(flist,Zlist)
                var_set.append(h)

        
        return Z_TF 
            
    elif 'IM_Kn_' in filename:
        for idx,h in enumerate(h1):
            if idx == 0:
                freq = np.asarray(pd.DataFrame(df[h,h2[idx]]))
                flist = []
                for n in range(len(freq)):
                    flist.append(freq[n][0])
            elif h not in var_set:
                
                Z_re = np.asarray(pd.DataFrame(df[h,'11_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'11_im']))
                Z11 = Z_re + 1j*Z_im
                # print(Z11[1][0])

                
                Z_re = np.asarray(pd.DataFrame(df[h,'12_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'12_im']))
                Z12 = Z_re + 1j*Z_im
                
                Z_re = np.asarray(pd.DataFrame(df[h,'21_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'21_im']))
                Z21 = Z_re + 1j*Z_im
                
                Z_re = np.asarray(pd.DataFrame(df[h,'22_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'22_im']))
                Z22 = Z_re + 1j*Z_im
                Zlist = []
                for n in range(len(flist)):
                    Zlist.append(np.asarray([[Z11[n][0],Z12[n][0]],[Z21[n][0],Z22[n][0]]]))
                                                   
                Z_TF[h] = TF(flist,Zlist)        
                var_set.append(h)
                
        return Z_TF
    elif "IM_Zdc_MIMO" in filename:
        
        for idx,h in enumerate(h1):
            if idx == 0:
                freq = np.asarray(pd.DataFrame(df[h,h2[idx]]))
                flist = []
                for n in range(len(freq)):
                    flist.append(freq[n][0])
            elif h not in var_set:
                
                Z_re = np.asarray(pd.DataFrame(df[h,'dc_11_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'dc_11_im']))
                Z11 = Z_re + 1j*Z_im
                # print(Z11[1][0])

                
                Z_re = np.asarray(pd.DataFrame(df[h,'dc_12_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'dc_12_im']))
                Z12 = Z_re + 1j*Z_im
                
                Z_re = np.asarray(pd.DataFrame(df[h,'dc_21_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'dc_21_im']))
                Z21 = Z_re + 1j*Z_im
                
                Z_re = np.asarray(pd.DataFrame(df[h,'dc_22_re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'dc_22_im']))
                Z22 = Z_re + 1j*Z_im
                Zlist = []
                for n in range(len(flist)):
                    Zlist.append(np.asarray([[Z11[n][0],Z12[n][0]],[Z21[n][0],Z22[n][0]]]))
                                                   
                Z_TF[h] = TF(flist,Zlist)        
                var_set.append(h)
        return Z_TF
    else:
        # for SISO csv files
        #!!!
        for idx,h in enumerate(h1):
            if idx == 0:
                freq = np.asarray(pd.DataFrame(df[h,h2[idx]]))
                flist = []
                for n in range(len(freq)):
                    flist.append(freq[n][0])
            elif h not in var_set:
                
                Z_re = np.asarray(pd.DataFrame(df[h,'re']))
                Z_im = np.asarray(pd.DataFrame(df[h,'im']))
                Z = Z_re + 1j*Z_im
                # print(Z11[1][0])

                Zlist = []
                for n in range(len(flist)):
                    Zlist.append(np.asarray(Z[n][0]))
                                                   
                Z_TF[h] = TF(flist,Zlist)        
                var_set.append(h)
        return Z_TF
        



"""
===================================================================================
---------------------------------- Plotting library ------------------------------
===================================================================================
"""


# Calculate magnitude
def Mat_mag(Mat_vec):
    mag = 20*np.log10(np.abs(Mat_vec))
    return mag
# Calculate phase
# Update on 14-11-2024 (YLI): "unwrap_on" parameter removed from "Mat_pha" function
def Mat_pha(Mat_vec,deg_true=True):
    pha = np.angle(Mat_vec,deg=deg_true)
    return pha




