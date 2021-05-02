'''
There are a bunch of print statements, that print out basically everything from Q through the Rs for Max Stress and
Tsai-Wu and the critical loads. Those last two are summarized at the end so they're easy to find.  I prettied them up a
bit to make them clear and easy to read. Hope this helps!

DIFFERENT MATERIAL PROPERTIES ARE COMMENTED OUT AND COLLAPSED ON LINE 217
'''

import math
import numpy as np
from numpy import linalg as lg
from numpy import matmul as mm  # Matrix multiplication
from math import sin as s
from math import cos as c


# Calculating the local (Q) and global (Q_bar) stiffness matrices
def calculate_Q_and_Q_bar(E11, E22, G12, V12, V21, N, T, T_hat):
    Q11 = E11 / (1 - V12 * V21)
    Q12 = (V21 * E11) / (1 - V12 * V21)
    Q21 = (V12 * E22) / (1 - V12 * V21)
    Q22 = E22 / (1 - V12 * V21)
    Q = np.array([[Q11, Q12, 0], [Q21, Q22, 0], [0, 0, G12]],dtype=float)
    Q_bar = []
    for i in range(N):
        Q_bar.append(mm(lg.inv(T[i]), mm(Q, T_hat[i])))  # The global/laminate stiffness matrix, pg 114
    return Q, Q_bar


def calculate_T(N, angle):
    T = []
    for i in range(N):
        T.append(np.array([[c(angle[i]) ** 2, s(angle[i]) ** 2, 2 * s(angle[i]) * c(angle[i])],
                         [s(angle[i]) ** 2, c(angle[i]) ** 2, -2 * s(angle[i]) * c(angle[i])],
                         [-s(angle[i]) * c(angle[i]), s(angle[i]) * c(angle[i]), c(angle[i]) ** 2 - s(angle[i]) ** 2]]))
    return T


def calculate_T_hat(N, angle):
    T_hat = []
    for i in range(N):
        T_hat.append(np.array([[c(angle[i]) ** 2, s(angle[i]) ** 2, s(angle[i]) * c(angle[i])],
                             [s(angle[i]) ** 2, c(angle[i]) ** 2, -s(angle[i]) * c(angle[i])],
                [-2 * s(angle[i]) * c(angle[i]), 2 * s(angle[i]) * c(angle[i]), c(angle[i]) ** 2 - s(angle[i]) ** 2]]))
    return T_hat


def calculate_A(N, Q_bar, t_ply):
    A = [[0] * 3] * 3
    for i in range(N):
        A += Q_bar[i] * t_ply
    return A


def calculate_B(N, Q_bar, z, t_ply):
    B = [[0] * 3] * 3
    for i in range(N):
        B += (1 / 2) * (Q_bar[i] * ((z[i] ** 2) - ((z[i] - t_ply) ** 2)))
    return B


def calculate_D(N, Q_bar, z, t_ply):
    D = [[0] * 3] * 3
    for i in range(N):
        D += (1 / 3) * (Q_bar[i] * ((z[i] ** 3) - ((z[i] - t_ply) ** 3)))
    return D


def calculate_ABD(A, B, D):
    ABD = np.array([[A[0][0], A[0][1], A[0][2], B[0][0], B[0][1], B[0][2]],
                    [A[1][0], A[1][1], A[1][2], B[1][0], B[1][1], B[1][2]],
                    [A[2][0], A[2][1], A[2][2], B[2][0], B[2][1], B[2][2]],
                    [B[0][0], B[0][1], B[0][2], D[0][0], D[0][1], D[0][2]],
                    [B[1][0], B[1][1], B[1][2], D[1][0], D[1][1], D[1][2]],
                    [B[2][0], B[2][1], B[2][2], D[2][0], D[2][1], D[2][2]]])
    return ABD


def calculate_midplane_strains_and_curvatures(ABD, stress_resultant):
    mps_and_curvs = lg.inv(ABD) @ stress_resultant
    midplane_strains = np.array([mps_and_curvs[0], mps_and_curvs[1], mps_and_curvs[2]])
    curvatures = np.array([mps_and_curvs[3], mps_and_curvs[4], mps_and_curvs[5]])
    return midplane_strains, curvatures


# Global Strains at mid-plane of each ply
def calculate_global_strains(N, mid_plane_strains,  z_mid_plane, curvatures):
    global_strains = [[[0]] * 3] * N
    for i in range(N):
        global_strains[i] = mid_plane_strains + z_mid_plane[i] * curvatures
    return global_strains


# Global Stresses at mid-plane of each ply
def calculate_global_stresses(N, Q_bar, global_strains):
    global_stresses = [[[0]] * 3] * N
    for i in range(N):
        global_stresses[i] = mm(Q_bar[i], global_strains)
    return global_stresses


# Local strains at mid-plane of each ply
def calculate_local_strains(N, T_hat, global_strains):
    local_strains = [[[0]] * 3] * N
    for i in range(N):
            local_strains[i] = mm(T_hat[i], global_strains)
    return local_strains


# Local stresses at mid-plane of each ply
def calculate_local_stresses(N, Q, local_strains):
    local_stresses = [[[0]] * 3] * N
    for i in range(N):
        local_stresses[i] = mm(Q, local_strains[i])
    return local_stresses


# Strength ratios for max stress failure criterion
def R_Max_Stress(N, SLt, STt, SLTs, local_stresses):
    R_sig_11 = []
    for i in range(N):
        R_sig_11.append(SLt / math.fabs(local_stresses[i][0]))
    R_sig_22 = []
    for i in range(N):
        R_sig_22.append(STt / math.fabs(local_stresses[i][1]))
    R_tau_12 = []
    for i in range(N):
        R_tau_12.append(SLTs / math.fabs(local_stresses[i][2]))
    R_MS = []
    for i in range(N):
        R_MS.append(min(R_sig_11[i], R_sig_22[i], R_tau_12[i]))
    return R_MS


# Max stress failure criterion critical loads
def R_MS_Critical_Loads(N, R_MS, stress_resultant):
    N_MScrit = []
    for i in range(N):
        N_MScrit.append(R_MS[i] * max(stress_resultant[0], stress_resultant[1], stress_resultant[2], key=abs))
    M_MScrit = []
    for i in range(N):
        M_MScrit.append(R_MS[i] * max(stress_resultant[3], stress_resultant[4], stress_resultant[5], key=abs))
    return N_MScrit, M_MScrit\


# Define Tsai-Wu quadratic function coefficients (aR^2 + bR + cc = 0)
def Tsai_Wu_coeffs(N, SLt, SLc, STt, STc, SLTs, local_stresses):
    # Tsai-Wu Coefficients
    F11 = 1 / (SLt * SLc)
    F22 = 1 / (STt * STc)
    F12 = (-1 / 2) * math.sqrt(F11 * F22)
    F66 = 1 / (SLTs ** 2)
    F1 = (1 / SLt) - (1 / SLc)
    F2 = (1 / STt) - (1 / STc)
    a = []
    for i in range(N):
        a.append(float((F11 * (local_stresses[i][0] ** 2)) + (2 * F12 * local_stresses[i][0] * local_stresses[i][1]) + (
                    F22 * (local_stresses[i][1] ** 2)) + (F66 * (local_stresses[i][2] ** 2))))
    b = []
    for i in range(N):
        b.append(float((F1 * local_stresses[i][0]) + (F2 * local_stresses[i][1])))
    cc = [-1] * N
    return a, b, cc, F11, F22, F12, F66, F1, F2


# Strength ratios for Tsai-Wu criterion
def R_Tsai_Wu(N, a, b, cc):
    R_1 = []
    for i in range(N):
        R_1.append((-b[i] + math.sqrt((b[i] ** 2) - 4 * a[i] * cc[i])) / (2 * a[i]))
    R_2 = []
    for i in range(N):
        R_2.append((-b[i] - math.sqrt((b[i] ** 2) - 4 * a[i] * cc[i])) / (2 * a[i]))
    R_TW = []
    for i in range(N):
        R_TW.append(float(R_1[i]))
    return R_TW


def calculate_E_xx(A, h):
    E_xx = (A[0][0] / h) * (1 - ((A[0][1] ** 2) / (A[0][0] * A[1][1])))
    return E_xx


def calculate_G_xy(A, h):
    G_xy = A[2][2] / h
    return G_xy


def calculate_I_f_I_w_and_I_s(H, W, h, E_R, E_xx_f, E_xx_w):
    I_f = 2 * (((W*(h**3))/12) + W*h*(((H+h)/2)**2))
    I_w = (1/12) * h * (H**3)
    I_s = ((E_xx_f*I_f)/E_R) + ((E_xx_w*I_w)/E_R)
    return I_f, I_w, I_s


def calculate_Q_f_and_Q_w(E_xx_f, E_xx_w, E_R, H, h, W):
    Q_f = (E_xx_f/E_R) * (((H+h)/2)*h*(W/2))
    Q_w = ((E_xx_w/E_R)*(((H**2)*h)/8)) + ((E_xx_f/E_R)*(((H+h)/2)*h*W))
    return Q_f, Q_w


def calculate_Nxxf_Nyyf_Nxyf_matrix(E_f, E_R, M, H, h, I_s, V, Q_f):
    Nxxf = (E_f/E_R) * (M*((H+h)/2)*h)/I_s
    Nxyf = V * Q_f / I_s
    Nxxf_matrix = np.array([[Nxxf], [0], [Nxyf]])
    return Nxxf_matrix


def new_section():
    for i in range(100):
        print('_', end='')
    print('\n')


def main():
    '''
    # Independent material properties for Scotchply 1002 in SI units
    E11  = 38.6  * (10**9)  # Pascals
    E22  = 8.27 * (10**9)  # Pascals
    G12  = 4.14 * (10**9)  # Pascals
    V12  = 0.26            # unit-less
    V21 = (V12*E22)/E11    # unit-less

    # # Typical strengths of Scotchply 1002 in SI units
    SLt  = 1062 * (10**6)  # Pascals
    SLc  = 610 * (10**6)  # Pascals
    STt  = 31   * (10**6)  # Pascals
    STc  = 118  * (10**6)  # Pascals
    SLTs = 72   * (10**6)  # Pascals




    # Independent material properties for Scotchply 1002 in US units
    E11  = 5.6  * (10**6)  # psi
    E22  = 1.2  * (10**6)  # psi
    G12  = 0.6  * (10**6)    # psi
    V12  = 0.26            # unit-less
    V21 = (V12*E22)/E11    # unit-less

    # # Typical strengths of Scotchply 1002 in US units
    SLt  = 154  * (10**3)  # psi
    SLc  = 88.5 * (10**3)  # psi
    STt  = 4.5  * (10**3)  # psi
    STc  = 17.1 * (10**3)  # psi
    SLTs = 10.4 * (10**3)  # psi




    # Independent material properties for AS/3501 graphite epoxy in SI units
    E11  = 138  * (10**9)  # Pascals
    E22  = 8.96 * (10**9)  # Pascals
    G12  = 7.1 * (10**9)  # Pascals
    V12  = 0.3            # unit-less
    V21 = (V12*E22)/E11    # unit-less

    # # Typical strengths of AS/3501 graphite epoxy in SI units
    SLt  = 1447 * (10**6)  # Pascals
    SLc  = 1447 * (10**6)  # Pascals
    STt  = 51.7 * (10**6)  # Pascals
    STc  = 206  * (10**6)  # Pascals
    SLTs = 93   * (10**6)  # Pascals




    # Independent material properties for AS/3501 graphite epoxy in US units
    E11  = 20.01 * (10**6)  # psi
    E22  = 1.3   * (10**6)  # psi
    G12  = 1.03  * (10**6)  # psi
    V12  = 0.3              # unit-less
    V21 = (V12*E22)/E11     # unit-less

    # Typical strengths of AS/3501 graphite epoxy in US units
    SLt  = 209.9 * (10**3)  # psi
    SLc  = 209.9 * (10**3)  # psi
    STt  = 7.5   * (10**3)  # psi
    STc  = 29.9  * (10**3)  # psi
    SLTs = 13.5  * (10**3)  # psi




    # Independent material properties for T300/5208 graphite epoxy in SI units
    E11  = 181  * (10**9)  # Pascals
    E22  = 10.3 * (10**9)  # Pascals
    G12  = 7.17 * (10**9)  # Pascals
    V12  = 0.28            # unit-less
    V21 = (V12*E22)/E11    # unit-less

    # # Typical strengths of T300/5208 graphite epoxy in SI units
    SLt  = 1500 * (10**6)  # Pascals
    SLc  = 1500 * (10**6)  # Pascals
    STt  = 40   * (10**6)  # Pascals
    STc  = 246  * (10**6)  # Pascals
    SLTs = 68   * (10**6)  # Pascals
    '''

    # Independent material properties for T300/5208 graphite epoxy in US units
    E11  = 26.25 * (10**6)  # psi
    E22  = 1.49  * (10**6)  # psi
    G12  = 1.04  * (10**6)  # psi
    V12  = 0.28             # unit-less
    V21 = (V12*E22)/E11     # unit-less

    # Typical strengths of T300/5208 graphite epoxy in US units
    SLt  = 217.5 * (10**3)  # psi
    SLc  = 217.5 * (10**3)  # psi
    STt  = 5.80  * (10**3)  # psi
    STc  = 35.7  * (10**3)  # psi
    SLTs = 9.86  * (10**3)  # psi

    # [Nxx, Nyy, Nxy, Mxx, Myy, Mxy] N/m and N-m/m
    stress_resultant = np.array([[100], [100], [0], [1], [1], [0]])

    # Enter a desired ply orientation angles in degrees here:
    angle_in_degrees = [45,90,-45,45,-45,0,0,0,0,-45,45,-45,90,45]

    N = len(angle_in_degrees)  # total number of plies
    t_ply = 0.005 # ply thickness in inches
    h = t_ply * N

    # Distance from laminate mid-plane to out surfaces of plies)
    z0 = -h / 2
    z = []
    for i in range(N):
        z.append((-h / 2) + ((i + 1) * t_ply))

    # Distance from laminate mid-plane to mid-planes of plies
    z_mid_plane = []
    for i in range(N):
        z_mid_plane.append((-h / 2) - (t_ply / 2) + ((i + 1) * t_ply))

    # Ply orientation angle translated to radians to simplify equations below
    angle = []
    for i in range(N):
        angle.append(math.radians(angle_in_degrees[i]))

    # Stress Transformation (Global to Local)
    T = calculate_T(N ,angle)

    # Strain Transformation (Global-to-Local)
    T_hat = calculate_T_hat(N, angle)

    # Calculating the local (Q) and global (Q_bar) stiffness matrices
    Q_array, Q_bar_array = calculate_Q_and_Q_bar(E11, E22, G12, V12, V21, N, T, T_hat)

    A_array = calculate_A(N, Q_bar_array, t_ply)
    B_array = calculate_B(N, Q_bar_array, z, t_ply)
    D_array = calculate_D(N, Q_bar_array, z, t_ply)

    ABD = calculate_ABD(A_array, B_array, D_array)

    midplane_strains, curvatures = calculate_midplane_strains_and_curvatures(ABD, stress_resultant)

    global_strains = calculate_global_strains(N, midplane_strains, z_mid_plane, curvatures)
    global_stresses = calculate_global_stresses(N, Q_bar_array, midplane_strains)
    local_strains = calculate_local_strains(N, T_hat, midplane_strains)
    local_stresses = calculate_local_stresses(N, Q_array, local_strains)

    R_MS = R_Max_Stress(N, SLt, STt, SLTs, local_stresses)
    R_MS_min = min(R_MS)

    Nxxc_MS = float(R_MS_min * stress_resultant[0])
    Nyyc_MS = float(R_MS_min * stress_resultant[1])
    Nxyc_MS = float(R_MS_min * stress_resultant[2])
    Mxxc_MS = float(R_MS_min * stress_resultant[3])
    Myyc_MS = float(R_MS_min * stress_resultant[4])
    Mxyc_MS = float(R_MS_min * stress_resultant[5])

    a, b, cc, F11, F22, F12, F66, F1, F2 = Tsai_Wu_coeffs(N, SLt, SLc, STt, STc, SLTs, local_stresses)
    R_TW = R_Tsai_Wu(N, a, b, cc)
    R_TW_min = min(R_TW)

    Nxxc_TW = float(R_TW_min * stress_resultant[0])
    Nyyc_TW = float(R_TW_min * stress_resultant[1])
    Nxyc_TW = float(R_TW_min * stress_resultant[2])
    Mxxc_TW = float(R_TW_min * stress_resultant[3])
    Myyc_TW = float(R_TW_min * stress_resultant[4])
    Mxyc_TW = float(R_TW_min * stress_resultant[5])

    # PRINT ALL THE THINGS
    print('[Q] = {}\n'.format(Q_array))

    new_section()
    for i in range(N):
        print('For layer {} [Q_bar[{}]] =\n{}\n\n'.format(i+1, i+1, Q_bar_array[i]))

    new_section()
    print('\n[A] =\n{}\n\n[B] =\n{}\n\n[D] =\n{}\n\n'.format(A_array,B_array,D_array))

    new_section()
    print('\nMidplane strains =\n{}\n'.format(midplane_strains))

    new_section()
    print('\nCurvatures =\n{}\n\n'.format(curvatures))

    new_section()
    for i in range(N):
        print('Global Strains for layer {}=\n{}\n'.format(i+1, global_strains[i]))

    new_section()
    for i in range(N):
        print('Global Stresses for layer {}=\n{}\n'.format(i+1, global_stresses[i]))

    new_section()
    for i in range(N):
        print('Local Strains for layer {}=\n{}\n'.format(i+1, local_strains[i]))

    new_section()
    for i in range(N):
        print('Local Stresses for layer {}=\n{}\n'.format(i+1, local_stresses[i]))

    new_section()
    print('R values for Max Stress Failure:')
    for i in range(N):
        if i > 8:
            if i == N/2:
                print('\nLayer {0}: R = {1:.{2}f}   |   '.format(i + 1, R_MS[i], 2), end='')
            else:
                print('Layer {0}: R = {1:.{2}f}   |   '.format(i+1, R_MS[i], 2),end='')
        else:
            if i == N/2:
                print('\nLayer {0}{1}: R = {2:.{3}f}   |   '.format(0, i + 1, R_MS[i], 2), end='')
            else:
                print('Layer {0}{1}: R = {2:.{3}f}   |   '.format(0, i + 1, R_MS[i], 2), end='')

    print()
    new_section()
    print('These are used to find the Tsai-Wu Coefficients')
    print('F11 = {0:.{6}e}   |   F22 = {1:.{6}e}   |   F12 = {2:.{6}e}\nF66 = {3:.{6}e}   |   F1  = {4:.{6}e}   |   F2  = {5:.{6}e}\n'.format(F11, F22, F12, F66, F1, F2, 2))
    print('These are the Tsai-Wu Coefficients')
    for i in range(N):
        if i < 9:
            print('Layer {5}{4}:     a = {0:.{3}e}   |   b = {1:.{3}e}   |   cc = {2}'.format(a[i],b[i],cc[i], 2, i+1, 0))
        else:
            print('Layer {4}:     a = {0:.{3}e}   |   b = {1:.{3}e}   |   cc = {2}'.format(a[i],b[i],cc[i], 2, i+1, 0))

    new_section()
    print('R values for Tsai-Wu Failure:')
    for i in range(N):
        if i > 8:
            if (i)%4 == 0:
                print('\nLayer {0}: R = {1:.{2}f}   |   '.format(i + 1, R_TW[i], 2), end='')
            else:
                print('Layer {0}: R = {1:.{2}f}   |   '.format(i+1, R_TW[i], 2),end='')
        else:
            if (i)%4 == 0:
                print('\nLayer {0}{1}: R = {2:.{3}f}   |   '.format(0, i + 1, R_TW[i], 2), end='')
            else:
                print('Layer {0}{1}: R = {2:.{3}f}   |   '.format(0, i + 1, R_TW[i], 2), end='')

    print()

    print('\n\n' + format('\033[1m\033[94mSUMMARY OF R VALUES AND CRIT LOADS FOR MAX STRESS THEN TSAI-WU\033[0M', '^200s'))
    for i in range(2):
        print()
        for j in range(200):
            print('_', end='')
    print('\033[0m')

    print()
    print('The minimum strength ratio (\033[1m\033[94mMAX STRESS\033[0m) for this iteration is:\n\033[91mR_MS_min = {0:.{1}f},  This happens in Layer {2}\033[0m'.format(R_MS_min, 2, R_MS.index(min(R_MS))+1))

    new_section()
    print('The Critical Loads under \033[1m\033[94mMAX STRESS\033[0m Failure are:')
    print('\033[91mNxxc = {0:.{3}e}   |   Nyyc = {1:.{3}e}   |   Nxyc = {2:.{3}e}'.format(Nxxc_MS, Nyyc_MS, Nxyc_MS, 2))
    print('Mxxc = {0:.{3}e}   |   Myyc = {1:.{3}e}   |   Mxyc = {2:.{3}e}\033[0m'.format(Mxxc_MS, Myyc_MS, Mxyc_MS, 2))

    new_section()
    print('The minimum strength ratio (\033[1m\033[94mTSAI-WU\033[0m) for this iteration is:\n\033[91mR_TW_min = {0:.{1}f},  This happens in \033[4mLayer {2}\033[0m'.format(R_TW_min, 2, R_TW.index(min(R_TW))+1))

    new_section()
    print('The Critical Loads under \033[1m\033[94mTSAI-WU\033[0m Failure are:')
    print('\033[91mNxxc = {0:.{3}e}   |   Nyyc = {1:.{3}e}   |   Nxyc = {2:.{3}e}'.format(Nxxc_TW, Nyyc_TW, Nxyc_TW, 2))
    print('Mxxc = {0:.{3}e}   |   Myyc = {1:.{3}e}   |   Mxyc = {2:.{3}e}'.format(Mxxc_TW, Myyc_TW, Mxyc_TW, 2))


if __name__ == '__main__':
    main()