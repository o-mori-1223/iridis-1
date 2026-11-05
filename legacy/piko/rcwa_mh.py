# ファイル名 'rcwa_mh.py'

#!/usr/bin/env python3

from numbers import Complex
import scipy.interpolate, scipy.special, scipy.linalg
import math
import cmath
import numpy as np
import functools

@functools.cache
def Rcwa1d( pol, lambda0, kx0, period, layer, norder):

    """ RCWA for 1D binary grating
    pol: 偏光、'p'または's'
    lambda0: 入射光の波長 (μm)
    kx0: 入射光の面内波数 (1/μm)
    period: 周期 (μm)
    layer: 層構成
    norder: 計算に取り込む回折次数 (±N次まで取り込む場合は2N+1) """

    nlayer = len(layer) # 入射側媒質と透過側媒質を含んだ層数
    depth = np.zeros(nlayer)                    # 各層の厚さ
    metal = np.array([False]*nlayer)            # 層中の媒質が誘電率の虚数部を持つ場合はTrue
    maxsect = max([len( v) for v in layer])//2  # 1周期中の媒質の分割要素数の最大値
    nsect = np.zeros(nlayer, dtype = int)       # 1周期中の媒質の分割要素数
    refra = np.zeros((nlayer, maxsect),dtype=Complex)         # 分割要素の（複素）屈折率
    filfac = np.zeros((nlayer, maxsect))        # 周期で規格化した分割要素の幅

    for j in range(nlayer):         # layerから各パラメーターの抽出
        nsect[j] = len(layer[j])//2
        nsect[0] = 1
        nsect[nlayer-1] = 1
        depth[j] = layer[j][0]
        for i in range(nsect[j]):
            refra[j][i] = layer[j][i*2+1]
            if abs( refra[j][i].imag) > 1e-100:
                metal[j] = True
            filfac[j][i] = layer[j][i*2+2]
        filfac[0][0] = 1.
        filfac[nlayer-1][0] = 1.

    lambda0=lambda0+1.0e-6
    
    k0 = 2.0*math.pi/lambda0        # 真空中の波数
    kc = k0*refra[nlayer-1][0]      # 入射側媒質における波数
    ks = k0*refra[0][0]             # 透過側媒質における波数
    nmax = (norder-1)/2             # 最大回折次数
    I = np.arange(-nmax, nmax+ 1)   # 回折次数の配列

    Zm = np.zeros([norder, norder]) # 零行列
    p = norder//2                   # 配列中の0次の位置
    Eye = np.eye( norder)           # 単位行列
    M = norder-1 # 誘電率分布のフーリエ級数の最大次数

    K = 2.0*math.pi/period          # 格子ベクトル
    kx = kx0+I*K                    # 回折光の面内波数
    #print(f'wl={lambda0},k0={k0},kc={kc},ks={ks},K={K},kx0={kx0},kx={kx}')

    #kzc = np.sqrt((kc**2-kx**2).astype(np.complex)) # 入射側媒質における回折光の波数の法線成分
    kzc = np.sqrt((kc**2-kx**2).astype(complex)) # 入射側媒質における回折光の波数の法線成分
    np.where((kzc.real+kzc.imag)> 0, kzc, - kzc)    # 符号の修正
   

    #kzs = np.sqrt((ks**2-kx**2).astype(np.complex)) # 透過側媒質における回折光の波数の法線成分
    kzs = np.sqrt((ks**2-kx**2).astype(complex)) # 透過側媒質における回折光の波数の法線成分

    if metal[0]:
        np.where((kzs.imag)>0, kzs, - kzs)          # 符号の修正
    else:
        np.where((kzs.real+ kzs.imag)>0, kzs, - kzs) # 符号の修正

    Kx = np.diag( kx)/k0            # 回折光の面内波数の対角行列
    #Kzc = np.diag(kzc)/k0           # 入射側回折光の波数の法線成分の対角行列
    #Kzs = np.diag(kzs)/k0           # 透過側回折光の波数の法線成分の対角行列

    #EpsilonX = np.zeros([nlayer, norder, norder], dtype=np.complex)     # 誘電率のフーリエ係数のToeplitz行列の格納用配列
    #AlphaX = np.zeros([nlayer, norder, norder], dtype=np.complex)       # 誘電率の逆数のフーリエ係数のToeplitz行列の格納用配列
    EpsilonX = np.zeros([nlayer, norder, norder], dtype=np.complex128)     # 誘電率のフーリエ係数のToeplitz行列の格納用配列
    AlphaX = np.zeros([nlayer, norder, norder], dtype=np.complex128)       # 誘電率の逆数のフーリエ係数のToeplitz行列の格納用配列


    for kk in range(0, nlayer):
        if nsect[kk] > 1:
            vX = np.zeros(M*2+1)    # 誘電率のフーリエ係数の格納用配列
            ivX = np.zeros(M*2+1)   # 誘電率の逆数のフーリエ係数の格納用配列

            for jj in range(0, nsect[kk]): # フーリエ級数の計算
                disp = np.sum(filfac[kk][ 0:jj+ 1])- filfac[kk][jj]/2.0
                epsX = refra[ kk][ jj]**2 # permittivity
                # 矩形波のsinc係数の計算
                asinc = filfac[kk][jj]*np.sinc(filfac[kk][jj]*np.arange(1, M+ 1))
                # セクション(jj)の「中心(x=0)仮定」でのフーリエ係数 (ε版)
                vm = epsX*asinc[::-1]                   # 負の次数 (m = -M ... -1)
                v0 = np.array([epsX*filfac[kk][jj]])    # 0次の係数 (m = 0)
                vp = epsX*asinc                         # 正の次数 (m = 1 ... M)
                # 係数の合成、シフト、および層全体への加算 (ε版)
                vX = vX + np.concatenate((vm, v0, vp))*np.exp(-1j* 2*math.pi*disp*np.arange(-M, M+ 1))

                ivm = 1/epsX* asinc[::-1]
                iv0 = np.array([1/epsX*filfac[kk][jj]])
                ivp = 1/epsX*asinc
                ivX = ivX + np.concatenate((ivm, iv0, ivp))*np.exp(-1j*2*math.pi*disp*np.arange(-M, M+ 1))

            EpsilonX[kk, :, :] = scipy.linalg.toeplitz( vX[norder-1: 2* norder-1], vX[norder- 1: : -1]) # 誘電率のフーリエ係数のToeplitz行列の生成
            AlphaX[kk, :, :] = scipy.linalg.toeplitz(ivX[ norder-1: 2* norder-1], ivX[norder-1: : -1])  # 誘電率の逆数のフーリエ係数のToeplitz行列の生成

        else: # 層が均一な場合のToeplitz行列の生成
            EpsilonX[kk, :, :] = Eye* (refra[ kk][0]**2)
            AlphaX[kk, :, :] = Eye/ (refra[kk][0]**2)

    if pol == "s": # s-偏光（TE偏光）の場合
        Rdu = Zm
        Rud = Zm
        Tuu = Eye
        Tdd = Eye
        for ii in range(0, nlayer):

            epsr = refra[ii][0]** 2 # 透過側媒質の誘電率

            if nsect[ii]> 1:
                A = Kx*Kx - EpsilonX[ii, :, :]  # 式(5.14)右辺の行列
                Eigen, W1 = np.linalg.eig(A)    # 式(5.14)右辺の行列の固有値と固有ベクトル
            else:
                W1= Eye # 層が均質な場合の固有ベクトル
                #Eigen = ((kx/k0)**2-epsr).astype(np.complex)    # 層が均質な場合の固有値
                Eigen = ((kx/k0)**2-epsr).astype(complex)    # 層が均質な場合の固有値
            if ii == 0:
                W00 = W1

            Q = np.sqrt(-Eigen) # 式(5.20)の行列Qの対角成分

            if metal[ii]:
                Q = np.where(Q.imag>0.0, Q, -Q)             # 符号の修正
            else:
                Q = np.where( (Q.real+ Q.imag)>0.0, Q, -Q)  # 符号の修正

            V1 = np.dot(W1, np.diag(Q))     # 式(5.20)
            if ii> 0:
                Q1 = np.dot(np.linalg.inv(W1), W0) # 式(5.118)
                Q2 = np.dot(np.linalg.inv(V1), V0) # 式(5.118)
                RudTilde = np.dot(Phip, np.dot(Rud, Phip)) # 式(5.110)
                TddTilde = np.dot(Tdd, Phip)    # 式(5.111)
                F = np.dot(Q1, Eye+ RudTilde)   # 式(5.116)
                G = np.dot(Q2, Eye- RudTilde)   # 式(5.117)

                Tau = np.linalg.inv(F+ G)       # 式(5.117)
                Rud = Eye- 2.0* np.dot(G, Tau)  # 式(5.120)
                Tdd = 2.0*np.dot(TddTilde, Tau) # 式(5.121)

            if ii != nlayer-1:
                Phip = np.diag(np.exp(1j*k0*Q*depth[ ii]))  # 式(5.25)のΦ+
                W0 = W1
                V0 = V1

        Rud = np.dot( np.dot(W1,Rud), np.linalg.inv(W1))    # 式(5.131)の右辺（iを除く）
        Tdd = np.dot( np.dot(W00,Tdd), np.linalg.inv(W1))   # 式(5.132)右辺（iを除く）

        Rs = Rud[:, p] # 式(5.131)
        Ts = Tdd[:, p] # 式(5.132)

        IR = (np.abs( Rs)**2)*np.real(kzc)/np.real(kzc[p]) # 反射回折効率
        IT = (np.abs( Ts)**2)*np.real(kzs)/np.real(kzc[p]) # 透過回折効率



    else: # p-偏光（TM偏光）の場合
        Rdu = Zm
        Rud = Zm
        Tuu = Eye
        Tdd = Eye

        for ii in range(0, nlayer):
            epsr = refra[ii][0]**2 # 透過側媒質の誘電率
            if nsect[ii]> 1:
                A = np.dot( Kx, np.dot(np.linalg.inv( EpsilonX[ii, :, :]), Kx)) - Eye   # 式(5.39)右辺の括弧内
                Eigen, W1 = np.linalg.eig(np.dot( np.linalg.inv( AlphaX[ii, :, :]), A)) # 式(5.39)右辺の行列の固有値と固有ベクトル
            else:
                W1 = Eye # 層が均質な場合の固有ベクトル
                #Eigen = ((kx/k0)**2-epsr).astype(np.complex)    # 層が均質な場合の固有値
                Eigen = ((kx/k0)**2-epsr).astype(complex)    # 層が均質な場合の固有値

            if ii == 0:
                W00 = W1

            Q = np.sqrt(-Eigen) # 式(5.39)の行列Qの対角成分

            if metal[ii]:
                Q = np.where(Q.imag>0.0, Q, -Q) # 符号の修正
            else:
                Q = np.where((Q.real+Q.imag)>0.0, Q, -Q) # 符号の修正

            if nsect[ii]> 1:
                V1 = np.dot(np.dot(AlphaX[ii, :, :], W1), np.diag(Q))   # 式(5.47)
            else:
                V1 = np.diag(Q)/epsr                                    # 式(5.47)

            if ii > 0:
                Q1 = np.dot(np.linalg.inv(W1), W0) # 式(5.118)
                Q2 = np.dot(np.linalg.inv(V1), V0) # 式(5.118)
                RudTilde = np.dot(np.dot(Phip, Rud), Phip) # 式(5.110)
                TddTilde = np.dot(Tdd, Phip)    # 式(5.111)
                F = np.dot(Q1, (Eye+ RudTilde)) # 式(5.116)
                G = np.dot(Q2, (Eye- RudTilde)) # 式(5.117)
                Tau = 2.0*np.linalg.inv(F+ G)   # 式(5.117)
                Rud = Eye-np.dot(G, Tau)    # 式(5.120)
                Tdd = np.dot(TddTilde, Tau) # 式(5.121)

            if ii != nlayer-1:
                Phip = np.diag(np.exp(1j* k0* Q* depth[ ii]))   # 式(5.25)のΦ+
                W0 = W1
                V0 = V1

        Rud = np.dot(np.dot(W1, Rud), np.linalg.inv(W1))    # 式(5.131)の右辺（iを除く）
        Tdd = np.dot(np.dot(W00, Tdd), np.linalg.inv(W1))   # 式(5.132)の右辺（iを除く）

        Rp = Rud[:, p] # 式(5.131)
        Tp = Tdd[:, p] # 式(5.132)
        IR = (np.abs( Rp)** 2)*np.real( kzc)/np.real(kzc[ p]) # 反射回折効率
        IT = (np.abs( Tp)** 2)*np.real( kzs/refra[0][0]** 2) / np.real( kzc[p]/ refra[nlayer-1][0]** 2) # 透過回折効率

    return IR, IT


# if __name__ == "__main__":

#     import matplotlib.pyplot as plt

#     layer= ( ( 0, 1.5, 1.0), ( 0.25, 1.5, 1/2, 1.0, 1/3, 1.5, 1/6), ( 0.25, 1.5, 1/3, 1.0, 2/3), ( 0, 1.0, 1.0)) # 層構成
#     pitch = 1. # 周期（μm）
#     norder = 21 # 計算にとり入れる回折次数（2N+1）
#     disporder = range(-2,3) # 表示する回折次数
#     angle = 30*math.pi/180 # 入射角（rad）
#     # wl_start = 0.5 + 1e-10 # 計算開始波長（μm）
#     wl_start = 0.6 # 計算開始波長（μm）
#     wl_end = 1.5 # 計算終了波長（μm）
#     wl = np.linspace(wl_start, wl_end, 200) # 計算波長の配列
#     imax = len(wl)
#     ir = np.zeros([imax, norder]) # 反射回折効率の格納用
#     it = np.zeros([imax, norder]) # 透過回折効率の格納用

#     for i in range( 0, imax):
#         ir[i,:], it[i,:] = Rcwa1d('p', wl[ i], 2* math.pi*math.sin( angle)/wl[ i], pitch, layer, norder)    # RCWAの呼び出し
    
#     sum_ir=np.sum(ir,1)
#     sum_it=np.sum(it,1)

#     plt.figure(1) # 透過回折効率の表示
#     lines = ('solid', 'dashed', 'dashdot', 'dotted', 'solid')
#     for m in disporder:
#          plt.plot(wl, it[:, m+ norder//2], label= "m = {0}".format(m),  linewidth= 3, linestyle= lines[m-disporder[0]])

#     plt.xlim( wl_start, wl_end)
#     plt.xlabel('Wavelength ($\mu$m)', fontsize=16)
#     plt.ylabel('Transmittance', fontsize=16)
#     plt.legend(loc='center', frameon=False, fontsize=16)

#     plt.figure(2) # 反射回折効率の表示

#     for m in disporder:
#          plt.plot(wl, ir[:, m+norder//2], label="m = {0}".format(m),  linewidth=3, linestyle=lines[m-disporder[0]])

#     plt.xlim(wl_start, wl_end)
#     plt.xlabel('Wavelength ($\mu$m)', fontsize=16)
#     plt.ylabel('Reflectance', fontsize=16)
#     plt.legend(loc='center', frameon=False, fontsize=16)

#     plt.show()