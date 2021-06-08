import numpy as np


def zenkerParams():

    heartParamsDict = {
        "C_PRSW_min": 25.9,     # [mm HG] the minimal contractility
        "C_PRSW_max": 103.8,    # [mm HG] the maximal contractility
        "f_HR_min": 40/60,      # [Hz] the minimal heart rate (corresponding to 40 bpm)
        "f_HR_max": 180/60,     # [Hz] the maximal heart rate (corresponding to 180 bpm)
        "T_sys": 4/15,          # [sec] "time of systole" - 80% of cardiac cycle in a maximal heart rate
        "V_ed_0": 7.1441,       # [ml] ventricular pressure volume relationship (not in the article, from Netta)
        "P_0_lv": 2.0325,       # [mm HG] ventricular pressure volume relationship
        "k_E_lv": 0.0657,       # [] ventricular pressure volume relationship
        "R_valve": 0.0025  # [mm Hg s/ml] atrial resistance
    }

    P_0_lv, R_valve, k_E_lv, V_ed_0 = heartParamsDict["P_0_lv"], heartParamsDict["R_valve"], heartParamsDict["k_E_lv"], heartParamsDict["V_ed_0"]
    heartParamsDict["k1"] = - P_0_lv/R_valve * np.exp(-k_E_lv*V_ed_0)

    systemicParamsDict = {
        "R_TPR_min": 0.5335,    # [mm Hg s/ml] - the minimal peripheral resistance (Netta used 1.2 for kids)
        "R_TPR_max": 2.134,     # [mm Hg s/ml] - the minimal peripheral resistance (Netta used 2.7 for kids)
        "V_a0": 700,            # [ml] - unstressed volume in the arteries (Netta used 20 for kids)
        "V_v0_min": 2700,       # [ml] - minimal venous unstressed volume  (Netta used 900 for kids)
        "V_v0_max": 3100,       # [ml] - maximal venous unstressed volume  (Netta used 1100 for kids)
        "C_a": 4,               # [ml/mm Hg] - the arterial contractility  (Netta used 1 for kids)
        "C_v": 111.11           # [ml/mm Hg] - the venous contractility
    }

    controlParams = {
        "P_a_set": 70,          # [mm Hg]
        "K_width": 0.1838,      # [(mm Hg)^{-1}] - the baroreflex parameter
        "tau_Baro": 20          # [sec] - time constant for control of unstressed venous volume
    }


    paramsDict = {
        "heartParamsDict": heartParamsDict,
        "systemicParamsDict": systemicParamsDict,
        "controlParams": controlParams
    }

    return paramsDict