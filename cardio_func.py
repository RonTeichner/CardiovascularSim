import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lotkavolterra(t, z, *args):

    #a, b, c, d = args#dictParams["a"], dictParams["b"], dictParams["c"], dictParams["d"]
    dictk = args[0]
    a, b, c, d = dictk["a"], dictk["b"], dictk["c"], dictk["d"]
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]

class ZenkerToolbox:
    def __init__(self):

        self.paramsDict = self.zenkerParams()

    def zenkerParams(self):
        heartParamsDict = {
            "C_PRSW_min": 25.9,  # [mm HG] the minimal contractility
            "C_PRSW_max": 103.8,  # [mm HG] the maximal contractility
            "f_HR_min": 40 / 60,  # [Hz] the minimal heart rate (corresponding to 40 bpm)
            "f_HR_max": 180 / 60,  # [Hz] the maximal heart rate (corresponding to 180 bpm)
            "T_sys": 4 / 15,  # [sec] "time of systole" - 80% of cardiac cycle in a maximal heart rate
            "V_ed_0": 7.1441,  # [ml] ventricular pressure volume relationship (not in the article, from Netta)
            "P_0_lv": 2.0325,  # [mm HG] ventricular pressure volume relationship
            "k_E_lv": 0.0657,  # [] ventricular pressure volume relationship
            "R_valve": 0.0025  # [mm Hg s/ml] atrial resistance
        }

        P_0_lv, R_valve, k_E_lv, V_ed_0 = heartParamsDict["P_0_lv"], heartParamsDict["R_valve"], heartParamsDict[
            "k_E_lv"], heartParamsDict["V_ed_0"]
        heartParamsDict["k1"] = - P_0_lv / R_valve * np.exp(-k_E_lv * V_ed_0)

        systemicParamsDict = {
            "R_TPR_min": 0.5335,  # [mm Hg s/ml] - the minimal peripheral resistance (Netta used 1.2 for kids)
            "R_TPR_max": 2.134,  # [mm Hg s/ml] - the minimal peripheral resistance (Netta used 2.7 for kids)
            "V_a0": 700,  # [ml] - unstressed volume in the arteries (Netta used 20 for kids)
            "V_v0_min": 2700,  # [ml] - minimal venous unstressed volume  (Netta used 900 for kids)
            "V_v0_max": 3100,  # [ml] - maximal venous unstressed volume  (Netta used 1100 for kids)
            "C_a": 4,  # [ml/mm Hg] - the arterial contractility  (Netta used 1 for kids)
            "C_v": 111.11,  # [ml/mm Hg] - the venous contractility
            "V_t": 5000  # [ml] - total blood volume
        }

        controlParamsDict = {
            "P_a_set": 70,  # [mm Hg]
            "K_width": 0.1838,  # [(mm Hg)^{-1}] - the baroreflex parameter
            "tau_Baro": 20  # [sec] - time constant for control of unstressed venous volume
        }

        paramsDict = {
            "heartParamsDict": heartParamsDict,
            "systemicParamsDict": systemicParamsDict,
            "controlParamsDict": controlParamsDict
        }

        return paramsDict

    def runModel(self, simDuration, initList):
        sol = solve_ivp(self.zenkerModel, [0, simDuration], initList, args=[self.paramsDict], dense_output=True)
        return sol

    def zenkerModel(self, t, stateVec, *args):
        # we use prefix s_ to denote state variables and prefix p_ to denote model parameters
        zenkerParamsDict = args[0]
        heartParamsDict, systemicParamsDict, controlParamsDict = zenkerParamsDict["heartParamsDict"], zenkerParamsDict[
            "systemicParamsDict"], zenkerParamsDict["controlParamsDict"]

        s_Ves, s_Ved, s_Va, s_Vv, s_S = stateVec

        # calculating the variables affected by the sympathetic signal:
        p_f_HR_min, p_f_HR_max, p_C_PRSW_min, p_C_PRSW_max = heartParamsDict["f_HR_min"], heartParamsDict["f_HR_max"], \
                                                             heartParamsDict["C_PRSW_min"], heartParamsDict[
                                                                 "C_PRSW_max"]
        p_R_TPR_min, p_R_TPR_max, p_V_v0_min, p_V_v0_max = systemicParamsDict["R_TPR_min"], systemicParamsDict[
            "R_TPR_max"], systemicParamsDict["V_v0_min"], systemicParamsDict["V_v0_max"]
        f_HR = s_S * (p_f_HR_max - p_f_HR_min) + p_f_HR_min
        R_TPR = s_S * (p_R_TPR_max - p_R_TPR_min) + p_R_TPR_min
        C_PRSW = s_S * (p_C_PRSW_max - p_C_PRSW_min) + p_C_PRSW_min
        V_v0 = (1 - s_S) * (p_V_v0_max - p_V_v0_min) + p_V_v0_min

        # calculating Ped - the pressure in the left ventricle at end of diastole
        p_k_E_lv, p_P_0_lv, p_V_ed_0 = heartParamsDict["k_E_lv"], heartParamsDict["P_0_lv"], heartParamsDict["V_ed_0"]
        Ped = p_P_0_lv * (np.exp(
            p_k_E_lv * (s_Ved - p_V_ed_0)) - 1)  # [mmHg] pressure in the left ventricle at end of diastole

        # calculating Pa - the pressure at the arteries
        p_V_a0, p_C_a = systemicParamsDict["V_a0"], systemicParamsDict["C_a"]
        Pa = (s_Va - p_V_a0) / p_C_a

        # calculating tilde_V_es - the current set point of end systolic volume in left ventricle
        hat_V_es = s_Ved - C_PRSW * (s_Ved - p_V_ed_0) / (Pa - Ped)
        if Pa > Ped:
            tilde_V_es = np.max([p_V_ed_0, hat_V_es])
        else:
            tilde_V_es = p_V_ed_0

        # calculating dot_Ves - the current rate of change of the systolic end volume in left ventricle
        dot_Ves = (tilde_V_es - s_Ves) * f_HR

        # calculating Pes - the pressure in the left ventricle at end of systole
        Pes = p_P_0_lv * (np.exp(
            p_k_E_lv * (s_Ves - p_V_ed_0)) - 1)  # [mmHg] pressure in the left ventricle at end of systole

        # calculating Pv - the pressure in the veins
        p_C_v = systemicParamsDict["C_v"]
        Pv = (s_Vv - V_v0) / p_C_v

        # calculating tilde_V_ed - the current set point of end diastolic volume in left ventricle
        p_k1, p_R_valve, p_T_sys = heartParamsDict["k1"], heartParamsDict["R_valve"], heartParamsDict["T_sys"]
        k3 = (p_P_0_lv + Pv) / p_R_valve
        t_diastole = 1 / f_HR - p_T_sys
        hat_V_ed = -(1 / p_k_E_lv) * np.log(
            p_k1 / k3 * (np.exp(-p_k_E_lv * k3 * t_diastole) - 1) + np.exp(-p_k_E_lv * (s_Ves + k3 * t_diastole)))
        if Pv > Pes:
            tilde_V_ed = hat_V_ed
        else:
            tilde_V_ed = s_Ves

        # calculating dot_Ved - the current rate of change of the diastolic end volume in left ventricle
        dot_Ved = (tilde_V_ed - s_Ved) * f_HR

        # calculating blood flows, cardiac ouput and the rate of change in arteries blood volume
        Ic = (Pa - Pv) / p_C_a  # arterio -> venous (capillary blood flow)
        Vs = (s_Ved - s_Ves)  # stroke volume
        Ico = f_HR * Vs  # cardiac output
        dot_Va = Ico - Ic

        # calculating the rate of change in venous blood volume
        dot_Vv = - dot_Va

        # calculating the rate of change in sympathetic signal
        p_K_width, p_P_a_set, p_tau_Baro = controlParamsDict["K_width"], controlParamsDict["P_a_set"], \
                                           controlParamsDict["tau_Baro"]
        logisticFunc = 1 / (1 + np.exp(-p_K_width * (Pa - p_P_a_set)))
        dot_S = (1 / p_tau_Baro) * (1 - logisticFunc - s_S)

        return [dot_Ves, dot_Ved, dot_Va, dot_Vv, dot_S]






