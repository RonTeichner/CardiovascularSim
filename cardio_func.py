import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.signal import unique_roots
import matplotlib.pyplot as plt

def lotkavolterra(t, z, *args):

    #a, b, c, d = args#dictParams["a"], dictParams["b"], dictParams["c"], dictParams["d"]
    dictk = args[0]
    a, b, c, d = dictk["a"], dictk["b"], dictk["c"], dictk["d"]
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]

class EpidemicToolbox:
    def __init__(self):
        self.paramsDict = self.epidemicParams()

    def epidemicParams(self):
        rateParamsDict = {
            "a": 0.5,
            "b": 0.5,
            "c": 1.0,
            "N": 1000.0
        }

        paramsDict = {
            "rateParamsDict": rateParamsDict
        }

        return paramsDict

    def epidemicModelNoExternalInput(self, t, stateVecNoInput, *args):
        # we use prefix s_ to denote state variables and prefix p_ to denote model parameters
        # stateVec: [S, I]
        epidemicParamsDict = args[0]
        rateParamsDict = epidemicParamsDict["rateParamsDict"]

        p_a, p_b, p_c, p_N = rateParamsDict["a"], rateParamsDict["b"], rateParamsDict["c"], rateParamsDict["N"]
        s_S, s_I = stateVecNoInput

        dot_S = -p_a*s_S*s_I + p_c*(p_N - s_S - s_I)
        dot_I = p_a*s_S*s_I - p_b*s_I

        return [dot_S, dot_I]

    def epidemicModelNoExternalInput_rootWrapper(self, stateVecNoInput, *args):
        # stateVec: [S, I]
        epidemicParamsDict = args[0]
        t=0 # not in use
        return self.epidemicModelNoExternalInput(t, stateVecNoInput, epidemicParamsDict)

    def computeFixedPoints(self, initialGuess):

        return root(self.epidemicModelNoExternalInput_rootWrapper, initialGuess, self.paramsDict)

    def computeTheoreticalFixedPoints(self):
        rateParamsDict = self.paramsDict["rateParamsDict"]
        p_a, p_b, p_c, p_N = rateParamsDict["a"], rateParamsDict["b"], rateParamsDict["c"], rateParamsDict["N"]
        x0 = [p_N, 0]
        x1 = [p_b/p_a, p_c*(p_a*p_N-p_b)/(p_a*(p_b+p_c))]
        return [x0, x1]


class ZenkerToolbox:
    def __init__(self):

        self.paramsDict = self.zenkerParams()

        self.enableBaroreflexControl = True

        # Empiric figure limits
        self.ylimDict = {
            "Ves": [7, 15], # [7, 12]
            "Ved": [0, 70],
            "Vs": [0, 70],
            "Va": [500, 2200],
            "S": [0, 1],
            "CO": [0, 5000],
            "Pa": [-20, 200], # [-20, 80]
            "Pv": [8, 15]
        }

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
            "R_TPR_max": 2.134,  # [mm Hg s/ml] - the maximal peripheral resistance (Netta used 2.7 for kids)
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

        displayParamsDict = {
            "fs": 100  # [Hz] - figures time resolution
        }

        paramsDict = {
            "heartParamsDict": heartParamsDict,
            "systemicParamsDict": systemicParamsDict,
            "controlParamsDict": controlParamsDict,
            "displayParamsDict": displayParamsDict
        }

        return paramsDict

    def runModel(self, simDuration, initArray, enableExternalInput):
        batchSize = initArray.shape[2]
        solList = list()
        for b in range(batchSize):
            initList = initArray[:, 0, b].tolist()
            if enableExternalInput:
                sol = solve_ivp(self.zenkerModel, [0, simDuration], initList, args=[self.paramsDict], dense_output=True)
            else:
                sol = solve_ivp(self.zenkerModelNoExternalInput, [0, simDuration], initList, args=[self.paramsDict], dense_output=True, rtol=1e-9)
            solList.append(sol)

        return solList

    def zenkerModelNoExternalInput_rootWrapper(self, stateVecNoInput, *args):
        # the blood volume in the venous is resultant
        # stateVec: [Ves, Ved, Va, S]
        zenkerParamsDict = args[0]
        t=0 # not in use
        return self.zenkerModelNoExternalInput(t, stateVecNoInput.tolist(), zenkerParamsDict)

    def zenkerModelNoExternalInput(self, t, stateVecNoInput, *args):
        # the blood volume in the venous is resultant
        # stateVec: [Ves, Ved, Va, S] (must be list to be compatible with solve_ivp)
        zenkerParamsDict = args[0]
        heartParamsDict, systemicParamsDict, controlParamsDict = zenkerParamsDict["heartParamsDict"], zenkerParamsDict["systemicParamsDict"], zenkerParamsDict["controlParamsDict"]

        _, Vs, _, _, Vv = self.computeResultantVariables(np.asarray(stateVecNoInput)[None, None, :])
        stateVecComplete = [stateVecNoInput[0], stateVecNoInput[1], stateVecNoInput[2], Vv[0,0], stateVecNoInput[3]]

        dotList = self.zenkerModel(t, stateVecComplete, zenkerParamsDict)
        dot_Ves, dot_Ved, dot_Va, dot_Vv, dot_S = dotList

        return [dot_Ves, dot_Ved, dot_Va, dot_S]

    def zenkerModel(self, t, stateVec, *args):
        # we use prefix s_ to denote state variables and prefix p_ to denote model parameters
        # stateVec: [Ves, Ved, Va, Vv, S]
        zenkerParamsDict = args[0]
        heartParamsDict, systemicParamsDict, controlParamsDict = zenkerParamsDict["heartParamsDict"], zenkerParamsDict["systemicParamsDict"], zenkerParamsDict["controlParamsDict"]

        s_Ves, s_Ved, s_Va, s_Vv, s_S = stateVec

        # calculating the variables affected by the sympathetic signal:
        p_f_HR_min, p_f_HR_max, p_C_PRSW_min, p_C_PRSW_max = heartParamsDict["f_HR_min"], heartParamsDict["f_HR_max"], heartParamsDict["C_PRSW_min"], heartParamsDict["C_PRSW_max"]
        p_R_TPR_min, p_R_TPR_max, p_V_v0_min, p_V_v0_max = systemicParamsDict["R_TPR_min"], systemicParamsDict["R_TPR_max"], systemicParamsDict["V_v0_min"], systemicParamsDict["V_v0_max"]
        f_HR = s_S * (p_f_HR_max - p_f_HR_min) + p_f_HR_min  # [Hz]
        R_TPR = s_S * (p_R_TPR_max - p_R_TPR_min) + p_R_TPR_min
        C_PRSW = s_S * (p_C_PRSW_max - p_C_PRSW_min) + p_C_PRSW_min
        V_v0 = (1 - s_S) * (p_V_v0_max - p_V_v0_min) + p_V_v0_min

        # calculating Ped - the pressure in the left ventricle at end of diastole
        p_k_E_lv, p_P_0_lv, p_V_ed_0 = heartParamsDict["k_E_lv"], heartParamsDict["P_0_lv"], heartParamsDict["V_ed_0"]
        Ped = p_P_0_lv * (np.exp(p_k_E_lv * (s_Ved - p_V_ed_0)) - 1)  # [mmHg] pressure in the left ventricle at end of diastole

        # calculating Pa - the pressure at the arteries
        p_V_a0, p_C_a = systemicParamsDict["V_a0"], systemicParamsDict["C_a"]
        Pa = (s_Va - p_V_a0) / p_C_a


        if Pa > Ped:
            # calculating tilde_V_es - the current set point of end systolic volume in left ventricle
            hat_V_es = s_Ved - C_PRSW * (s_Ved - p_V_ed_0) / (Pa - Ped)
            tilde_V_es = np.max([p_V_ed_0, hat_V_es])
        else:
            tilde_V_es = p_V_ed_0

        # calculating dot_Ves - the current rate of change of the systolic end volume in left ventricle
        dot_Ves = (tilde_V_es - s_Ves) * f_HR

        # calculating Pes - the pressure in the left ventricle at end of systole
        Pes = p_P_0_lv * (np.exp(p_k_E_lv * (s_Ves - p_V_ed_0)) - 1)  # [mmHg] pressure in the left ventricle at end of systole

        # calculating Pv - the pressure in the veins
        p_C_v = systemicParamsDict["C_v"]
        Pv = (s_Vv - V_v0) / p_C_v

        # calculating tilde_V_ed - the current set point of end diastolic volume in left ventricle
        p_k1, p_R_valve, p_T_sys = heartParamsDict["k1"], heartParamsDict["R_valve"], heartParamsDict["T_sys"]
        k3 = (p_P_0_lv + Pv) / p_R_valve
        t_diastole = 1 / f_HR - p_T_sys

        if Pv > Pes:
            hat_V_ed = -(1 / p_k_E_lv) * np.log(p_k1 / k3 * (np.exp(-p_k_E_lv * k3 * t_diastole) - 1) + np.exp(-p_k_E_lv * (s_Ves + k3 * t_diastole)))
            tilde_V_ed = hat_V_ed
        else:
            tilde_V_ed = s_Ves

        # calculating dot_Ved - the current rate of change of the diastolic end volume in left ventricle
        dot_Ved = (tilde_V_ed - s_Ved) * f_HR

        # calculating blood flows, cardiac ouput and the rate of change in arteries blood volume
        Ic = (Pa - Pv) / R_TPR  # arterio -> venous (capillary blood flow)
        Vs = (s_Ved - s_Ves)  # stroke volume
        Ico = f_HR * Vs  # cardiac output
        dot_Va = Ico - Ic

        # calculating the rate of change in venous blood volume
        dot_Vv = - dot_Va

        # calculating the rate of change in sympathetic signal
        if self.enableBaroreflexControl:
            p_K_width, p_P_a_set, p_tau_Baro = controlParamsDict["K_width"], controlParamsDict["P_a_set"], controlParamsDict["tau_Baro"]
            logisticFunc = 1 / (1 + np.exp(-p_K_width * (Pa - p_P_a_set)))
            dot_S = (1 / p_tau_Baro) * (1 - logisticFunc - s_S)
        else:
            dot_S = 0

        return [dot_Ves, dot_Ved, dot_Va, dot_Vv, dot_S]

    def computeFixedPoints(self, batchSize, thr, initialGuess=0):
        # because the function root() might not find all fixed points we call it multiple times with random initial guesses and collect the fixed points. Then we find the unique fixed points
        if batchSize > 0:
            fixedPointsList = list()
            initValues = self.randomizeInitValues(batchSize)
            for b in range(batchSize):
                fixedPoints = root(self.zenkerModelNoExternalInput_rootWrapper, initValues[:, 0, b], self.paramsDict)
                #  s_Ves, s_Ved, s_Va, s_S  (order of values in fixedPoints)
                cardiacOutput = fixedPoints.x[1] - fixedPoints.x[0]
                if fixedPoints.success and cardiacOutput >= 0: fixedPointsList.append(fixedPoints.x)
            fixedPointsArray = np.asarray(fixedPointsList)
            fixedPointsArray = thr * np.round(fixedPointsArray/thr) # rounding to threshold
            uniqueFixedPoints = np.unique(fixedPointsArray, axis=0)

        else:
            uniqueFixedPoints = root(self.zenkerModelNoExternalInput_rootWrapper, initialGuess, self.paramsDict)

        return uniqueFixedPoints

    def randomizeInitValues(self, batchSize):
        zenkerParamsDict = self.paramsDict
        heartParamsDict, systemicParamsDict, controlParamsDict = zenkerParamsDict["heartParamsDict"], zenkerParamsDict["systemicParamsDict"], zenkerParamsDict["controlParamsDict"]
        p_totalBloodVol = systemicParamsDict["V_t"]

        # draw init values, including non physiological init values
        # every column in self.initValues is [Ves; Ved; Va; S] so that the total blood volume is constant

        # draw some numbers for the volumes:
        Ves, Va, Vv = np.random.rand(1, batchSize), np.random.rand(1, batchSize), np.random.rand(1, batchSize)
        Ved = Ves + np.random.rand(1, batchSize) # otherwise the stroke volume, Ved-Ves is negative, In addition, rate up to ten between them sounds reasonable
        Ved2Ves = np.divide(Ved, Ves)
        # normalize to have proper total blood volume:
        Vs = Ved - Ves  # stroke volume
        totalBloodVolPreNormalization = Va + Vv + 0.5*Vs   # we assume the mean blood volume in the heart to be half the stroke volume
        unnormalizedVolumes = np.concatenate((Vs, Va, Vv), axis=0)
        normalizedVolumes = unnormalizedVolumes * p_totalBloodVol/totalBloodVolPreNormalization

        Vs_normalized, Va_normalized, Vv_normalized = normalizedVolumes[0:1, :], normalizedVolumes[1:2, :], normalizedVolumes[2:3, :]

        if False: # sanity check for total blood volume
            totalBloodVolNormalized = Va_normalized + Vv_normalized + 0.5*Vs_normalized

        Ves_normalized = np.divide(Vs_normalized, Ved2Ves - 1)
        Ved_normalized = Vs_normalized + Ves_normalized

        initSympatheticSig = np.random.rand(1, batchSize)

        self.initValues = np.concatenate((Ves_normalized, Ved_normalized, Va_normalized, initSympatheticSig), axis=0)  # not including Vv in the init values
        self.initValues = self.initValues[:, None, :]  # standartization of [observable, time, batch]
        return self.initValues

    def runModelOnBatchOfInitValues(self, batchSize, simDuration, enableExternalInput):
        solList = self.runModel(simDuration, self.randomizeInitValues(batchSize), enableExternalInput)

        p_fs = self.paramsDict["displayParamsDict"]["fs"]
        tVec = np.linspace(0, simDuration - 1/p_fs, p_fs*simDuration)

        stateVec = np.zeros((tVec.shape[0], batchSize, 4))
        for b, sol in enumerate(solList):
            stateVec[:, b] = np.transpose(sol.sol(tVec))

        return stateVec

    def computeResultantVariables(self, stateVec):
        zenkerParamsDict = self.paramsDict
        heartParamsDict, systemicParamsDict, controlParamsDict, displayParamsDict = zenkerParamsDict["heartParamsDict"], zenkerParamsDict["systemicParamsDict"], zenkerParamsDict["controlParamsDict"], zenkerParamsDict["displayParamsDict"]
        p_V_a0, p_C_a = systemicParamsDict["V_a0"], systemicParamsDict["C_a"]
        p_C_v = systemicParamsDict["C_v"]
        p_V_v0_min, p_V_v0_max = systemicParamsDict["V_v0_min"], systemicParamsDict["V_v0_max"]
        p_totalBloodVol = systemicParamsDict["V_t"]

        s_Ves, s_Ved, s_Va, s_S = stateVec[:, :, 0], stateVec[:, :, 1], stateVec[:, :, 2], stateVec[:, :, 3]

        Vs = s_Ved - s_Ves  # stroke volume

        p_f_HR_min, p_f_HR_max = heartParamsDict["f_HR_min"], heartParamsDict["f_HR_max"]
        f_HR = s_S * (p_f_HR_max - p_f_HR_min) + p_f_HR_min  # [Hz]

        cardiacOutput = 60 * Vs * f_HR # [ml/min]

        V_v0 = (1 - s_S) * (p_V_v0_max - p_V_v0_min) + p_V_v0_min
        Vv = p_totalBloodVol - (0.5 * Vs + s_Va)  # we assume the mean blood volume in the heart to be half the stroke volume

        Pa = (s_Va - p_V_a0)/p_C_a
        Pv = (Vv - V_v0) / p_C_v

        return cardiacOutput, Vs, Pa, Pv, Vv

    def fixedPoint_wrt_parameterChange(self, parameterSubDictionaryName, parameterName, parameterUnits, minVal, maxVal, nValues, batchSize, thr):
        nominalParameterValue = self.paramsDict[parameterSubDictionaryName][parameterName]
        parameterValues = np.linspace(minVal, maxVal, nValues).tolist()
        fixedPoints_wrt_parameterVal = list()
        for i, parameterValue in enumerate(parameterValues):
            self.paramsDict[parameterSubDictionaryName][parameterName] = parameterValue
            fixedPoints = self.computeFixedPoints(batchSize, thr)
            if fixedPoints.shape[0] == 0:
                fixedPoints_wrt_parameterVal.append(fixedPoints)
            else:
                #  s_Ves, s_Ved, s_Va, s_S  (order of values in fixedPoints)
                assert fixedPoints.shape[1] > 1
                fixedPoints_wrt_parameterVal.append(fixedPoints[0, :])

        # excluding values with no fixed point
        fixedPoints_wrt_parameterVal_ValidList, parameterValues_ValidList = list(), list()
        for i, fixedPoint in enumerate(fixedPoints_wrt_parameterVal):
            if fixedPoint.shape[0] > 0:  # not empty
                fixedPoints_wrt_parameterVal_ValidList.append(fixedPoint)
                parameterValues_ValidList.append(parameterValues[i])

        stateVec = np.asarray(fixedPoints_wrt_parameterVal_ValidList)[:, None, :]  # [nValues x 1 x #stateVec]
        figTitle = "fixed point wrt " + parameterName + "; nominalValue = " + np.array_str(np.array(nominalParameterValue)) + " [" + parameterUnits + "]"
        xlabel = parameterName + " [" + parameterUnits + "]"
        fileName = "fixed point wrt " + parameterName + ".png"

        self.printStateVec(stateVec, figTitle, fileName, parameterValues_ValidList, xlabel, '.', True)
        # returning the parameter value to the nominal value
        self.paramsDict[parameterSubDictionaryName][parameterName] = nominalParameterValue


    def printStateVec(self, stateVec, figTitle, fileName, xVec, xlabel, lineStyle='-', useYlim=False):

        s_Ves, s_Ved, s_Va, s_S = stateVec[:, :, 0], stateVec[:, :, 1], stateVec[:, :, 2], stateVec[:, :, 3]

        cardiacOutput, Vs, Pa, Pv, _ = self.computeResultantVariables(stateVec)

        fig, axs = plt.subplots(3, 3, sharex=True, sharey=False, figsize=(16,16))

        fig.suptitle(figTitle, fontsize=16)

        axs[0, 0].plot(xVec, s_Ves, lineStyle, label='Ves')
        #axs[0, 0].set_xlabel(xlabel)
        axs[0, 0].set_ylabel('ml')
        axs[0, 0].set_title('Ves')
        if useYlim: axs[0, 0].set_ylim(self.ylimDict["Ves"])

        axs[1, 0].plot(xVec, s_Ved, lineStyle, label='Ved')
        #axs[1, 0].set_xlabel(xlabel)
        axs[1, 0].set_ylabel('ml')
        axs[1, 0].set_title('Ved')
        if useYlim: axs[1, 0].set_ylim(self.ylimDict["Ved"])

        axs[0, 1].plot(xVec, s_Va, lineStyle, label='Va')
        #axs[0, 1].set_xlabel(xlabel)
        axs[0, 1].set_ylabel('ml')
        axs[0, 1].set_title('Va')
        if useYlim: axs[0, 1].set_ylim(self.ylimDict["Va"])

        axs[1, 1].plot(xVec, s_S, lineStyle, label='S')
        #axs[1, 1].set_xlabel(xlabel)
        axs[1, 1].set_title('S')
        if useYlim: axs[1, 1].set_ylim(self.ylimDict["S"])

        axs[2, 1].plot(xVec, cardiacOutput, lineStyle)
        axs[2, 1].set_xlabel(xlabel)
        axs[2, 1].set_ylabel('ml / min')
        axs[2, 1].set_title('cardiac output')
        if useYlim: axs[2, 1].set_ylim(self.ylimDict["CO"])

        axs[2, 0].plot(xVec, Vs, lineStyle)
        axs[2, 0].set_xlabel(xlabel)
        axs[2, 0].set_ylabel('ml')
        axs[2, 0].set_title('stroke vol.')
        if useYlim: axs[2, 0].set_ylim(self.ylimDict["Vs"])

        axs[0, 2].plot(xVec, Pa, lineStyle)
        #axs[0, 2].set_xlabel(xlabel)
        axs[0, 2].set_ylabel('mm Hg')
        axs[0, 2].set_title('Pa')
        if useYlim: axs[0, 2].set_ylim(self.ylimDict["Pa"])

        axs[1, 2].plot(xVec, Pv, lineStyle)
        axs[1, 2].set_xlabel(xlabel)
        axs[1, 2].set_ylabel('mm Hg')
        axs[1, 2].set_title('Pv')
        if useYlim: axs[1, 2].set_ylim(self.ylimDict["Pv"])

        plt.savefig(fileName, dpi=150)


    def printModelTrajectories(self, stateVec, figTitle, fileName, useYlim=False):
        batchSize, nSamples = stateVec.shape[1], stateVec.shape[0]

        zenkerParamsDict = self.paramsDict
        heartParamsDict, systemicParamsDict, controlParamsDict, displayParamsDict = zenkerParamsDict["heartParamsDict"], zenkerParamsDict["systemicParamsDict"], zenkerParamsDict["controlParamsDict"], zenkerParamsDict["displayParamsDict"]

        p_fs = displayParamsDict["fs"]  # [Hz]
        p_totalBloodVol = systemicParamsDict["V_t"]  # [ml]

        tVec = (1/p_fs) * np.arange(0, nSamples)  # [sec]

        self.printStateVec(stateVec, figTitle, fileName, tVec, "sec", useYlim=useYlim)

    def plot_Vlv(self):
        fs = 1000
        duration = 1  # sec
        tVec = np.arange(0, int(np.ceil(fs*duration)))/fs

        heartParamsDict, systemicParamsDict, controlParamsDict = self.paramsDict["heartParamsDict"], self.paramsDict["systemicParamsDict"], self.paramsDict["controlParamsDict"]

        k2 = heartParamsDict["k_E_lv"]
        k1 = heartParamsDict["k1"]
        p_P_0_lv, p_R_valve = heartParamsDict["P_0_lv"], heartParamsDict["R_valve"]
        Pv, Ves = 9.83, 7.142  # taken manually from simulation
        k3 = (p_P_0_lv + Pv) / p_R_valve

        Vlv = -(1 / k2) * np.log(k1 / k3 * (np.exp(-k2 * k3 * tVec) - 1) + np.exp(-k2 * (Ves + k3 * tVec)))
        plt.figure()
        plt.plot(tVec, Vlv)
        plt.xlabel('sec')
        plt.ylabel('ml')

    def plot_S_vs_Pa_nullcline(self):
        zenkerParamsDict = self.paramsDict
        heartParamsDict, systemicParamsDict, controlParamsDict, displayParamsDict = zenkerParamsDict["heartParamsDict"], zenkerParamsDict["systemicParamsDict"], zenkerParamsDict["controlParamsDict"], zenkerParamsDict["displayParamsDict"]

        Pa_vec = np.arange(20, 140, 0.1)
        p_K_width, p_P_a_set, p_tau_Baro = controlParamsDict["K_width"], controlParamsDict["P_a_set"], controlParamsDict["tau_Baro"]
        logisticFunc = 1 / (1 + np.exp(-p_K_width * (Pa_vec - p_P_a_set)))
        S_vec = 1 - logisticFunc
        plt.figure()
        plt.plot(Pa_vec - p_P_a_set, S_vec)
        plt.xlabel(r'$P_a - P_{set}$')
        plt.ylabel('S')
        plt.grid()
        plt.title(r'$S(P_a)$ nullcline; $\dot{S}=0$')
        plt.savefig('S_Pa_nullcline.png', dpi=150)
















