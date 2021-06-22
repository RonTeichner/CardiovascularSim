import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cardio_func import *
"""
dictk = dict()
dictk["a"], dictk["b"], dictk["c"], dictk["d"] = 1.5, 1, 3, 1
#listk = [dictk]  #[1.5, 1, 3, 1]
sol = solve_ivp(lotkavolterra, [0, 15], [10, 5], args=[dictk], dense_output=True)

t = np.linspace(0, 15, 300)
z = sol.sol(t)

plt.plot(t, z.T)
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=True)
plt.title('Lotka-Volterra System')
plt.show()
"""

epidemicComunity = EpidemicToolbox()
fixedPoints = epidemicComunity.computeFixedPoints(initialGuess = [500.0, 500.0])
theoreticalFixedPoints = epidemicComunity.computeTheoreticalFixedPoints()

ZenkerPatient = ZenkerToolbox()
totalBloodVol = ZenkerPatient.paramsDict["systemicParamsDict"]["V_t"]

# state init values:
V_es_init = 20  # [ml]
V_ed_init = 70  # [ml]
V_a_init = 450  # [ml]
V_s = V_ed_init - V_es_init
V_v_init = totalBloodVol - (0.5*V_s + V_a_init)
S_init = 0

initArrayNoInput = np.array([[V_es_init], [V_ed_init], [V_a_init], [S_init]])
initArrayNoInput = initArrayNoInput[:, :, None]  # standartization of [observable, time, batch]

#fixedPoints = ZenkerPatient.computeFixedPoints(batchSize=1000, thr=1e-3)
#print(f'Zenker model fixed point found at [Ves, Ved, Va, S] = {fixedPoints}')

simDuration = 100  # [sec]
nValues, batchSize = 100, 500

print('starting fixed point with respect to parameter change')

ZenkerPatient.fixedPoint_wrt_parameterChange("heartParamsDict", "C_PRSW_max", parameterUnits = 'mm Hg', minVal=70, maxVal=150, nValues=nValues, batchSize=batchSize, thr=1e-3)
print('finished ' + "C_PRSW_max")
ZenkerPatient.fixedPoint_wrt_parameterChange("heartParamsDict", "C_PRSW_min", parameterUnits = 'mm Hg', minVal=10, maxVal=80, nValues=nValues, batchSize=batchSize, thr=1e-3)
print('finished ' + "C_PRSW_min")
ZenkerPatient.fixedPoint_wrt_parameterChange("heartParamsDict", "R_valve", parameterUnits = 'mm Hg s/ml', minVal=0.00025, maxVal=0.025, nValues=nValues, batchSize=batchSize, thr=1e-3)
print('finished ' + "R_valve")
ZenkerPatient.fixedPoint_wrt_parameterChange("systemicParamsDict", "V_a0", parameterUnits = 'ml', minVal=300, maxVal=2000, nValues=nValues, batchSize=batchSize, thr=1e-3)
print('finished ' + "V_a0")
ZenkerPatient.fixedPoint_wrt_parameterChange("systemicParamsDict", "R_TPR_max", parameterUnits = 'mm Hg s/ml', minVal=1, maxVal=5, nValues=nValues, batchSize=batchSize, thr=1e-3)
print('finished ' + "R_TPR_max")
ZenkerPatient.fixedPoint_wrt_parameterChange("systemicParamsDict", "R_TPR_min", parameterUnits = 'mm Hg s/ml', minVal=0.05335, maxVal=2, nValues=nValues, batchSize=batchSize, thr=1e-3)
print('finished ' + "R_TPR_min")
ZenkerPatient.fixedPoint_wrt_parameterChange("systemicParamsDict", "C_a", parameterUnits = 'ml / mm Hg', minVal=2, maxVal=6, nValues=nValues, batchSize=batchSize, thr=1e-3)
print('finished ' + "C_a")
ZenkerPatient.fixedPoint_wrt_parameterChange("systemicParamsDict", "V_t", parameterUnits = 'ml', minVal=2000, maxVal=6000, nValues=nValues, batchSize=batchSize, thr=1e-3)
print('finished ' + "V_t")
ZenkerPatient.fixedPoint_wrt_parameterChange("controlParamsDict", "P_a_set", parameterUnits = 'mm Hg', minVal=50, maxVal=100, nValues=nValues, batchSize=batchSize, thr=1e-3)
print('finished ' + "P_a_set")
ZenkerPatient.fixedPoint_wrt_parameterChange("controlParamsDict", "K_width", parameterUnits = '1 / mm Hg', minVal=0.01838, maxVal=1.838, nValues=nValues, batchSize=batchSize, thr=1e-3)
print('finished ' + "K_width")
ZenkerPatient.fixedPoint_wrt_parameterChange("controlParamsDict", "tau_Baro", parameterUnits = 'sec', minVal=10, maxVal=30, nValues=nValues, batchSize=batchSize, thr=1e-3)
print('finished ' + "tau_Baro")

plt.show()

ZenkerPatient.enableBaroreflexControl = False
stateVec = ZenkerPatient.runModelOnBatchOfInitValues(batchSize=50, simDuration=simDuration, enableExternalInput=False)
ZenkerPatient.printModelTrajectories(stateVec, figTitle='Control is off', fileName='Control is off')

ZenkerPatient.enableBaroreflexControl = True
stateVec = ZenkerPatient.runModelOnBatchOfInitValues(batchSize=50, simDuration=simDuration, enableExternalInput=False)
ZenkerPatient.printModelTrajectories(stateVec, figTitle='Control is on', fileName='Control is on')

plt.show()
