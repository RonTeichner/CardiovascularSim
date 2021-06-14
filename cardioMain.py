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
V_v_init = totalBloodVol - (V_ed_init + V_es_init + V_a_init)
S_init = 0
initListComplete = [V_es_init, V_ed_init, V_a_init, V_v_init, S_init]
initListNoInput = [V_es_init, V_ed_init, V_a_init, S_init]
initArrayNoInput = np.array([[V_es_init], [V_ed_init], [V_a_init], [S_init]])

fixedPoints = ZenkerPatient.computeFixedPoints(initialGuess=initListNoInput)
print(f'Zenker model fixed point found at [Ves, Ved, Va, S] = {fixedPoints.x.round()}')


ZenkerPatient.enableBaroreflexControl = False

simDuration = 100  # [sec]
stateVec = ZenkerPatient.runModelOnBatchOfInitValues(batchSize=50, simDuration=simDuration, enableExternalInput=False)
ZenkerPatient.printModelTrajectories(stateVec)
