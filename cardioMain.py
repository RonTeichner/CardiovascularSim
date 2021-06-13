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

simDuration = 100  # [sec]
sol = ZenkerPatient.runModel(simDuration=simDuration, initList=initListNoInput, enableExternalInput=False)

t = np.linspace(0, simDuration, 100*simDuration)
z = sol.sol(t)

nColumns = 3
plt.subplot(nColumns, 1, 1)
plt.plot(t, z.T[:, [0, 1]])
plt.xlabel('sec')
plt.legend(['Ves', 'Ved'], shadow=True)
#plt.title('Cardiovascular System')
plt.grid()

plt.subplot(nColumns, 1, 2)
plt.plot(t, z.T[:, [2]])
Vv = totalBloodVol - (z.T[:, 0] + z.T[:, 1] + z.T[:, 2])
plt.plot(t, Vv)
plt.xlabel('sec')
plt.legend(['Va', 'Vv'], shadow=True)
#plt.title('Cardiovascular System')
plt.grid()

plt.subplot(nColumns, 1, 3)
plt.plot(t, z.T[:, [3]])
plt.xlabel('sec')
plt.legend(['S'], shadow=True)
#plt.title('Cardiovascular System')
plt.grid()

plt.show()