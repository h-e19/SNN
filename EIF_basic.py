import numpy as np
import matplotlib.pyplot as plt

# tau_m = membrane time constant
# E_L = resting potential
# D = threshold sharpness
# V_T = soft threshold
# V_th = spike cutoff
# V_re = rest potenial

tau_m = 15.0
E_L = -72.0
D = 2.0
V_T = -55.0
V_th = 5.0
V_re = -75.0

dt = 0.1
T = 100000.0

def dVdt(V, I):
    return (-(V - E_L) + D * np.exp((V - V_T) / D) + I) / tau_m

time = np.arange(0, T, dt)
V = np.zeros(len(time))


I_0 = 10.0
sigma = 8

I_e = 0.0
I_i = 0.0

tau_e = 5.0  
tau_i = 5.0       # ms
K_e = 200
K_i = 50
r_e = 8/1000
r_i = 15/1000     # spikes per ms
j_e = 25.0
j_i = -40.0       # mV

time_array = np.arange(0, T, dt)
r = 20 / 1000     # 20 Hz, converted to spikes per ms
p = r * dt

# spike_times = []

# for bin in range(len(time_array)):
#     if np.random.rand() < p:
#         spike_times.append(time_array[bin])

def sim(I_e, I_i):
    spike_times = []
    V[0] = E_L

    for i in range(len(time)-1):   
        n_e = np.random.poisson(K_e * r_e * dt)
        n_i = np.random.poisson(K_i * r_i * dt)
        I_e = I_e + dt * (-I_e / tau_e) + (n_e * j_e)/tau_e
        I_i = I_i + dt * (-I_i / tau_i) + (n_i * j_i)/tau_i
        I = I_e + I_i
        V[i+1] = V[i] + dt * dVdt(V[i], I)
        if V[i+1] >= V_th:
            V[i+1] = V_re
            spike_times.append(time[i+1])
    return spike_times

isi = np.diff(sim(I_e, I_i))

print(f"mean ISI: {np.mean(isi):.2f} ms")
print(f"CV: {np.std(isi)/np.mean(isi):.3f}")

plt.hist(isi, bins=60, density=True)
plt.xlabel('ISI (ms)')
plt.ylabel('probability density')
plt.show()

# plt.plot(time[:2000], V[:2000])
# plt.xlabel('time (ms)')
# plt.ylabel('V (mV)')
# plt.title('EIF neuron, I_0 = 20')
# plt.show()

# currents = np.arange(0, 50, 0.5)
# rates = []

# for I in currents:
#     rates.append(len(sim(I, sigma))/(T/1000.0))  

# plt.plot(currents, rates)
# plt.xlabel('Input current (mV)')
# plt.ylabel('Firing rate (Hz)')
# plt.title('F-I curve of EIF neuron')
# plt.show()

