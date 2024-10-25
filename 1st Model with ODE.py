import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the system of equations
def thermal_model(y, t, params):
    # Unpack variables
    Tint, Ts = y

    # Unpack parameters
    Cres, Cs, Ri, Ro, Rf, Text, Qres, Qs = params
    
    # First ODE for the internal temperature
    dTint_dt = (1/Cres) * (
        (1/Ri + 1/Ro) * Tint + (1/Rf) * Text + Qres
    )
    
    # Second ODE for the structural temperature
    dTs_dt = (1/Cs) * (
        (1/Ri) * Tint + (1/Ro) * Text + Qs
    )
    
    return [dTint_dt, dTs_dt]

# Example parameters for Zone i (adjusted to more typical values)
Cres = 1e6   # J/K (indoor capacity, typical for a building)
Cs = 5e5     # J/K (wall capacity, typical value)
Ri = 5       # K/W (internal resistance, typical)
Ro = 10      # K/W (external resistance, typical)
Rf = 3       # K/W (window resistance, typical)
Text = 10    # °C (external temperature, cooler environment)
Qres = 5000  # W (heating power, more typical)
Qs = 2000    # W (solar power, reduced)

# Pack parameters into a tuple
params = (Cres, Cs, Ri, Ro, Rf, Text, Qres, Qs)

# Initial conditions: [Tint, Ts]
y0 = [20, 18]

# Time points where the solution is computed
t = np.linspace(0, 1000, 100)

# Solve the system of equations
sol = odeint(thermal_model, y0, t, args=(params,))

# Plot the results
plt.plot(t, sol[:, 0], label='Internal Temp (Tint)')
plt.plot(t, sol[:, 1], label='Structural Temp (Ts)')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()