#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:25:00 2024

Power curve computed following the three-regimes strategy.

In this implementation, a constant lift-to-drag ratio E is prescribed during
reel-in, which means that the elevation angle during reel-in is computed as a
dependent variable.

Original code from: Roland Schmehl adapted by Joren Bredael and Rafael Dux

References:

Van der Vlugt, R., Peschel, J., Schmehl, R. (2013). Design and Experimental
Characterization of a Pumping Kite Power System. In: Ahrens, U., Diehl, M.,
Schmehl, R. (eds) Airborne Wind Energy. Green Energy and Technology. Springer,
Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-39965-7_23

@authors: Roland Schmehl, Joren Bredael, Rafael Dux
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize as op
from scipy.optimize import fsolve
from pylab import np
from scipy.integrate import solve_ivp
import bisect
from wind_curve import produce_weibull, convert_velocity, histogram
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#mpl.rcParams['font.family'] = "Open Sans"
mpl.rcParams.update({'font.size': 18})
mpl.rcParams['figure.figsize'] = 10, 5.625
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams['pdf.fonttype'] = 42 # Output Type 3 (Type3) or Type 42 (TrueType)

# Plot settings

plot_powercurve = True
plot_operations = True
plot_histogram = True
plot_trajectory = True
plot_cycle_times = True
v_trajectory = 12.5         # m/s


# Environmental properties
atmosphere_density        =  1.225   # kg/**3
wind_speed_min            =  1.      # m/s
wind_speed_max            =  20.     # m/s
wind_speed_rated          = 12.5     # m/s
wind_speed_delta          =  0.01     # m/s
wind_speed_cutin          = 4.4     # m/s @ 20 m


# Kite properties
kite_planform_area        =  400 # m²
kite_planform_area_non_projected = 470 # m²
kite_lift_coefficient_out =  1.423      # -
kite_drag_coefficient_out =  0.113 + 0.012   # -
kite_lift_coefficient_in  =  0.14    # -
kite_drag_coefficient_in  =  0.07    # -

# Tether properties
tether_drag_coefficient   =  1.1     # -
tether_diameter           =  0.056   # m
tether_length_max         =  1000.    # m
tether_length_min         =  400.    # m
tether_safety_factor      =  4       # -
nominal_tether_force      =  250 * 9.81 * 1000 / tether_safety_factor # N (break load taken from Dyneema site)

# Generator properties
nominal_generator_power   =  3650000.  # W

# Operational parameters
beta_out                  =  30.     # deg
reeling_speed_min_limit   =  -18     # m/s
reeling_speed_max_limit   =   8.     # m/s

# Power parameters
rated_power_limit         = 2300000 # W
pto_efficiency            = 0.9 # efficiency of the power take-off system


# Derived properties
rm_out = 0.5 * (tether_length_min + tether_length_max)
CD_out = (kite_drag_coefficient_out + 0.25 * tether_drag_coefficient \
         * tether_diameter * rm_out / kite_planform_area) * 1.1
E2out  = (kite_lift_coefficient_out / CD_out)**2
E2in   = (kite_lift_coefficient_in  / kite_drag_coefficient_in)**2
cosine_beta_out  = np.cos(np.radians(beta_out))
force_factor_out = kite_lift_coefficient_out * np.sqrt(1+1/E2out) * (1+E2out)
force_factor_in  = kite_lift_coefficient_in  * np.sqrt(1+1/E2in)
power_factor_ideal = force_factor_out * cosine_beta_out**3 * 4/27
operation_height = np.sin(np.radians(beta_out)) * (rm_out)      # m
wind_speed_range = wind_speed_max - wind_speed_min
num_wind_speeds  = int(wind_speed_range/wind_speed_delta + 1)
wind_speed  = np.linspace(wind_speed_min, wind_speed_max, num_wind_speeds)
KCU_mass = 7/25 * kite_planform_area # kg
KCU_power = 200/20000 * rated_power_limit # scaling done to match the power of the KCU to the rated power
wind_speed_cutin_operation = convert_velocity(wind_speed_cutin,operation_height) # m/s
packedvolume = kite_planform_area_non_projected * 3/225 # m³


# Arrays
reeling_factor_out = np.array([])
reeling_factor_in = np.array([])
reeling_factor_in_capped = np.array([])
tether_force_out = np.array([])
tether_force_in = np.array([])
tether_force_in_capped = np.array([])
power_out = np.array([])
power_in = np.array([])
power_in_capped = np.array([])
cycle_power = np.array([])
cycle_power_capped = np.array([])
power_ideal = np.array([])
elevation_angle_in = np.array([])
elevation_angle_in_capped = np.array([])
elevation_angle_out = np.array([])
reel_out_time_fraction = np.array([])
t_out = np.array([])
t_in = np.array([])
t_in_capped = np.array([])
cycle_time = np.array([])


# Objective function for the three wind speed domains
def objective_function_1(x):
    f_out    = x[0]
    f_in     = x[1]
    return -((cosine_beta_out - f_out)**2 \
             - (force_factor_in / force_factor_out) \
               * (np.sqrt(1 + E2in*(1 - f_in**2)) - f_in)**2/(1 + E2in)) \
            * (f_in*f_out) / (f_in - f_out)

def objective_function_2(x, mu_F, f_nF):
    f_in     = x[0]
    b        = (mu_F - 1) * cosine_beta_out + f_nF
    return -(((cosine_beta_out - f_nF) / mu_F)**2  \
             - (force_factor_in / force_factor_out) \
               * (np.sqrt(1 + E2in*(1 - f_in**2)) - f_in)**2/(1 + E2in))  \
            * f_in*b/(mu_F*f_in-b)

def objective_function_3(x, mu_P, f_nP):
    f_in     = x[0]
    return -((cosine_beta_out - f_nP/mu_P)**2  \
             - (force_factor_in / force_factor_out) \
               * (np.sqrt(1 + E2in*(1 - f_in**2)) - f_in)**2/(1 + E2in))  \
            * f_in*f_nP/(mu_P*f_in-f_nP)

def cap_cycle_power(f_in):
    """
    This function is used to cap the cycle power, by root finding.
 
    Args:
        f_in (float): Reeling factor during reel-in phase
 
    Returns:
        float: A value used during root finding
    """
    t_i = (tether_length_max - tether_length_min) / (v_w * -f_in)
    t_o = (tether_length_max - tether_length_min) / (v_w * f_out)
    Ft_out = q * kite_planform_area * force_factor_out  \
               * (cosine_beta_out - f_out)**2
    Ft_in  = q * kite_planform_area * force_factor_in  \
               * (np.sqrt(1 + E2in*(1 - f_in**2)) - f_in)**2/(1 + E2in)
    P_in   = Ft_in  * v_w * f_in * pto_efficiency
    P_out  = Ft_out * v_w * f_out * pto_efficiency
    return t_i*(P_in - rated_power_limit)- t_o * (rated_power_limit-P_out)


###############################################################################

# Initialize wind speed regimes
wind_speed_regime      = 1
wind_speed_force_limit = 0
wind_speed_power_limit = 0
print("Wind speed regime 1")

# Loop over wind speed range
for v_w in wind_speed:

    # Dynamic pressure
    q  = 0.5 * atmosphere_density * v_w**2

    # Wind power density
    P_w = q*v_w

    # Reeling factor limits
    f_max = reeling_speed_max_limit / v_w
    f_min = reeling_speed_min_limit / v_w
    # Unconstrained operation
    if wind_speed_regime == 1:

        starting_point = (0.001, -0.001)
        bounds         = ((0.001,  f_max), (f_min, -0.001),)

        optimisation_result = op.minimize(objective_function_1, \
                                          starting_point,       \
                                          bounds=bounds,        \
                                          method='SLSQP')

        # Reeling factors
        global f_out
        f_out = optimisation_result['x'][0]
        f_in  = optimisation_result['x'][1]
        # Normalized cycle power
        p_c = -objective_function_1 ([f_out, f_in]) * pto_efficiency

        # Tether force during reel-out
        Ft_out = q * kite_planform_area * force_factor_out  \
                   * (cosine_beta_out - f_out)**2

        if Ft_out > nominal_tether_force:
            wind_speed_regime = 2

            # Determine precise value of v_w,F by interval bisection
            v_b  = v_w
            v_a  = v_w - wind_speed_delta
            c    = 0.5 * atmosphere_density * kite_planform_area \
                       * force_factor_out * (cosine_beta_out - f_out)**2
            nmax = 100
            eps  = 0.1
            for i in range(nmax):
                v  = (v_a + v_b)/2
                Ft = c * v**2
                if Ft > nominal_tether_force:
                    v_b = v
                else:
                    v_a = v
                if abs(Ft-nominal_tether_force) < eps:
                    break
            else:
                print("!!! search v_w,F stopped after nmax=", nmax, "iterations")
                print("--> increase nmax and rerun")

            wind_speed_force_limit = v
            f_nF  = f_out # works because f_out is constant in regime 1

            print()
            print("Wind speed regime 2 with v_n,F at", "{:5.2f}".format(wind_speed_force_limit))
            print()

    # Constrained tether force
    if wind_speed_regime == 2:

        mu_F = v_w / wind_speed_force_limit

        starting_point = (-0.001)
        bounds         = ((f_min, -0.001),)

        optimisation_result = op.minimize(objective_function_2, \
                                          starting_point,       \
                                          args=(mu_F, f_nF),    \
                                          bounds=bounds,        \
                                          method='SLSQP')

        # Reeling factors
        f_out = (cosine_beta_out * (mu_F - 1) + f_nF)/mu_F
        f_in  = optimisation_result['x'][0]

        # Normalized cycle power
        p_c = -objective_function_2 ([f_in], mu_F, f_nF) * pto_efficiency

        # Tether force and mechanical power during reel out
        Ft_out = q * kite_planform_area * force_factor_out * \
                     (cosine_beta_out - f_out)**2

        # Mechanical power during reel out
        P_out  = Ft_out * v_w * f_out

        if P_out > nominal_generator_power:
            wind_speed_regime = 3

            # Determine precise value of v_w,P by interval bisection
            v_b  = v_w
            v_a  = v_w - wind_speed_delta
            c    = 0.5 * atmosphere_density * kite_planform_area \
                       * force_factor_out
            nmax = 100
            eps  = 1
            for i in range(nmax):
                v  = (v_a + v_b)/2
                mu = v / wind_speed_force_limit
                f  = (cosine_beta_out * (mu - 1) + f_nF)/mu
                P  = c * (cosine_beta_out - f)**2 * v**3 * f
                if P > nominal_generator_power:
                    v_b = v
                else:
                    v_a = v
                if abs(P-nominal_generator_power) < eps:
                    break
            else:
                print("!!! search v_w,P stopped after nmax=", nmax, "iterations")
                print("--> increase nmax and rerun")

            wind_speed_power_limit = v
            f_nP = f

            print()
            print("Wind speed regime 3 with v_n,P at", "{:5.2f}".format(wind_speed_power_limit))
            print()

    # Constrained tether force and generator power
    if wind_speed_regime == 3:

        mu_P  = v_w / wind_speed_power_limit
        f_out = f_nP / mu_P

        # Reduce force factor to comply with tether force limit
        # force_factor_out = nominal_tether_force / (q * kite_planform_area \
        #                    * (cosine_beta_out - f_out)**2)

        # Alternative strategy to depower: increasing the elevation angle
        cosine_beta_out = np.sqrt(nominal_tether_force / (q \
                          * kite_planform_area * force_factor_out)) + f_out

        beta_out = np.degrees(np.arccos(cosine_beta_out))
        starting_point = (-0.001)
        bounds         = ((f_min, -0.001),)

        optimisation_result = op.minimize(objective_function_3, \
                                          starting_point,       \
                                          args=(mu_P, f_nP),    \
                                          bounds=bounds,        \
                                          method='SLSQP')

        # Reeling factors
        f_in  = optimisation_result['x'][0]
        
        # Normalized cycle power
        p_c = -objective_function_3 ([f_in], mu_P, f_nP) * pto_efficiency

    
    f_in_capped = 0
    if p_c * force_factor_out * kite_planform_area * P_w > rated_power_limit:  
        f_in_capped = fsolve(cap_cycle_power, f_in)[0]
    else:
        f_in_capped = f_in

    # Cycle characteristics
    if v_w == wind_speed_rated:
        reel_out_time_fraction_rated = (f_in/(f_in-f_out))
        t_out_rated = (tether_length_max- tether_length_min)/(v_w * f_out)
        t_in_rated = (tether_length_max- tether_length_min)/(v_w * -f_in)
        cycle_time_rated = t_out_rated + t_in_rated
        reel_out_factor_rated = f_out 

    # Tether force
    Ft_out = q * kite_planform_area * force_factor_out  \
               * (cosine_beta_out - f_out)**2
    Ft_in  = q * kite_planform_area * force_factor_in  \
               * (np.sqrt(1 + E2in*(1 - f_in**2)) - f_in)**2/(1 + E2in)
    Ft_in_capped  = q * kite_planform_area * force_factor_in  \
               * (np.sqrt(1 + E2in*(1 - f_in_capped**2)) - f_in_capped)**2/(1 + E2in)

    # Mechanical power
    P_out  = Ft_out * v_w * f_out * pto_efficiency
    P_in   = Ft_in  * v_w * f_in / pto_efficiency
    P_in_capped   = Ft_in_capped  * v_w * f_in_capped / pto_efficiency

    # Elevation angle reel-in-phase
    beta_in = np.arccos((np.sqrt(1 + E2in*(1 - f_in_capped**2)) \
                                   + f_in_capped*E2in)/(1 + E2in))
    # Cycle times
    t_i = (tether_length_max - tether_length_min) / (v_w * -f_in)
    t_i_capped = (tether_length_max - tether_length_min) / (v_w * -f_in_capped)
    t_o = (tether_length_max - tether_length_min) / (v_w * f_out)

    # print("{:4.1f}".format(v_w),    \
    #       "{:5.3f}".format(f_out),  \
    #       "{:5.3f}".format(f_in),   \
    #       "{:5.0f}".format(Ft_out), \
    #       "{:5.0f}".format(Ft_in),  \
    #       "{:6.0f}".format(P_out),  \
    #       "{:6.0f}".format(P_in),   \
    #       "{:4.1f}".format(v_w * f_out), \
    #       "{:4.1f}".format(v_w * f_in), \
    #       "{:5.2f}".format(force_factor_out), \
    #       "{:5.2f}".format(force_factor_in), \
    #       "{:4.1f}".format(np.degrees(beta_in)))
    reeling_factor_out = np.append(reeling_factor_out,f_out)
    reeling_factor_in = np.append(reeling_factor_in, f_in)
    reeling_factor_in_capped = np.append(reeling_factor_in_capped, f_in_capped)
    tether_force_out = np.append(tether_force_out, Ft_out)
    tether_force_in = np.append(tether_force_in, Ft_in)
    tether_force_in_capped = np.append(tether_force_in_capped, Ft_in_capped)
    power_out = np.append(power_out, P_out)
    power_in = np.append(power_in, P_in)
    power_in_capped = np.append(power_in_capped, P_in_capped)
    cycle_power = np.append(cycle_power, ((P_out*t_o + P_in*t_i)/ (t_o + t_i)))
    if p_c * force_factor_out * kite_planform_area * P_w > rated_power_limit:       
        cycle_power_capped = np.append(cycle_power_capped,((P_out*t_o + P_in_capped*t_i_capped)/ (t_o + t_i_capped)))
    else:
        cycle_power_capped = np.append(cycle_power_capped, p_c * force_factor_out * kite_planform_area * P_w)
    power_ideal = np.append(power_ideal, power_factor_ideal * kite_planform_area * P_w)
    elevation_angle_in = np.append(elevation_angle_in, np.degrees(beta_in))
    elevation_angle_out = np.append(elevation_angle_out, beta_out)
    reel_out_time_fraction = np.append(reel_out_time_fraction, f_in_capped/(f_in_capped-f_out))
    t_out = np.append(t_out, t_o)
    t_in = np.append(t_in, t_i)
    t_in_capped = np.append(t_in_capped, t_i_capped)
    cycle_time = np.append(cycle_time, t_i_capped + t_o)


# Capacity factor calculation
pdf = produce_weibull(wind_speed_min, wind_speed_max, operation_height, wind_speed_delta)[1] # Windspeed probability density @ operation height 
power_pdf = pdf * cycle_power_capped
index_cutin = bisect.bisect_left(wind_speed, wind_speed_cutin_operation)
avg_power = np.sum(power_pdf[index_cutin:])
capacity_factor = avg_power/max(cycle_power_capped)
flight_hours = np.sum(pdf[index_cutin:]) * 365*24
# Other characteristics
avg_cycle_time = np.sum(cycle_time[index_cutin:] * pdf[index_cutin:])
avg_tether_force = np.sum((tether_force_out[index_cutin:] *reel_out_time_fraction[index_cutin:]+tether_force_in_capped[index_cutin:] *(1-reel_out_time_fraction[index_cutin:] ))* pdf[index_cutin:])
average_tether_loading_factor = avg_tether_force/nominal_tether_force


print("----------POWER----------")
print("> Maximal cycle power: ", "{:1.4f}".format(max(cycle_power)/1000000), "MW at ", \
      "{:4.1f}".format(wind_speed[cycle_power.argmax()]), "m/s")
print("> Rated power: ", str(round(float(cycle_power[np.where(wind_speed == wind_speed_rated)])/1000000,2)), "MW")
print("> Nominal generator power: ", str(round(nominal_generator_power/1000000,2)), "MW")
print("> Maximal reel-out power: ", str(round(max(power_out)/1000000,2)), "MW")
print("> Maximal reel-in power: ", str(round(min(power_in)/1000000,2)), "MW")
print("> Reel-in power at rated: ", str(round(float(power_in_capped[np.where(wind_speed == wind_speed_rated)])/1000000,2)), "MW")
print("> Average power: ", str(round(avg_power/1000000,4)), "MW ")
print("----------Force----------")
print("> Nominal tether force: ", str(round(nominal_tether_force,0)), "N")
print("> Average tether force: ", str(round(avg_tether_force,0)), "N")
print("> Average tether loading factor: ", str(round(average_tether_loading_factor,2)), "-")
print("> Maximal reel-in tether force: ", str(round(max(tether_force_in_capped),0)), "N")
print("----------CAPACITY FACTOR CHARACTERISTICS----------")
print("> Capacity factor: " + str(round(capacity_factor,4)))
print("> Yearly flight hours: " + str(round(flight_hours)))
print("> Cut-in windspeed @20m: ", str(round(wind_speed_cutin,2)), "m/s")
print("> Cut-in windspeed @"+ str(round(operation_height))+"m: "+ str(round(wind_speed_cutin_operation,2)), "m/s")
print("----------CYCLE CHARACTERISTICS----------")
print("> Reel-out time percentage @"+ str(round(wind_speed_rated,1)) + " m/s: "+ str(round((reel_out_time_fraction_rated*100),2)) + "%")
print("> Maximum reel-out time percentage "+ str(round((max(reel_out_time_fraction)*100),2)) + "%")
print("> Reel-out time  @"+ str(round(wind_speed_rated,1)) + " m/s: "+ str(round(t_out_rated,2)) + "s")
print("> Reel-in time  @"+ str(round(wind_speed_rated,1)) + " m/s: "+ str(round(t_in_rated,2)) + "s")
print("> Cycle time  @"+ str(round(wind_speed_rated,1)) + " m/s: "+ str(round(cycle_time_rated,2)) + "s")
print("> Average cycle time  @"+ str(round(wind_speed_rated,1)) + " m/s: "+ str(round(avg_cycle_time,2)) + "s")
print("> Maximum reel-out speed "+ str(round((max(reeling_factor_out*wind_speed)),2)) + " m/s")
print("> Maximum reel-in speed "+ str(round((min(reeling_factor_in_capped*wind_speed)),2)) + " m/s")
print("> Reel-out fraction  @"+ str(round(wind_speed_rated,1)) + " m/s: "+ str(round(reel_out_factor_rated,2)))
print("----------Operational Charachteristics----------")
print("> Average operational height: ", str(round(operation_height)), "m")
print("> Highest reel-out angle: ", str(round(max(elevation_angle_out))), "°")
# print("> Highest reel-in angle: ", str(round(max(elevation_angle_in))), "°")
# print("> Lowest reel-in angle: ", str(round(min(elevation_angle_in))), "°")
print("----------KCU----------")
print("> KCU mass: "+ str(round(KCU_mass,2)) + " kg")
print("> KCU power: "+ str(round(KCU_power/1000,2)) + " kW")
print("----------Kite----------")
print("> Kite L/D reel-out: "+ str(round(np.sqrt(E2out),2)))
print("> Kite packed volume: "+ str(round(packedvolume,2)) + " m³")
print("\n")



if plot_powercurve:
    fig, ax1 = plt.subplots()
    ax1.set(xlabel=r"Wind speed, m/s", ylabel=r"Mechanical power, MW")
    ax1.set_xlim([0, wind_speed_max])
    ax1.set_ylim([0, 3.6])
    #ax1.grid()
    ax1.vlines(wind_speed_force_limit, 0, 100, colors='k', linestyles=':')
    ax1.vlines(wind_speed_power_limit, 0, 100, colors='k', linestyles=':')
    #ax1.vlines(wind_speed_cutin, 0, 100, colors='r', linestyles=':')
    ax1.vlines(wind_speed_cutin_operation, 0, 100, colors='r', linestyles=':', label=r"$v_{cut-in}$")
    ax1.annotate("1",(6.8,0.3), ha="center", va="center", bbox={"boxstyle" : "circle", "color":"white", "ec" : "k"})
    ax1.annotate("2",(10,1), ha="center", va="center", bbox={"boxstyle" : "circle", "color":"white", "ec" : "k"})
    ax1.annotate("3",(15.5,1), ha="center", va="center", bbox={"boxstyle" : "circle", "color":"white", "ec" : "k"})
    ax1.annotate(r"$v_{\mathrm{n,F}}$",(24.5,-2.5), annotation_clip=False, ha="center", va="center")
    ax1.annotate(r"$v_{\mathrm{n,P}}$",(35.,-2.5), annotation_clip=False, ha="center", va="center")
    ax2 = ax1.twinx()
    ax2.plot(wind_speed, pdf*1000, 'y', linestyle=':', label=r'pdf')
    ax2.set(ylabel=r"Windspeed pdf, $10^{-3}$")
    ax2.set_xlim([0, wind_speed_max])
    ax2.set_ylim([0, (np.max(pdf*1000)+0.2*np.max(pdf*1000))])
    #ax1.plot(wind_speed,  np.asarray(power_ideal)/1000000, 'k', linestyle=':', label=r"$P_{\mathrm{opt}}$")
    ax1.plot(wind_speed,  np.asarray(cycle_power)/1000000, 'b', linestyle='-', label=r"$P_{\mathrm{c, non-capped}}$")
    ax1.plot(wind_speed,  np.asarray(cycle_power_capped)/1000000, 'm', linestyle='-', label=r"$P_{\mathrm{c}}$")
    ax1.plot(wind_speed,  np.asarray(power_out)/1000000, 'g', linestyle='--', label=r"$P_{\mathrm{o}}$")
    ax1.plot(wind_speed, -np.asarray(power_in_capped)/1000000, 'r', linestyle='--', label=r"$-P_{\mathrm{i}}$")
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.9),facecolor="none", edgecolor="none", fontsize="small")
    fig.savefig("scripts/AWES/figures/powercurve_const_LoD_in.png")

if plot_operations:
    fig, ax1 = plt.subplots()
    ax1.set(xlabel=r"Wind speed, m/s", ylabel=r"Reeling factor, -")
    ax1.set_xlim([0, wind_speed_max])
    ax1.set_ylim([0, 1.5])
    #ax1.grid()
    ax1.vlines(wind_speed_force_limit, 0, 100, colors='k', linestyles=':')
    ax1.vlines(wind_speed_power_limit, 0, 100, colors='k', linestyles=':')
    ax1.annotate(r"$v_{\mathrm{n,F}}$",(24.5,-0.045), annotation_clip=False, ha="center", va="center")
    ax1.annotate(r"$v_{\mathrm{n,P}}$",(35,-0.045), annotation_clip=False, ha="center", va="center")
    ax1.plot(wind_speed,  np.asarray(reeling_factor_out), 'g', linestyle='--', label=r"$f_{\mathrm{o}}$")
    ax1.plot(wind_speed, -np.asarray(reeling_factor_in), 'r', linestyle='--', label=r"$-f_{\mathrm{i, non-capped}}$")
    ax1.plot(wind_speed, -np.asarray(reeling_factor_in_capped), 'm', linestyle='--', label=r"$-f_{\mathrm{i}}$")
    ax2 = ax1.twinx()
    ax2.set(ylabel=r"Elevation angle, deg")
    ax2.set_ylim([0, 230])
    ax2.plot(wind_speed,  np.asarray(elevation_angle_in), 'r', linestyle='-', label=r"$\beta_{\mathrm{i}}$")
    ax2.plot(wind_speed,  np.asarray(elevation_angle_out), 'g', linestyle='-', label=r"$\beta_{\mathrm{o}}$")
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.9),facecolor="none", edgecolor="none", fontsize="small")
    fig.savefig("scripts/AWES/figures/operations_const_LoD_in.png")

if plot_histogram:
    histogram(operation_height, wind_speed_max)

def l(t): # tether length
    return tether_length_max + v_in*t 

def f(t, b): # db/dt
    return v_trajectory/l(t) * ((kite_lift_coefficient_in/kite_drag_coefficient_in) * (np.cos(b) - f_in) - np.sin(b))

if plot_trajectory:

    f_in = float(reeling_factor_in[np.where(wind_speed == v_trajectory)])               # reel-in factor [-]

    v_in = v_trajectory*f_in
    t_in_trajectory = (tether_length_min - tether_length_max)/(v_in)
    b_ideal = np.arccos((np.sqrt(1 + E2in*(1 - f_in**2)) + f_in*E2in)/(1 + E2in))

    tether_range = np.linspace(tether_length_min, tether_length_max, int(tether_length_max-tether_length_min+1))
    
    # Solve ODE for elevation_angle_in
    trajectory = solve_ivp(f, (0, t_in_trajectory), [np.deg2rad(beta_out)], t_eval=np.linspace(0, t_in_trajectory, 100))

    # reel-out trajectory
    fig, ax1 = plt.subplots()
    ax1.plot(tether_range*np.cos(np.deg2rad(beta_out)), tether_range*np.sin(np.deg2rad(beta_out)), 'g', linestyle='-', label="Reel-out")
    # reel-in trajectory
    ax1.plot(l(trajectory.t) * np.cos(trajectory.y)[0], l(trajectory.t) * np.sin(trajectory.y)[0], 'r', linestyle='-',label="Reel-in")
    # minimum tether length
    ax1.plot(np.linspace(-tether_length_min, tether_length_min, int(2*tether_length_min+1)), np.sqrt(tether_length_min**2 - np.linspace(-tether_length_min, tether_length_min, int(2*tether_length_min+1))**2), 'k:', label="Minimum tether length")
    # Ideal reel-in trajectory
    ax1.plot(tether_range*np.cos(b_ideal), tether_range*np.sin(b_ideal), 'r', linestyle='--', label="Asymptotic reel-in")
    ax1.set(xlabel=r"Horizontal distance, m")
    ax1.set(ylabel=r"Height, m")
    ax1.set_xlim(-tether_length_max, tether_length_max)
    ax1.set_ylim(0, 3/4 * (2*tether_length_max+1))
    ax1.set_aspect("equal")
    fig.legend(loc='upper left', bbox_to_anchor=(0.25, 0.9),facecolor="none", edgecolor="none", fontsize="small")
    fig.savefig("scripts/AWES/figures/trajectory.png")

if plot_cycle_times:
    fig, ax1 = plt.subplots()
    ax1.set(xlabel=r"Wind speed, m/s", ylabel=r"Time, s")
    ax1.set_xlim([0, wind_speed_max])
    ax1.set_ylim([0, 400])
    #ax1.grid()
    ax1.vlines(wind_speed_force_limit, 0, 1000, colors='k', linestyles=':')
    ax1.vlines(wind_speed_power_limit, 0, 1000, colors='k', linestyles=':')
    ax1.annotate(r"$v_{\mathrm{n,F}}$",(24.5,-0.045), annotation_clip=False, ha="center", va="center")
    ax1.annotate(r"$v_{\mathrm{n,P}}$",(35,-0.045), annotation_clip=False, ha="center", va="center")
    ax1.plot(wind_speed,  np.asarray(t_out), 'g', linestyle='--', label=r"Reel-out time")
    ax1.plot(wind_speed, np.asarray(t_in), 'r', linestyle='--', label=r"Reel-in time non-capped")
    ax1.plot(wind_speed, np.asarray(t_in_capped), 'm', linestyle='--', label=r"Reel-in time")
    ax1.plot(wind_speed, np.asarray(cycle_time), 'b', linestyle='--', label=r"Cycle time")
    ax2 = ax1.twinx()
    ax2.set(ylabel=r"Percentage, %")
    ax2.set_ylim([0, 100])
    ax2.plot(wind_speed,  np.asarray(reel_out_time_fraction)*100, 'g', linestyle='-', label=r"Reel-out percentage")
    #ax2.plot(wind_speed,  (1-np.asarray(reel_out_time_fraction))*100, 'r', linestyle='-', label=r"Reel-in percentage")
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.5),facecolor="none", edgecolor="none", fontsize="small")
    fig.savefig("scripts/AWES/figures/cycle_times_const_LoD_in.png")

# # Investigation why tether force during reel in is so low
# tether_force_investigation = True
# if tether_force_investigation:
#     fig, ax1 = plt.subplots()
#     ax1.set(xlabel=r"reel in factor", ylabel=r"tether force in")
#     ax1.set_xlim([-2, 0])
#     ax1.set_ylim([0, 50000])
#     test_reel = np.linspace(-2, 0, num_wind_speeds)
#     q  = 0.5 * atmosphere_density * 12.5**2
#     test_force = q * kite_planform_area * force_factor_in  \
#                * (np.sqrt(1 + E2in*(1 - test_reel**2)) - test_reel)**2/(1 + E2in)
#     ax1.plot(test_reel,  test_force, 'g', linestyle='--')

#     fig.savefig("scripts/AWES/figures/tether_force_investigation.png")