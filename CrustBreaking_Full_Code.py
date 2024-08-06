#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.linalg import solve as scipy_solve
from sympy import symbols, Eq, solve, init_printing, latex, Rational, Function, cos, sin,  diff
import sympy as sp
import numpy as np
import pandas as pd
import math

#Varables that will be used symbollically in our eqautions
A, B, a, b, R, Rprime, R0,v_k, c_t, ang, theta = symbols(' A B a b RNS RNSprime R_o Vₖ Cₜ x theta')

#Constants
G = 6.67430e-11

TurnSolar = 0
while TurnSolar < 3:
    

    if TurnSolar == 0:
        SolarMass = 1.4
        M =  1.98849e30 * SolarMass
    elif TurnSolar == 1:
        SolarMass = 1.6
        M =  1.98849e30 * SolarMass
    elif TurnSolar == 2:
        SolarMass = 2.0
        M =  1.98849e30 * SolarMass

    turn = 0
    #Store the final results of the Freqcucny and Local Strain If the data is needed
    FArr = []
    StrainArr = []
    TotalArr = np.array([])
    ToalArr2 = np.array([])

    Radius = 10.0
    Ro = 10
    while Radius < 14:

        CrustPercent = 0.02
        while CrustPercent < 0.13:
            PrimeRadius = Radius - (Radius * CrustPercent)


            #Vk and OmegaK
            vk = math.sqrt(G*M/(Radius*1e3))
            omegak = vk/(Radius*1e3)


            #These are symbolically reprsetned
            a1 = a
            a2 = A * R0**2
            a3 = B/R0**3
            a4 = b/R0**5
            Unit_R = R/R0
            Unit_Rprime = Rprime/R0

            ang = 1
            #The equations for solving the constants A B a b
            Eq_5 = Eq(a1 - Rational(8,21)*a2*Unit_R**2 - a3/(2* Unit_R**3) + Rational(8,3) * a4/Unit_R**5,0)
            Eq_6 = Eq(a1 - Rational(8,21)*a2*Unit_Rprime**2 - a3/(2* Unit_Rprime**3) + Rational(8,3) * a4/Unit_Rprime**5,0)
            Eq_7 = Eq(-2* (a1 - (3*a2/7)* Unit_R**2 + a3/(Unit_R**3) - (4*a4/Unit_R**5))  - Rational(2,5)* (v_k**2/c_t**2) * (a1 - (a2/7) * Unit_R**2 - a3/(2*Unit_R**3) + a4/Unit_R**5)
                        +Rational(2,3) * ang * (v_k**2)/c_t**2, a2* Unit_R**2 + (a3/Unit_R**3)) 

            Eq_8 = Eq(a1 + (a2/14)* (Unit_Rprime)**2 + Rational(3/2) *( a3/(Unit_Rprime)**3) - (4*a4/(Unit_Rprime)**5) , 0)

            #Subing in numercial values in the equations to get numerical values
            Eq_5s = Eq_5.subs({R:Radius, R0:Ro})
            Eq_6s = Eq_6.subs({ Rprime:PrimeRadius, R0:Ro})
            Eq_7s = Eq_7.subs({R:Radius, R0:Ro, c_t:1e6, v_k:vk})
            Eq_8s = Eq_8.subs({ Rprime:PrimeRadius, R0:Ro})
            solution = sp.solve({Eq_7s,Eq_8s,Eq_5s,Eq_6s}, {a,A,B,b})

            #Now these are resprected with real numbercial numbers
            A1 = solution[a]
            A2 = solution[A] * Ro**2
            A3 = solution[B]/Ro**3
            A4 = solution[b]/Ro**5

            #Unitized the R values
            Unit_R = PrimeRadius/ Ro

            #Angle of the equator
            angle = 90
            Rads = angle * (math.pi/180)


            P2 = (3* cos(theta)**2 -1)/2
            diffP2 = diff(P2, theta)

            #Tensor Components
            SigmaRR = (a1 - (3*a2/7) * (Unit_R**2) + (a3)/(Unit_R**3) - 4*(a4)/(Unit_R**5)) * P2
            SigmaThetaTheta = (Rational(1,2)*a1 - Rational(5,42)*a2*(Unit_R)**2 - Rational(1,3)* (a4)/(Unit_R**5)) + \
                                (-a1 + Rational(1,3)*a2*(Unit_R**2)- Rational(1,2)*(a3)/(Unit_R**3) + Rational(7,3)* a4/(Unit_R**5)) * P2

            SigmaPP = -(Rational(1,2)* a1 - Rational(5,42)*a2*(Unit_R)**2 - Rational(1,3)* a4/(Unit_R**5)) + \
                        (Rational(2,21) * a2 * (Unit_R**2) - Rational(1,2)* a3/(Unit_R**3) + Rational(5,3)* a4/(Unit_R**5)) * P2  

            SigmaRTheta = Rational(1/2)*(a1 - Rational(8,21)*a2 *Unit_R**2 -Rational(1,2)*a3*(Unit_R)**3 + Rational(8,3)*a4* (Unit_R)**5)* diffP2
            SimgaThetaP = 0 
            SimgaPR = 0

            #subing in for varables to get a numbercial numbers in the tensor componets 
            SigmaRR_Sub = SigmaRR.subs({a1: A1, a2: A2, a3: A3, a4: A4, theta: Rads})
            SigmaThetaTheta_Sub = SigmaThetaTheta.subs({a1: A1, a2: A2, a3: A3, a4: A4, theta: Rads})
            SigmaPP_Sub = SigmaPP.subs({a1: A1, a2: A2, a3: A3, a4: A4, theta: Rads})
            SigmaRTheta_Sub = SigmaRTheta.subs({a1: A1, a2: A2, a3: A3, a4: A4, theta: Rads})


            angle = SigmaThetaTheta_Sub- SigmaPP_Sub
            angle2 = SigmaThetaTheta_Sub - SigmaRR_Sub


            #Making sure they stay a datatype floats 
            Sigma11 = float(SigmaRR_Sub)
            Sigma12 = float(SigmaRTheta_Sub)
            Sigma22 =  float(SigmaThetaTheta_Sub)
            Sigma33 =  float(SigmaPP_Sub)


            #Finding the EigenValues
            SIGMA = np.array([[Sigma11, Sigma12, 0],
                        [Sigma12, Sigma22, 0],
                        [0,     0,   Sigma33]
                        ])

            eigenvalues, eigenvectors = np.linalg.eig(SIGMA)


            #Gets the minimum and maxium values of the tensor components
            mins= np.min(eigenvalues)
            maxs = np.max(eigenvalues)

            #Subtract to get the local strain angle
            LocalStrainValue = maxs - mins
            # print('\n',LocalStrainValue)

            Omegaf = math.sqrt(2) * math.sqrt(0.1 / LocalStrainValue) * omegak
            max_initial_frequency = Omegaf / (2 * math.pi)

            # print(f'Maximum Initial Frequency: {max_initial_frequency} Hz\n')
            FArr.append(max_initial_frequency)
            StrainArr.append(LocalStrainValue)

            CrustPercent += 0.02
            TotalArr = np.vstack([FArr])
            TotalArr2 = np.vstack([StrainArr])


        Ro += 0.2
        CrustPercent = 0
        Radius += 0.2
        
    CrustArr = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12])
    RadiusArr = np.linspace(10.0, 14.0, 21)
    TotalArr.shape = (21,6)

    FinalArr = np.column_stack((RadiusArr , TotalArr))
    Final = np.row_stack((CrustArr, FinalArr))

    df = pd.DataFrame(Final)
    print('\n')
    # Set display options for pandas DataFrame
    pd.set_option('float_format', '{:.8f}'.format)

    print(f'Data Of {SolarMass}:')
    print(df)
    
    TurnSolar += 1



