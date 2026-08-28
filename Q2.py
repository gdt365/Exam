import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#THE MODEL
params = {
    "N": 50000, #Individuals
    "p_e": (0.40, 0.35, 0.25), #short, medium, or long education probabilities
    "S_e": (1, 3, 5), #years of education for short, medium, or long education
    "h_e0": (1.00, 1.20, 1.55), #initial human capital for short, medium, or long education
    "delta_e": (0.010, 0.020, 0.030), #human capital growth while employed for short, medium, or long education
    "delta": 0.06, #human capital depreciation while unemployed
    "sigma_psi": 0.10, #standard deviation of log-normal distribution for individual productivity shocks
    "lambda": 0.60, #probability of finding job while unemployed
    "sigma": 0.05, #probability of losing job while employed
    "y_su": 0.45, #student grant while in education
    "rho": 0.60, #replacement rate for the unemployed
    "y_floor": 0.35, #minimum income for the unemployed
    "age_start": 18,
    "age_end": 65,
    "seed": 2025,
    "ability_threshold": (0.00, 0.20, 0.35), # For 2.5: minimum ability for short, medium, and long education
}

#Random number generator
rng = np.random.default_rng(params["seed"])

edu = rng.choice(len(params["p_e"]), size=params["N"], p=params["p_e"]) #Education of 0 = short, 1 = medium, 2 = long given probabilities p_e


def sim_model(
    params,
    no_education_differences=False, #For 2.4
    no_hc_shocks=False, #For 2.4
    no_unemployment_depreciation=False, #For 2.4
    no_unemployment=False, #For 2.4
    ability_demotion=False, #For 2.5
): 
    
    rng = np.random.default_rng(params["seed"])
    education = rng.choice(len(params["p_e"]), size=params["N"], p=params["p_e"])

    #For
    if no_education_differences:
        education = np.ones(params["N"], dtype=int)
        intended_education = education.copy()

    #For 2.5: Introduce effort/abilities as a threshold to graduate at intended education
    intended_education = education.copy()  #preserves intended education before possible demotion
    ability = np.full(params["N"], np.nan) 
    if ability_demotion: 
        #Separate RNG preserves all baseline employment and shock draws.
        ability_rng = np.random.default_rng(params["seed"] + 1)
        ability = ability_rng.uniform(0, 1, size=params["N"]) #effort/ability is in interval [0,1]
        thresholds = np.asarray(params["ability_threshold"])
        failed = ability < thresholds[intended_education]
        education[failed] = np.maximum(intended_education[failed] - 1, 0) 
        
    #Create arrays for the parameters that depend on education level
    S_e = np.array(params["S_e"])
    h_e0 = np.array(params["h_e0"])
    delta_e = np.array(params["delta_e"])

    S_e_person = S_e[education] #years of education for each individual
    h_e0_person = h_e0[education] #initial human capital for each individual
    delta_e_person = delta_e[education] #human capital growth while employed for each individual


    #Initial states
    employed = np.zeros(params["N"], dtype=bool)
    active = np.zeros(params["N"], dtype=bool)
    human_capital = np.zeros(params["N"], dtype=float)
    income = np.full(params["N"], params["y_su"], dtype=float)
    last_job_income = np.zeros(params["N"], dtype=float)
    yearly_results = []

    for age in range(params["age_start"],params["age_end"]+1):
        years_since_start = age - params["age_start"]

        still_in_edu = years_since_start < S_e_person
        just_graduated = years_since_start == S_e_person

        #Graduates enter the labour force unemployed
        active[just_graduated] = True
        employed[just_graduated] = no_unemployment
        human_capital[just_graduated] = h_e0_person[just_graduated]

        previously_employed = employed.copy()

        #Probability of individual finding and loosing job
        job_find = rng.binomial(
            n=1,
            p=params["lambda"],
            size=params["N"]
        ).astype(bool)

        job_loss = rng.binomial(
            n=1,
            p=params["sigma"],
            size=params["N"]
        ).astype(bool)
        
        #Outcomes for active individuals
        newly_unemployed = (active & previously_employed & job_loss)

        newly_employed = (active & ~previously_employed & job_find)

        #Updated employment status
        if not no_unemployment:
            employed[newly_unemployed] = False
            employed[newly_employed] = True

        #Individual human capital shock
        psi_t = rng.lognormal(-0.5 * params["sigma_psi"] ** 2, params["sigma_psi"], size=params["N"])
        if no_hc_shocks:
            psi_t = np.ones(params["N"])

        #Updated human capital
        hc_employed = human_capital * (1 + delta_e_person) * psi_t
        unemployment_growth = 1 if no_unemployment_depreciation else 1 - params["delta"]
        hc_unemployed = human_capital * unemployment_growth * psi_t

        new_hc = np.where(
            employed,
            hc_employed,
            hc_unemployed
        )

        human_capital = np.where(
            active,
            new_hc,
            human_capital
        )

        #Income depends on the individual's human capital and employment in the current year.
        employed_income = human_capital
        last_job_income[employed] = employed_income[employed]

        unemployed_income = np.maximum(
            params["rho"] * last_job_income,
            params["y_floor"]
        )

        income = np.where(
            still_in_edu,
            params["y_su"],
            np.where(
                employed,
                employed_income,
                unemployed_income
            )
        )

        yearly_results.append(
            pd.DataFrame({
                "id": np.arange(params["N"]),
                "age": age,
                "education": education,
                "intended_education": intended_education, #For 2.5
                "ability": ability, #For 2.5
                "income": income.copy(),
                "human_capital": human_capital.copy(),
                "employed": employed.copy(),
                "active": active.copy(),
            })
        )

    results = pd.concat(yearly_results, ignore_index=True)

    return results