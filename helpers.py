#########################
# Define helper functions
#########################

from IPython.core.display import HTML, display
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import genpareto, lognorm
import scipy.stats as stats
import math

# Define URLs for sources
klotz_2021 = "https://armscontrolcenter.org/wp-content/uploads/2017/04/LWC-paper-final-version-for-CACNP-website.pdf"
marani_2021 = "https://www.pnas.org/doi/10.1073/pnas.2105482118"
gtd = "https://www.start.umd.edu/gtd/"
un_pop_projections = "https://www.worldometers.info/world-population/world-population-projections/"
un_pop_us = "https://www.macrotrends.net/countries/USA/united-states/population"
glennerster_2022 = "https://www.nber.org/system/files/working_papers/w30565/w30565.pdf"
watson_2022 = "https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(22)00320-6"
blake_2024 = "https://blog.joshuablake.co.uk/p/forecasting-accidentally-caused-pandemics?r=11048"
esvelt_2022 = "https://dam.gcsp.ch/files/doc/gcsp-geneva-paper-29-22"
esvelt_2023 = "https://dam.gcsp.ch/files/doc/securing-civilisation-against-catastrophic-pandemics-gp-31"
rethink_priorities_ccm = "https://ccm.rethinkpriorities.org/"
nti = "https://www.nti.org/about/programs-projects/project/preventing-the-misuse-of-dna-synthesis-technology/"
secure_dna = "https://secure-dna.up.railway.app/manuscripts/Random_Adversarial_Threshold_Screening.pdf"
open_phil_screening = "https://www.openphilanthropy.org/grants/massachusetts-institute-of-technology-media-lab-dna-synthesis-screening-methods/"
hellewell_2020 = "https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(20)30074-7"
sharma_2023 = "https://pubmed.ncbi.nlm.nih.gov/37367195/"
owid_covid = "https://ourworldindata.org/covid-deaths"
cdc_flu = "https://www.cdc.gov/flu/prevent/vaccine-selection.htm"
who_flu = "https://www.who.int/teams/global-influenza-programme/surveillance-and-monitoring/burden-of-disease"
cdc_ve = "https://www.cdc.gov/flu/vaccines-work/past-seasons-estimates.html"
cdc_ve_matched = "https://www.cdc.gov/flu/vaccines-work/vaccineeffect.htm"
# owid_flu = "https://ourworldindata.org/influenza-deaths"
cepi_cov = "https://cepi.net/news_cepi/the-race-to-future-proof-coronavirus-vaccines/"
gopal_2023 = "https://dam.gcsp.ch/files/doc/securing-civilisation-against-catastrophic-pandemics-gp-31"

# Set default template for plots
pio.templates.default = 'plotly_white'

class Parameter:
    def __init__(self, value, description, source_link=None, source_description=None, display=True, units=None):
        self.val = value
        self.description = description
        self.source_link = source_link
        self.source_description = source_description
        self.display = display
        self.units = units

class Params:
    class Global:
        num_years = Parameter(76, "Number of years. Default is until 2100.")
        population = Parameter(int(9.2e9), "World population.", un_pop_projections, "United Nations - World Population Projections")
        num_simulations = Parameter(100000, "Number of Monte Carlo simulations.")

    class Natural:
        dataset = Parameter("data/Epidemics dataset 21 March 2021.xlsx", "Path of the Marani dataset file.", marani_2021, "Marani 2021")
        mu = Parameter(1.000e-3, "Threshold for generalized Pareto distribution (GPD).", marani_2021, "Marani 2021")
        sigma = Parameter(0.0113, "Scale parameter for GPD.", marani_2021, "Marani 2021")
        xi = Parameter(1.40, "Shape parameter for GPD.", marani_2021, "Marani 2021")
        p0 = Parameter(0.62, "Probability that an epidemic intensity is less than μ.", marani_2021, "Marani 2021")
        max_intensity = Parameter(17.8, "Maximum intensity of an epidemic. Default is highest credible reports of deaths for a pandemic, 100 million deaths from 1918 influenza divided by population at the time and three-year duration.", glennerster_2022, "Glennerster et al. 2022")
        vaccines_multiplier = Parameter(0.37, "Multiplier to account for the effect of vaccines reducing the expected number of deaths from epidemics. Default refers to the estimated 63% reduction in deaths in the first year of COVID-19 vaccines.", watson_2022, "Watson et al. 2022")
        colour = Parameter("#2ca02c", "Colour of the natural epidemics for plotting", display=False)
        
    class Accidental:
        P_release = Parameter(0.00246, "Probability of community release from a single facility in a single year.", klotz_2021, "Klotz 2021")
        P_seeds_pandemic = Parameter((0.05, 0.4), "Probability that a virus release seeds a pandemic.", klotz_2021, "Klotz 2021")
        num_facilities = Parameter(14, "Number of facilities. Default is the number of  Highly Pathogenic Avian Influenza (HPAI) facilities", klotz_2021, "Klotz 2021")
        growth_rate = Parameter(0.025, "Estimate of the annual growth rate for new BSL-4 facilities.", blake_2024, "Blake 2024")
        fatality_rate = Parameter(0.025, "Case fatality rate (CFR). Default is 1918 influenza CFR", klotz_2021, "Klotz 2021")
        infection_rate = Parameter(0.15, "Infection rate of the pandemic. Default is % infected in typical flu season", klotz_2021, "Klotz 2021")
        colour = Parameter("#1f77b4", "Colour of the accidental epidemics for plotting", display=False)
        P_seed_bioweapon = Parameter((0.0062, 0.079), "Probability that a single bioweapon programme seeds a pandemic.", gopal_2023, "Gopal 2023")
    
    class Deliberate:
        dataset = Parameter("data/globalterrorismdb_0522dist.xlsx", "Path of the Global Terrorism Database (GTD) file.", gtd, "GTD")
        population_us_1995 = Parameter(226000000, "US population in 1995 (the midpoint of the time peroid analysed for the GTD dataset.", un_pop_us, "United Nations - World Population Prospects")
        deaths_per_attack = Parameter(2, "Number of deaths required for an attack to be considered someone wanting to cause mass harm.")
        num_indv_capability = Parameter(int(30000/20), "Number of new individuals with the capability to assemble a virus each year", esvelt_2022, "Esvelt 2022")
        num_indv_capability_intent_last_century = Parameter(2, "Number of individuals last century who we assume would have the capability and intent if they were born this century. Deafult is 2 for members from Aum Shinrikyo and al Qaeda", esvelt_2023, "Esvelt 2023")
        retrain_indv_multiplier = Parameter((2,4), "Multiplier for additional number of individuals who will retrain in virology.", esvelt_2022, "Esvelt 2023")
        # retrain_indv_multiplier_min = Parameter(2, "Minimum multiplier value for additional number of individuals who will retrain in virology.", esvelt_2023, "Esvelt 2023")
        # retrain_indv_multiplier_max = Parameter(4, "Maximum multiplier value for additional number of individuals who will retrain in virology.", esvelt_2023, "Esvelt 2023")
        deliberate_multiplier = Parameter((1,10), "Number of times more deaths that a deliberate pandemic would cause due to multiple pathogens and/or releases etc.")
        num_years_until_blueprints = Parameter((0,76*2), "Number of years until blueprints for a pandemic capable pathogen become available.")
        colour = Parameter("#ff7f0e", "Colour of the deliberate pandemics for plotting", display=False)
        # Additional parameters for capability calculations
        years_since_start = Parameter(0, "Years since the start of the timeframe considered for capability assessment.", display=False)
        other_labs_years = Parameter(1, "Years taken for other labs to reproduce the advancements.", esvelt_2022, "Esvelt 2022", display=False)
        adapted_use_years = Parameter(3, "Years taken for adapted use of advancements.", esvelt_2022, "Esvelt 2022", display=False)
        undergrad_years = Parameter(5, "Years taken for undergraduates to reproduce the advancements.", esvelt_2022, "Esvelt 2022", display=False)
        high_school_years = Parameter(12, "Years taken for high school students to reproduce the advancements.", esvelt_2022, "Esvelt 2022", display=False)
        other_labs_multiplier = Parameter(3, "Multiplier for the number of individuals in other labs relative to current doctorates.", display=False)
        adapted_use_multiplier = Parameter(3, "Multiplier for the number of individuals using adapted advancements.", display=False)
        undergrad_multiplier = Parameter(10, "Multiplier for the number of undergraduates relative to adapted use.", display=False)
        high_school_multiplier = Parameter(10, "Multiplier for the number of high school students relative to undergraduates.", display=False)
        us_share_of_doctorates = Parameter(1/3, "Share of US doctorates in the global context.", esvelt_2022, "Esvelt 2022", display=False)
    
    class Interventions:
        give_well_cost_effectiveness = Parameter(21, "Cost effectiveness for GiveWell funding bar.", rethink_priorities_ccm, "Rethink Priorities CCM", units="DALYs/$1000")
        cage_free_chicken_campaign_cost_effectiveness = Parameter(717, "Cost effectiveness of cage free chicken campaigns.", rethink_priorities_ccm, "Rethink Priorities CCM", units="DALYs/$1000")
        average_dalys_per_life_saved = Parameter(30, "Average DALYs per life saved. Default = (global life expectancy - global mean age) x disability weight = (70 - 30) x 0.75") 
        
    class DNA_Screening:
        P_no_benchtop = Parameter(0.8, "Probability that benchtop DNA synthesis machines are not used to bypass the screening.")
        P_no_academic_approval = Parameter(0.75, "Probability that the order does not have academic approval (assuming orders with academic approval bypass the screening).", esvelt_2023, "Gopal et al. 2023")
        P_pathogen_in_database = Parameter(0.5, "Probability that the pathogen is listed in the screening database.")
        P_screening_effective = Parameter(0.9996, "Probability that the screening, if conducted, successfully identifies and stops a bioterrorist attack.", secure_dna, "SecureDNA")
        dna_screening_cost = Parameter(100000000, "Cost to fully develop, implement and regulate DNA synthesis screening. Open Phil has already donated 890K to SecureDNA and 10M+ to NTI", open_phil_screening, "Open Phil", units="$")

    class Sequencing:
        P_containment = Parameter(0.8, "Probability of containing an outbreak upon detection via contact tracing and isolation of cases. Assuming R0=2.5, contacts traced>=80%, transmission before symptom onset=<1% and initial cases=40.", hellewell_2020, "Hellewell et al. 2020")
        threatnet_cost = Parameter((400000000, 800000000), "Annual cost of ThreatNet system for early detection of novel pathogens in the US.", sharma_2023, "Sharma et al. 2023")
        us_pop_frac_global = Parameter(0.04, "Fraction of global population in the US.")

    class Broad_Vaccines:
        additional_vaccine_years = Parameter(5, "Number of years earlier that a broad-spectrum vaccine is developed as a result of increased funding.")
        covid_excess_deaths = Parameter(28370000, "Excess deaths due to COVID-19 as of 2024.", owid_covid, "Our World in Data")
        time_to_update_vaccine = Parameter((0.5,1), "Time to update vaccine.", cdc_flu, "CDC", units="years")
        current_flu_vaccine_effectiveness = Parameter((0.19,0.6), "Influenza vaccine effectiveness in the US from 2004 to 2023.", cdc_ve, "CDC")
        broad_vaccine_effectiveness = Parameter((0.4,0.6), "Effectiveness of current flu vaccines when antigenically matched.", cdc_ve_matched, "CDC")
        annual_global_flu_deaths = Parameter((290000, 650000), "Annual global flu deaths.", who_flu, "WHO")
        broad_cov_research_funding = Parameter(200000000, "Current funding for broadly protective coronavirus vaccine portfolio.", cepi_cov, "CEPI", units="$")
        dalys_lost_per_seasonal_flu_death = Parameter(2, "Rough estimate for DALYs lost per death from seasonal influenza.")

    @classmethod
    def print_category(cls, category_name):
        category = getattr(cls, category_name, None)
        if not category:
            print(f"No such category: {category_name}")
            return
        
        max_var_length = max([len(var) for var in vars(category) if not var.startswith("_")])
        
        for var, param in vars(category).items():
            if not var.startswith("_") and param.display:
                val = param.val if not isinstance(param.val, tuple) else f"{param.val[0]} - {param.val[1]}"
                if param.units == "%":
                    val = f"{val}%"
                elif param.units == "$":
                    val = f"${format_number(val)}"
                elif param.units:
                    val = f"{val} {param.units}"
                source = f'(<a href="{param.source_link}">{param.source_description}</a>)' if param.source_link and param.source_description else ""
                display(HTML(f"<strong>{var.ljust(max_var_length)}:</strong> {val} {source}<br><em>{param.description}</em><br>"))

def display_text(text, size=20):
    return display(HTML(f"<span style=\"font-size: {size}px;\">{text}</span>"))

##############
# Natural risk
##############

def format_number(n):
    """Format number with commas and replace numbers > 1 million with 'million' representation."""
    n = int(n)
    if n >= 1e6:
        million_value = n/1e6
        if million_value == int(million_value):  # Check if the decimal part is zero
            return "{} million".format(int(million_value))
        else:
            return "{:.1f} million".format(million_value)
    else:
        return "{:,}".format(n)


def load_and_preprocess_natural_data(marani_xls):
    """Load and preprocess the data."""
    # Load the data from "Sheet1" up to row 540
    data = pd.read_excel(marani_xls, 'Sheet1', nrows=539)

    # Filter out rows where "# deaths (thousands)" is -999 (placeholder for missing data)
    filtered_data = data[data["# deaths (thousands)"] != -999]

    # Remove any unnamed columns
    filtered_data = filtered_data.loc[:, ~filtered_data.columns.str.startswith('Unnamed')]

    # Convert the deaths from thousands to raw values
    filtered_data["# deaths (raw)"] = filtered_data["# deaths (thousands)"] * 1000

    # Handle NaN values in the 'Disease' column by assigning them a default label
    filtered_data['Disease'].fillna('Unknown Disease', inplace=True)

    # Calculate total deaths for each disease
    disease_totals = filtered_data.groupby('Disease')['# deaths (raw)'].sum().sort_values(ascending=False)

    # Add a column to the dataframe for "disease (total deaths)"
    filtered_data['disease (total deaths)'] = filtered_data['Disease'].apply(
        lambda x: f"{x} ({format_number(disease_totals.get(x, 0))} deaths)"
    )

    # Add a column just for the disease total deaths number
    filtered_data['disease_total_deaths'] = filtered_data['Disease'].map(disease_totals)

    # Calculate exceedance probability
    # filtered_data['Rank'] = filtered_data["Intensity (deaths per mil/year)"].rank(ascending=True)
    filtered_data = filtered_data.sort_values(by="Intensity (deaths per mil/year)", ascending=True)
    filtered_data["Exceedance Probability"] = 1 - np.arange(len(filtered_data)) / len(filtered_data) 
    # = 1 - filtered_data['Rank'] / len(filtered_data)

    # Rename the units for the intensity column - I think it's supposed to be deaths per thousand
    filtered_data.rename(columns={'Intensity (deaths per mil/year)': 'Intensity (deaths per thousand/year)'}, inplace=True)

    # Add a column for the duration of the outbreak
    filtered_data["Duration"] = filtered_data["End Year"] - filtered_data["Start Year"] + 1

    # Sort dataframe based on the disease total deaths number
    filtered_data = filtered_data.sort_values(by='disease_total_deaths', ascending=False)

    return filtered_data, disease_totals


def generate_color_map(disease_totals):
    """Generate a color map for diseases."""
    # Create a color mapping for each unique disease label in the order of total deaths
    ordered_diseases = disease_totals.sort_values(ascending=False).index
    color_map = {disease: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, disease in enumerate(ordered_diseases)}
    return color_map

def plot_disease_timeline(filtered_data, disease_totals, color_map):
    """Plot disease outbreaks over time."""
    fig = go.Figure()
    added_to_legend = set()

    # Add traces for each disease event, ordered by total deaths
    for disease, total in disease_totals.items():
        subset_data = filtered_data[filtered_data["Disease"] == disease]

        for _, row in subset_data.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["Start Year"], row["End Year"]],
                    y=[row["# deaths (raw)"], row["# deaths (raw)"]],
                    mode="lines+markers",
                    name=f"{disease} ({format_number(total)} deaths)",
                    line=dict(color=color_map[disease]),
                    marker=dict(color=color_map[disease]),
                    legendgroup=disease,  # Grouping by disease to unify the legend behavior
                    showlegend=True if disease not in added_to_legend else False
                )
            )
            # Add the disease to our set to ensure it doesn't get added to the legend again
            added_to_legend.add(disease)
    
    # Colour the background for years 1600-1944

    fig.update_layout(
        title="Epidemics over Time",
        xaxis_title="Year",
        yaxis_title="Number of Deaths",
        yaxis_type="log",
        shapes=[
            # Add a rectangular shape to the background
            dict(
                type="rect",
                xref="x",
                yref="paper",  # Use 'paper' for yref to cover the full y range
                x0=1600,  # Start year
                x1=1944,  # End year
                y0=0,  # Start at the bottom
                y1=1,  # Extend to the top
                fillcolor="#2ca02c",
                opacity=0.1,  # Adjust the opacity if needed
                layer="below",
                line_width=0,
            )
        ]
    )

    fig.show()
    return fig

def compute_gpd(x, mu=None, sigma=None, xi=None):
    """Compute the GPD."""
    if mu == None or sigma == None or xi == None:
        xi, mu, sigma = genpareto.fit(x)
    gpd_values = 1 - genpareto.cdf(x - mu, xi, scale=sigma)
    return gpd_values

def compute_lognorm(intensities):
    """Compute the log-normal."""
    s, loc_ln, scale_ln = lognorm.fit(intensities)
    lognorm_values = 1 - lognorm.cdf(intensities, s=s, loc=loc_ln, scale=scale_ln)
    return lognorm_values

def compute_lognorm(marani_df, x, intensities):
    """Compute the log-normal."""
    s, loc_ln, scale_ln = lognorm.fit(marani_df[x])
    lognorm_values = 1 - lognorm.cdf(intensities, s=s, loc=loc_ln, scale=scale_ln)
    return lognorm_values

def wilson_score_interval(x, n, confidence=0.95):
    """
    Calculate the Wilson score interval for a binomial proportion.

    Parameters:
        x (int): Number of successes.
        n (int): Sample size.
        confidence (float): Confidence level (default is 0.95 for 95% confidence).

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = x / n
    lower_bound = (p_hat + (z**2) / (2 * n) - z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n)
    upper_bound = (p_hat + (z**2) / (2 * n) + z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / (1 + (z**2) / n)
    return lower_bound, upper_bound

# vectorize the function to apply it to numpy arrays
wilson_score_interval = np.vectorize(wilson_score_interval)

def plot_exceedance_probability(
        intensities=None,
        gpd_values=None,
        marani_df=None,
        x="Intensity (deaths per thousand/year)",
        title_text="Exceedance frequency of epidemic intensity",
        hover_data_columns=['Location', 'Start Year', 'End Year', '# deaths (thousands)'],
        plot_gpd=True,
        plot_lognorm=True,
        plot_CI = True,
        log_axis=True,
        mu=1.000e-3,  # threshold for GPD
        sigma=0.0113,  # scale parameter from the paper
        xi=1.40,  # shape parameter from the paper
        confident_level=0.95, # 95% confidence interval
        colour="#2ca02c"
    ):
    """
    Plot intensity exceedance probability and overlay GPD and Log-normal fits using parameters from the paper.
    """
    # Plot scatter if data is provided
    if marani_df is not None:
        # Sort the dataframe by intensity
        sorted_data = marani_df.sort_values(by=x, ascending=True)
        exceedance_probs = 1 - np.arange(len(sorted_data)) / len(sorted_data)
        # Determine order of diseases by their total deaths
        disease_order = sorted_data.groupby("disease (total deaths)").max('disease_total_deaths')["# deaths (thousands)"].sort_values(ascending=False).index.tolist()
        
        fig = px.scatter(sorted_data, 
                         x=x, 
                         y=exceedance_probs, 
                         color="disease (total deaths)", 
                         log_x=True, 
                         log_y=True,
                         hover_data=hover_data_columns, 
                         category_orders={"disease (total deaths)": disease_order}
                        )
    else:
        fig = go.Figure()
        
    # Generate GPD values and add them to the plot
    if intensities is None:
        intensities = np.linspace(marani_df[x].min(), marani_df[x].max(), 1000) if marani_df is not None else np.linspace(0, 0.01, 1000)
    if gpd_values is None:
        gpd_values = compute_gpd(intensities, mu, sigma, xi) 
    if plot_gpd:
        fig.add_trace(go.Scatter(x=intensities, y=gpd_values, mode='lines', name='GPD Fit', line=dict(color=colour)))

    # Fit the log-normal distribution to the data and compute exceedance probabilities
    if marani_df is not None and plot_lognorm:
        lognorm_values = compute_lognorm(marani_df, x, intensities)
        fig.add_trace(go.Scatter(x=intensities, y=lognorm_values, mode='lines', name='Log-normal Fit', line=dict(color="red")))

    # Plot 95% confidence intervals using the Wilson score interval
    if marani_df is not None and plot_CI:
        n = len(sorted_data)
        exceedance_probs = 1 - np.arange(n) / n
        lowers, uppers = wilson_score_interval(exceedance_probs * n, n, confidence = confident_level)
        fig.add_trace(go.Scatter(x=sorted_data[x], y=lowers, name='95% CI Lower Bound', line=dict(color="gray", dash="dash")))
        fig.add_trace(go.Scatter(x=sorted_data[x], y=uppers, name='95% CI Upper Bound', line=dict(color="gray", dash="dash")))

    # Update titles and axes
    fig.update_layout(
        title=title_text,
        xaxis_title="Intensity (deaths per thousand/year)",
        yaxis_title="Exceedance Probability",
        legend_title="Disease",
    )
    if log_axis:
        fig.update_xaxes(type='log', exponentformat='power', showexponent='all')
        fig.update_yaxes(type='log', exponentformat='power', showexponent='all')
    
    fig.show()
    return fig

#################
# Accidental risk
#################

def plot_P_single_pandemic_hist(P_single_pandemic, accidental_colour="#1f77b4"):
    """
    Plots a histogram of the provided data.

    Parameters:
    - P_single_pandemic: The data to plot.
    - accidental_colour: The color to use for the histogram.

    Returns:
    - fig: The plotly figure object.
    """
    fig = px.histogram(
        P_single_pandemic, 
        nbins=100, 
        labels={'value': 'P(single_pandemic)'}, 
        title=f"Probability a single facility in a single year seeds a pandemic = {P_single_pandemic.mean():.2%}", 
        color_discrete_sequence=[accidental_colour],
        histnorm='probability'
    )
    # Turn off the legend
    fig.update_yaxes(title_text='Probability')
    fig.update_layout(showlegend=False)
    fig.show()
    return fig

def plot_E_accidental_pandemics_hist(E_accidental_pandemics, accidental_colour):
    """
    Plots a histogram of the expected number of accidental pandemics.

    Parameters:
    - E_accidental_pandemics: The data to plot.
    - accidental_colour: The color to use for the histogram.

    Returns:
    - fig: The plotly figure object.
    """
    fig = px.histogram(
        E_accidental_pandemics,
        nbins=100,
        labels={'value': '#accidental_pandemics'},
        title=f"Expected number of accidental pandemics this century = {E_accidental_pandemics.mean():.2f}",
        color_discrete_sequence=[accidental_colour],
        histnorm='probability'
    )
    # Turn off the legend
    fig.update_yaxes(title_text='Probability')
    fig.update_layout(showlegend=False)
    fig.show()
    return fig

def plot_E_accidental_deaths_hist(E_accidental_deaths, accidental_colour):
    """
    Plots a histogram of the expected number of accidental deaths.

    Parameters:
    - E_accidental_deaths: The data to plot.
    - accidental_colour: The color to use for the histogram.

    Returns:
    - fig: The plotly figure object.
    """
    fig = px.histogram(
        E_accidental_deaths,
        nbins=100,
        labels={'value': '#accidental_deaths'},
        title=f"Expected number of deaths from accidental pandemics this century = {E_accidental_deaths.mean()/1e6:.1f} million",
        color_discrete_sequence=[accidental_colour],
        histnorm='probability'
    )
    # Turn off the legend
    fig.update_yaxes(title_text='Probability')
    fig.update_layout(showlegend=False)
    fig.show()
    return fig

#################
# Deliberate risk
#################

def load_and_preprocess_deliberate_data(gtd_xls, deaths_per_attack):
    """
    Loads and preprocesses the deliberate risk data.
    """
    gtd_df = pd.read_excel(gtd_xls, sheet_name='Data')
    gtd_df = gtd_df[gtd_df["country_txt"] == "United States"]
    gtd_df = gtd_df[gtd_df["nkill"] >= deaths_per_attack]
    gtd_df = gtd_df[(gtd_df["iyear"] >= 1970) & (gtd_df["iyear"] <= 2020)]
    gtd_df["short_summary"] = gtd_df["summary"].str[:100] + "..."
    return gtd_df

def format_intent_fraction(frac_invd_intent):
    reciprocal_value = 1 / frac_invd_intent
    formatted_string = f"Fraction of individuals with the intent to cause mass harm = 1 in ~{reciprocal_value/1e6:,.1f} million"
    return formatted_string

def plot_deaths_per_attack_scatter(gtd_df, deaths_per_attack, num_events, frac_invd_intent=None):
    """
    Plots a scatter plot of the provided data.
    """
    title = f"Number of terrorist events in the US with ≥{deaths_per_attack} deaths (1970-2020) = {num_events}"
    if frac_invd_intent:
        title += f"→ \n" + format_intent_fraction(frac_invd_intent)
    fig = px.scatter(
        gtd_df,
        x="iyear",
        y="nkill",
        color="attacktype1_txt",
        hover_data=['short_summary'],
        log_y=True,
        labels={"iyear": "Year", "nkill": "Number of deaths", "attacktype1_txt": "Attack Type"},
        title=title
    )
    fig.show()
    return fig

def calculate_individual_capability(params):
    """Calculate the number of individuals with capability over time based on given parameters."""
    total_individuals = np.zeros(params.Global.num_years.val)
    for year in range(params.Global.num_years.val):
        if year >= params.Deliberate.years_since_start.val:
            total_individuals[year] += params.Deliberate.num_indv_capability.val
        if year >= params.Deliberate.years_since_start.val + params.Deliberate.other_labs_years.val:
            total_individuals[year] += params.Deliberate.num_indv_capability.val * params.Deliberate.other_labs_multiplier.val
        if year >= params.Deliberate.years_since_start.val + params.Deliberate.adapted_use_years.val:
            total_individuals[year] += params.Deliberate.num_indv_capability.val * params.Deliberate.other_labs_multiplier.val * params.Deliberate.adapted_use_multiplier.val
        if year >= params.Deliberate.years_since_start.val + params.Deliberate.undergrad_years.val:
            total_individuals[year] += params.Deliberate.num_indv_capability.val * params.Deliberate.other_labs_multiplier.val * params.Deliberate.adapted_use_multiplier.val * params.Deliberate.undergrad_multiplier.val
        if year >= params.Deliberate.years_since_start.val + params.Deliberate.high_school_years.val:
            total_individuals[year] += params.Deliberate.num_indv_capability.val * params.Deliberate.other_labs_multiplier.val * params.Deliberate.adapted_use_multiplier.val * params.Deliberate.undergrad_multiplier.val * params.Deliberate.high_school_multiplier.val
    total_individuals *= params.Deliberate.us_share_of_doctorates.val
    return total_individuals

def plot_capability_growth(params, deliberate_colour, start_year=2023):
    """Plot the growth of individuals with capability over time."""
    total_individuals = calculate_individual_capability(params)
    title_text = f"Number of Individuals with the Capability to Assemble a Virus This Century = {total_individuals[-1]:,.0f}"

    # Adjust x-values to start from 2023
    x_values = list(range(start_year, start_year + params.Global.num_years.val))
    
    fig = go.Figure(data=[
        go.Scatter(x=x_values, y=total_individuals, mode='lines', name='Total Individuals with Capability', line=dict(color=deliberate_colour))
    ])
    # Add annotations with adjusted positions
    annotations = [
        dict(x=start_year + params.Deliberate.years_since_start.val, y=total_individuals[params.Deliberate.years_since_start.val], xref="x", yref="y", text="Doctorates", showarrow=True, arrowhead=4, ax=0, ay=-20),
        dict(x=start_year + params.Deliberate.years_since_start.val + params.Deliberate.other_labs_years.val, y=total_individuals[params.Deliberate.years_since_start.val + params.Deliberate.other_labs_years.val], xref="x", yref="y", text="Other Labs", showarrow=True, arrowhead=4, ax=0, ay=-50),
        dict(x=start_year + params.Deliberate.years_since_start.val + params.Deliberate.adapted_use_years.val, y=total_individuals[params.Deliberate.years_since_start.val + params.Deliberate.adapted_use_years.val], xref="x", yref="y", text="Adapted Use", showarrow=True, arrowhead=4, ax=0, ay=-80),
        dict(x=start_year + params.Deliberate.years_since_start.val + params.Deliberate.undergrad_years.val, y=total_individuals[params.Deliberate.years_since_start.val + params.Deliberate.undergrad_years.val], xref="x", yref="y", text="Undergraduates", showarrow=True, arrowhead=4, ax=0, ay=-110),
        dict(x=start_year + params.Deliberate.years_since_start.val + params.Deliberate.high_school_years.val, y=total_individuals[params.Deliberate.years_since_start.val + params.Deliberate.high_school_years.val], xref="x", yref="y", text="High School Students", showarrow=True, arrowhead=4, ax=0, ay=-140)
    ]
    fig.update_layout(title=title_text, xaxis_title="Years", yaxis_title="Number of Individuals", annotations=annotations)
    fig.show()
    return fig

def plot_E_deliberate_deaths_over_time(E_deliberate_deaths_avg, num_years, deliberate_colour, start_year=2023):
    fig = go.Figure(data=[
        go.Scatter(x=list(range(start_year, start_year + num_years)), y=E_deliberate_deaths_avg, mode='lines', name='Expected Deaths', line=dict(color=deliberate_colour))
    ])
    fig.update_layout(title="Expected number of deaths from deliberate pandemics over this century", xaxis_title="Years", yaxis_title="Expected Deaths")
    fig.show()
    return fig

def plot_E_deliberate_deaths_hist(E_deliberate_deaths, deliberate_colour):
    """Plot a histogram of E_deliberate_deaths."""
    fig = px.histogram(
        E_deliberate_deaths,
        nbins=100,
        labels={'value': '#deliberate_deaths'},
        title=f"Expected number of deaths from deliberate pandemics this century = {E_deliberate_deaths.mean()/1e6:.1f} million",
        color_discrete_sequence=[deliberate_colour],
        histnorm='probability'
    )
    fig.update_yaxes(title_text='Probability')
    fig.update_layout(showlegend=False)
    fig.show()
    return fig

def plot_comparative_E_deliberate_deaths_hist(E_deliberate_deaths_without_screening, E_deliberate_deaths_with_screening, deliberate_colour):

    lives_saved = np.mean(E_deliberate_deaths_without_screening) - np.mean(E_deliberate_deaths_with_screening)

    # Create histograms for both scenarios
    trace1 = go.Histogram(
        x=E_deliberate_deaths_without_screening,
        histnorm='probability',
        name='Without Screening',
        opacity=0.75,
        marker=dict(color=deliberate_colour)
    )
    
    trace2 = go.Histogram(
        x=E_deliberate_deaths_with_screening,
        histnorm='probability',
        name='With Screening',
        opacity=0.75,
        marker=dict(color='green')
    )
    
    # Create the layout, including a title and axis labels
    layout = go.Layout(
        title=f"Expected number of lives saved by DNA synthesis screening = {lives_saved/1e6:.1f} million",
        xaxis=dict(title='Expected Number of Deaths'),
        yaxis=dict(title='Probability'),
        barmode='overlay'
    )
    
    # Create the figure with the two histograms
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    # Display the figure
    fig.show()

###############
# Interventions
###############

def estimate_deaths_vectorized(ve_array, base_deaths=650000, min_deaths=290000):
    """
    Estimates annual deaths based on an array of vaccine effectiveness values.

    :param ve_array: Numpy array of Vaccine Effectiveness values (ranging from 0 to 1).
    :param base_deaths: The number of deaths at the lowest VE (default 600,000).
    :param min_deaths: The minimum possible number of deaths (default 300,000).
    :return: Numpy array of estimated annual deaths corresponding to each VE.
    """
    if np.any((ve_array < 0) | (ve_array > 1)):
        raise ValueError("Vaccine effectiveness values must be between 0 and 1.")

    # Linear inverse proportion for each VE value
    deaths_array = base_deaths * (1 - ve_array) + min_deaths * ve_array

    return deaths_array.astype(int)

def plot_lives_saved_comparison(lives_saved_seasonal, lives_saved_pandemic):
    """
    Plots a comparison of lives saved due to seasonal and pandemic vaccines.

    :param lives_saved_seasonal: Numpy array of lives saved by seasonal vaccines.
    :param lives_saved_pandemic: Numpy array of lives saved by pandemic vaccines.
    """

    # Calculate the total lives saved
    total_lives_saved = np.mean(lives_saved_seasonal) + np.mean(lives_saved_pandemic)

    # Create histograms for both scenarios
    trace1 = go.Histogram(
        x=lives_saved_seasonal,
        histnorm='probability',
        name='Seasonal Influenza',
        opacity=0.75,
    )
    
    trace2 = go.Histogram(
        x=lives_saved_pandemic,
        histnorm='probability',
        name='Pandemic Coronavirus and Influenza',
        opacity=0.75,
    )
    
    # Create the layout, including a title and axis labels
    layout = go.Layout(
        title=f"Total lives saved by pan-coronavirus and pan-influenza vaccines = {total_lives_saved/1e6:.1f} million",
        xaxis=dict(title='Lives Saved'),
        yaxis=dict(title='Probability'),
        barmode='overlay'
    )
    
    # Create the figure with the two histograms
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    # Display the figure
    fig.show()
    return fig