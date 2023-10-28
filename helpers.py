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

# Define URLs for sources
klotz_2021 = "https://armscontrolcenter.org/wp-content/uploads/2017/04/LWC-paper-final-version-for-CACNP-website.pdf"
marani_2021 = "https://www.pnas.org/doi/10.1073/pnas.2105482118"
gtd = "https://www.start.umd.edu/gtd/"
un_pop_projections = "https://www.worldometers.info/world-population/world-population-projections/"
un_pop_us = "https://www.macrotrends.net/countries/USA/united-states/population"
esvelt_2022 = "https://dam.gcsp.ch/files/doc/gcsp-geneva-paper-29-22?_gl=1*1812zfe*_ga*MTk1NzA0MTU3My4xNjk2NzcyODA0*_ga_Z66DSTVXTJ*MTY5NzI4NTA4Ny4yLjEuMTY5NzI4NTE0MC43LjAuMA.."
# Set default template for plots
pio.templates.default = 'plotly_white'

class Parameter:
    def __init__(self, value, description, source_link=None, source_description=None, display=True):
        self.val = value
        self.description = description
        self.source_link = source_link
        self.source_description = source_description
        self.display = display

class Params:
    class Global:
        num_years = Parameter(76, "Number of years.")
        population = Parameter(int(9.2e9), "World population.", un_pop_projections, "United Nations - World Population Projections")
        num_simulations = Parameter(100000, "Number of Monte Carlo simulations.")

    class Natural:
        dataset = Parameter("data/Epidemics dataset 21 March 2021.xlsx", "Path of the Marani dataset file.", marani_2021, "Marani 2021")
        mu = Parameter(1.000e-3, "Threshold for generalized Pareto distribution (GPD).", marani_2021, "Marani 2021")
        sigma = Parameter(0.0113, "Scale parameter for GPD.", marani_2021, "Marani 2021")
        xi = Parameter(1.40, "Shape parameter for GPD.", marani_2021, "Marani 2021")
        p0 = Parameter(0.62, "Probability that an epidemic intensity is less than μ.", marani_2021, "Marani 2021")
        max_intensity = Parameter(100/3, "Maximum intensity of an epidemic. Default is highest credible reports of deaths for a pandemic, 100 million deaths from 1918 influenza over 3 years.")
        colour = Parameter("#2ca02c", "Colour of the natural epidemics for plotting", display=False)
        
    class Accidental:
        P_release = Parameter(0.00246, "Probability of community release from a single facility in a single year.", klotz_2021, "Klotz 2021")
        P_seeds_pandemic_min = Parameter(0.05, "Minimum probability that a virus release seeds a pandemic.", klotz_2021, "Klotz 2021")
        P_seeds_pandemic_max = Parameter(0.4, "Maximum probability that a virus release seeds a pandemic.", klotz_2021, "Klotz 2021")
        num_facilities = Parameter(14, "Number of facilities. Default is the number of  Highly Pathogenic Avian Influenza (HPAI) facilities", klotz_2021, "Klotz 2021")
        fatality_rate = Parameter(0.025, "Case fatality rate (CFR). Default is 1918 influenza CFR", klotz_2021, "Klotz 2021")
        infection_rate = Parameter(0.15, "Infection rate of the pandemic. Default is % infected in typical flu season", klotz_2021, "Klotz 2021")
        colour = Parameter("#1f77b4", "Colour of the accidental epidemics for plotting", display=False)
    
    class Deliberate:
        dataset = Parameter("data/globalterrorismdb_0522dist.xlsx", "Path of the Global Terrorism Database (GTD) file.", gtd, "GTD")
        population_us_1995 = Parameter(226000000, "US population in 1995.", un_pop_us, "United Nations - World Population Prospects")
        deaths_per_attack = Parameter(2, "Number of deaths required for an attack to be considered someone wanting to cause mass harm.")
        num_indv_capability = Parameter(30000, "Number of individuals with the capability to assemble a virus.", esvelt_2022, "Esvelt 2022")
        deliberate_multiplier_max = Parameter(10, "Maximum value for the number of times more deaths that a deliberate pandemic would cause due to multiple releases and/or pathogens.")
        colour = Parameter("#ff7f0e", "Colour of the deliberate pandemics for plotting", display=False)
        # Additional parameters for capability calculations
        years_since_start = Parameter(0, "Years since the start of the timeframe considered for capability assessment.")
        other_labs_years = Parameter(1, "Years taken for other labs to reproduce the advancements.", esvelt_2022, "Esvelt 2022")
        adapted_use_years = Parameter(3, "Years taken for adapted use of advancements.", esvelt_2022, "Esvelt 2022")
        undergrad_years = Parameter(5, "Years taken for undergraduates to reproduce the advancements.", esvelt_2022, "Esvelt 2022")
        high_school_years = Parameter(12, "Years taken for high school students to reproduce the advancements.", esvelt_2022, "Esvelt 2022")
        other_labs_multiplier = Parameter(3, "Multiplier for the number of individuals in other labs relative to current doctorates.")
        adapted_use_multiplier = Parameter(3, "Multiplier for the number of individuals using adapted advancements.")
        undergrad_multiplier = Parameter(10, "Multiplier for the number of undergraduates relative to adapted use.")
        high_school_multiplier = Parameter(10, "Multiplier for the number of high school students relative to undergraduates.")
        us_share_of_doctorates = Parameter(1/3, "Share of US doctorates in the global context.", esvelt_2022, "Esvelt 2022")

    @classmethod
    def print_category(cls, category_name):
        category = getattr(cls, category_name, None)
        if not category:
            print(f"No such category: {category_name}")
            return
        
        max_var_length = max([len(var) for var in vars(category) if not var.startswith("_")])
        
        for var, param in vars(category).items():
            if not var.startswith("_") and param.display:
                source = f'(<a href="{param.source_link}">{param.source_description}</a>)' if param.source_link and param.source_description else ""
                display(HTML(f"<strong>{var.ljust(max_var_length)}:</strong> {param.val} {source}<br><em>{param.description}</em><br>"))

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

def plot_exceedance_probability(
        intensities=None,
        gpd_values=None,
        marani_df=None,
        x="Intensity (deaths per mil/year)",
        title_text="Exceedance frequency of epidemic intensity",
        hover_data_columns=['Location', 'Start Year', 'End Year', '# deaths (thousands)'],
        plot_gpd=True,
        plot_lognorm=True,
        log_axis=True,
        mu=1.000e-3,  # threshold for GPD
        sigma=0.0113,  # scale parameter from the paper
        xi=1.40,  # shape parameter from the paper
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
        disease_order = sorted_data.groupby("disease (total deaths)").max()["# deaths (thousands)"].sort_values(ascending=False).index.tolist()
        
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

    # Update titles and axes
    fig.update_layout(
        title=title_text,
        xaxis_title="Intensity (deaths per mil/year)",
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
        title=f"Expected number of accidental deaths this century = {E_accidental_deaths.mean()/1e6:.1f} million",
        color_discrete_sequence=[accidental_colour],
        histnorm='probability'
    )
    # Turn off the legend
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
    gtd_df["short_summary"] = gtd_df["summary"].str[:100] + "..."
    return gtd_df

def format_intent_fraction(frac_invd_intent):
    reciprocal_value = 1 / frac_invd_intent
    formatted_string = f"Fraction of individuals with the intent to cause mass harm = 1 in ~{reciprocal_value/1e6:,.1f} million"
    return formatted_string

def plot_deaths_per_attack_scatter(gtd_df, deaths_per_attack, num_events, frac_invd_intent):
    """
    Plots a scatter plot of the provided data.
    """
    fig = px.scatter(
        gtd_df,
        x="iyear",
        y="nkill",
        color="attacktype1_txt",
        hover_data=['short_summary'],
        log_y=True,
        labels={"iyear": "Year", "nkill": "Number of deaths", "attacktype1_txt": "Attack Type"},
        title=f"Number of terrorist events in the US with ≥{deaths_per_attack} deaths (1970-2022) = {num_events} → \n" + format_intent_fraction(frac_invd_intent)
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

def plot_E_deliberate_deaths_hist(E_deliberate_deaths, deliberate_colour):
    """Plot a histogram of E_deliberate_deaths."""
    fig = px.histogram(
        E_deliberate_deaths,
        nbins=100,
        labels={'value': '#deliberate_deaths'},
        title=f"Expected number of deliberate deaths this century = {E_deliberate_deaths.mean()/1e6:.1f} million",
        color_discrete_sequence=[deliberate_colour],
        histnorm='probability'
    )
    fig.update_layout(showlegend=False)
    fig.show()
    return fig