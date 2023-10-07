#########################
# Define helper functions
#########################

from IPython.core.display import HTML
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

klotz_2021 = "https://armscontrolcenter.org/wp-content/uploads/2017/04/LWC-paper-final-version-for-CACNP-website.pdf"
marani_2021 = "https://www.pnas.org/doi/10.1073/pnas.2105482118"
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
        num_years = Parameter(100, "Number of years.")
        population = Parameter(int(9.2e9), "World population.")
        num_simulations = Parameter(100000, "Number of Monte Carlo simulations.")

    class Natural:
        dataset = Parameter("data/Epidemics dataset 21 March 2021.xlsx", "Path of the Marani dataset file.", marani_2021, "Marani 2021")
        colour = Parameter("#2ca02c", "Colour of the natural epidemics for plotting", display=False)
        
    class Accidental:
        P_release = Parameter(0.00246, "Probability of community release from a single facility in a single year.", klotz_2021, "Klotz 2021")
        P_seeds_pandemic_min = Parameter(0.05, "Minimum probability that a virus release seeds a pandemic.", klotz_2021, "Klotz 2021")
        P_seeds_pandemic_max = Parameter(0.4, "Maximum probability that a virus release seeds a pandemic.", klotz_2021, "Klotz 2021")
        num_facilities = Parameter(14, "Number of facilities. Default is the number of  Highly Pathogenic Avian Influenza (HPAI) facilities", klotz_2021, "Klotz 2021")
        fatality_rate = Parameter(0.025, "Case fatality rate (CFR). Default is 1918 influenza CFR", klotz_2021, "Klotz 2021")
        infection_rate = Parameter(0.15, "Infection rate of the pandemic. Default is % infected in typical flu season", klotz_2021, "Klotz 2021")
        colour = Parameter("#1f77b4", "Colour of the accidental epidemics for plotting", display=False)

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

def load_and_preprocess_data(marani_xls):
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

    fig.update_layout(
        title="Epidemics over Time",
        xaxis_title="Year",
        yaxis_title="Number of Deaths",
        yaxis_type="log",
    )

    fig.show()

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
        title=f"Expected number of accidental pandemics over the next century = {E_accidental_pandemics.mean():.2f}",
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
        title=f"Expected number of accidental deaths over the next century = {E_accidental_deaths.mean()/1e6:.1f} million",
        color_discrete_sequence=[accidental_colour],
        histnorm='probability'
    )
    # Turn off the legend
    fig.update_layout(showlegend=False)
    fig.show()
    return fig

