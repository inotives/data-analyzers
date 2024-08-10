import pandas as pd 
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = f"{ROOT_DIR}/data"

# Importing Data from CSV 
def load_csv_from_data(csv_file):
    
    file_path = f"{DATA_DIR}/{csv_file}.csv"

    return pd.read_csv(file_path)

# Exporting Data to CSV
def export_data_to_csv(data, filename):

    exported_dir = f"{DATA_DIR}/_OUTPUTS/{filename}.csv"
    
    data.to_csv(exported_dir, index=False)
    
    print(f">> Data exported to :: {exported_dir}")

    return 

# -- PLOTTING METHODS -----------------------------------------------------------------

'''Plot Line chart with Plotly'''
def plotly_line_chart(data, title, xlab, ylab): 
    # Plot using Plotly
    fig = go.Figure()

    for d in data: 
        # Add actual prices trace
        fig.add_trace(go.Scatter(x=d['xvals'], y=d['yvals'], mode=d['plotly_mode'], name=d['label']))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        legend_title='Legend'
    )

    return fig

'''Plot Linechart with matplotlib'''
def mpl_line_chart(data, title, xlab, ylab):

    plt.figure(figsize=(10, 5))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)

    for d in data: 
        plt.plot(d['xvals'], d['yvals'], label=d['label'], marker=d['marker'])

    plt.legend()
    plt.show()