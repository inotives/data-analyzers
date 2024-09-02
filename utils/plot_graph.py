import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


# -- VISUAL with PLOTLY  -----------------------------------------------------------------

def plotly_line_chart(data, title, xlab, ylab): 
    '''Plot Line chart with Plotly'''
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


def visualize_sentiment(data):
    """Plot sentimental distribution. """
    fig = px.bar(
        data_frame=data,
        x='sentiment_label',
        title="Sentiment Analysis Results",
        labels={'sentiment_label': 'Sentiment', 'count': 'Count'},
        color='sentiment_label'
    )
    fig.show()



# -- VISUAL with MATPLOTLIB  -----------------------------------------------------------------

def mpl_line_chart(data, title, xlab, ylab):
    '''Plot Linechart with matplotlib'''
    plt.figure(figsize=(10, 5))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)

    for d in data: 
        plt.plot(d['xvals'], d['yvals'], label=d['label'], marker=d['marker'])

    plt.legend()
    plt.show()