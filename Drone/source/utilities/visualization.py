import matplotlib.pyplot as plt
import seaborn as sns
import logging
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class DataVisualizer:
    """
    A class to handle various types of data visualizations, with integrated logging for monitoring visualization processes.
    Uses Seaborn, Matplotlib, and Plotly for plotting.
    """

    def __init__(self, logger=None):
        # Setting up the seaborn style for all plots
        sns.set(style="whitegrid")
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("DataVisualizer initialized with Seaborn style set to 'whitegrid'.")

    def plot_time_series(self, data, x, y, title="Time Series Plot", xlabel="Time", ylabel="Value", save_path=None):
        """
        Generate a time series plot with logging of key steps.

        Args:
            data (DataFrame): Pandas DataFrame containing the data to plot.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=data, x=x, y=y)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Time Series plot saved to %s.", save_path)
            plt.show()
            self.logger.info("Time Series plot created successfully for %s vs %s.", x, y)
        except Exception as e:
            self.logger.error("Failed to create time series plot: %s", e)

    def plot_histogram(self, data, column, title="Histogram", bins=30, save_path=None):
        """
        Generate a histogram for a specified column with error handling and logging.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            column (str): Column to plot histogram for.
            title (str): Title of the histogram.
            bins (int): Number of bins for the histogram.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[column], bins=bins, kde=True)
            plt.title(title)
            plt.xlabel(column)
            plt.ylabel("Frequency")
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Histogram saved to %s.", save_path)
            plt.show()
            self.logger.info("Histogram plotted successfully for column: %s.", column)
        except Exception as e:
            self.logger.error("Failed to plot histogram for %s: %s", column, e)

    def plot_correlation_matrix(self, data, title="Correlation Matrix", save_path=None):
        """
        Generate a heatmap for the correlation matrix of the dataframe.
        
        Args:
            data (DataFrame): Pandas DataFrame to compute the correlation matrix from.
            title (str): Title for the correlation matrix plot.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(12, 8))
            corr = data.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title(title)
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Correlation matrix heatmap saved to %s.", save_path)
            plt.show()
            self.logger.info("Correlation matrix heatmap generated successfully.")
        except Exception as e:
            self.logger.error("Failed to generate correlation matrix heatmap: %s", e)

    def plot_scatter(self, data, x, y, hue=None, title="Scatter Plot", xlabel="X", ylabel="Y", save_path=None):
        """
        Generate a scatter plot with detailed logging.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            hue (str, optional): Column name for grouping variable that will produce points with different colors.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x=x, y=y, hue=hue, palette="viridis")
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Scatter plot saved to %s.", save_path)
            plt.show()
            self.logger.info("Scatter plot created successfully for %s vs %s.", x, y)
        except Exception as e:
            self.logger.error("Failed to create scatter plot for %s vs %s: %s", x, y, e)

    def plot_pair(self, data, hue=None, title="Pair Plot", save_path=None):
        """
        Generate a pair plot for the dataframe.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            hue (str, optional): Column name for grouping variable that will produce points with different colors.
            title (str): Title for the pair plot.
            save_path (str): Optional path to save the plot image.
        """
        try:
            pair_plot = sns.pairplot(data, hue=hue, palette="husl")
            pair_plot.fig.suptitle(title, y=1.02)
            if save_path:
                pair_plot.savefig(save_path)
                self.logger.info("Pair plot saved to %s.", save_path)
            plt.show()
            self.logger.info("Pair plot created successfully.")
        except Exception as e:
            self.logger.error("Failed to create pair plot: %s", e)

    def plot_distribution(self, data, column, title="Distribution Plot", save_path=None):
        """
        Generate a distribution plot for a specified column.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            column (str): Column to plot distribution for.
            title (str): Title of the distribution plot.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data[column], shade=True, color="blue")
            plt.title(title)
            plt.xlabel(column)
            plt.ylabel("Density")
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Distribution plot saved to %s.", save_path)
            plt.show()
            self.logger.info("Distribution plot created successfully for column: %s.", column)
        except Exception as e:
            self.logger.error("Failed to create distribution plot for %s: %s", column, e)

    def interactive_scatter_plot(self, data, x, y, color=None, size=None, title="Interactive Scatter Plot"):
        """
        Generate an interactive scatter plot using Plotly.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            color (str, optional): Column name for color encoding.
            size (str, optional): Column name for size encoding.
            title (str): Title of the plot.
        """
        try:
            fig = px.scatter(data, x=x, y=y, color=color, size=size, title=title)
            fig.show()
            self.logger.info("Interactive scatter plot created successfully for %s vs %s.", x, y)
        except Exception as e:
            self.logger.error("Failed to create interactive scatter plot for %s vs %s: %s", x, y, e)

    def interactive_heatmap(self, data, title="Interactive Heatmap"):
        """
        Generate an interactive heatmap using Plotly.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            title (str): Title of the heatmap.
        """
        try:
            fig = px.imshow(data.corr(), text_auto=True, title=title, color_continuous_scale="Viridis")
            fig.show()
            self.logger.info("Interactive heatmap created successfully.")
        except Exception as e:
            self.logger.error("Failed to create interactive heatmap: %s", e)

    def plot_box(self, data, x, y, hue=None, title="Box Plot", save_path=None):
        """
        Generate a box plot for the dataframe with detailed logging.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            hue (str, optional): Column name for grouping variable that will produce points with different colors.
            title (str): Title for the box plot.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=data, x=x, y=y, hue=hue, palette="Set3")
            plt.title(title)
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Box plot saved to %s.", save_path)
            plt.show()
            self.logger.info("Box plot created successfully for %s vs %s.", x, y)
        except Exception as e:
            self.logger.error("Failed to create box plot for %s vs %s: %s", x, y, e)

    def plot_violin(self, data, x, y, hue=None, title="Violin Plot", save_path=None):
        """
        Generate a violin plot for the dataframe with detailed logging.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            hue (str, optional): Column name for grouping variable that will produce points with different colors.
            title (str): Title for the violin plot.
            save_path (str): Optional path to save the plot image.
        """
        try:
            plt.figure(figsize=(12, 6))
            sns.violinplot(data=data, x=x, y=y, hue=hue, palette="muted", split=True)
            plt.title(title)
            if save_path:
                plt.savefig(save_path)
                self.logger.info("Violin plot saved to %s.", save_path)
            plt.show()
            self.logger.info("Violin plot created successfully for %s vs %s.", x, y)
        except Exception as e:
            self.logger.error("Failed to create violin plot for %s vs %s: %s", x, y, e)

    def interactive_box_plot(self, data, x, y, color=None, title="Interactive Box Plot"):
        """
        Generate an interactive box plot using Plotly.
        
        Args:
            data (DataFrame): Pandas DataFrame containing the data.
            x (str): Column name for x-axis.
            y (str): Column name for y-axis.
            color (str, optional): Column name for color encoding.
            title (str): Title of the plot.
        """
        try:
            fig = px.box(data, x=x, y=y, color=color, title=title)
            fig.show()
            self.logger.info("Interactive box plot created successfully for %s vs %s.", x, y)
        except Exception as e:
            self.logger.error("Failed to create interactive box plot for %s vs %s: %s", x, y, e)

if __name__ == "__main__":
    # Example usage of the DataVisualizer class
    import pandas as pd
    import numpy as np

    # Example data
    df = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'C': np.random.choice(['X', 'Y', 'Z'], 100),
        'D': np.random.rand(100) * 10
    })

    logger = logging.getLogger("DataVisualizerExample")
    logging.basicConfig(level=logging.DEBUG)

    visualizer = DataVisualizer(logger=logger)
    visualizer.plot_time_series(df, x='A', y='B', title='Example Time Series')
    visualizer.plot_histogram(df, column='A', title='Example Histogram')
    visualizer.plot_correlation_matrix(df, title='Example Correlation Matrix')
    visualizer.plot_scatter(df, x='A', y='B', hue='C', title='Example Scatter Plot')
    visualizer.plot_pair(df, hue='C', title='Example Pair Plot')
    visualizer.plot_distribution(df, column='D', title='Example Distribution Plot')
    visualizer.interactive_scatter_plot(df, x='A', y='B', color='C', title='Example Interactive Scatter Plot')
    visualizer.interactive_heatmap(df, title='Example Interactive Heatmap')
    visualizer.plot_box(df, x='C', y='D', title='Example Box Plot')
    visualizer.plot_violin(df, x='C', y='D', title='Example Violin Plot')
    visualizer.interactive_box_plot(df, x='C', y='D', title='Example Interactive Box Plot')
