import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import streamlit as st


class SmartVisualizer:
    """Intelligent chart generator for data analysis"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize visualizer with dataframe"""
        self.df = df
        
    def auto_visualize(self, chart_type: str = "auto") -> Optional[go.Figure]:
        """Automatically create appropriate visualization"""
        if chart_type == "auto":
            chart_type = self._suggest_chart_type()
        
        try:
            if chart_type == "bar":
                return self._create_bar_chart()
            elif chart_type == "line":
                return self._create_line_chart()
            elif chart_type == "scatter":
                return self._create_scatter_plot()
            elif chart_type == "histogram":
                return self._create_histogram()
            else:
                return self._create_bar_chart()
        except Exception as e:
            st.error(f"Visualization error: {e}")
            return None
    
    def _suggest_chart_type(self) -> str:
        """Suggest best chart type based on data"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        date_cols = self.df.select_dtypes(include=['datetime']).columns
        
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            return "line"
        elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
            if self.df[categorical_cols[0]].nunique() < 10:
                return "bar"
            else:
                return "scatter"
        elif len(numeric_cols) >= 2:
            return "scatter"
        elif len(numeric_cols) == 1:
            return "histogram"
        else:
            return "bar"
    
    def _create_bar_chart(self) -> go.Figure:
        """Create bar chart"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return None
        
        x_col = categorical_cols[0]
        y_col = numeric_cols[0]
        
        if len(self.df) > 50:
            df_agg = self.df.groupby(x_col)[y_col].mean().reset_index()
        else:
            df_agg = self.df
        
        fig = px.bar(
            df_agg.head(20),
            x=x_col,
            y=y_col,
            title=f"{y_col} by {x_col}",
            template="plotly_dark"
        )
        
        fig.update_layout(height=500, showlegend=False)
        return fig
    
    def _create_line_chart(self) -> go.Figure:
        """Create line chart for time series"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        date_cols = self.df.select_dtypes(include=['datetime']).columns
        
        if len(date_cols) == 0:
            for col in self.df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])
                        date_cols = [col]
                        break
                    except:
                        pass
        
        if len(date_cols) == 0 or len(numeric_cols) == 0:
            return self._create_bar_chart()
        
        x_col = date_cols[0]
        y_col = numeric_cols[0]
        
        fig = px.line(
            self.df.sort_values(x_col),
            x=x_col,
            y=y_col,
            title=f"{y_col} over time",
            template="plotly_dark"
        )
        
        fig.update_layout(height=500)
        return fig
    
    def _create_scatter_plot(self) -> go.Figure:
        """Create scatter plot"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            return self._create_histogram()
        
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        color_col = categorical_cols[0] if len(categorical_cols) > 0 else None
        
        fig = px.scatter(
            self.df.sample(min(1000, len(self.df))),
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"{y_col} vs {x_col}",
            template="plotly_dark"
        )
        
        fig.update_layout(height=500)
        return fig
    
    def _create_histogram(self) -> go.Figure:
        """Create histogram for distribution"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            return None
        
        col = numeric_cols[0]
        
        fig = px.histogram(
            self.df,
            x=col,
            title=f"Distribution of {col}",
            template="plotly_dark",
            nbins=30
        )
        
        fig.update_layout(height=500)
        return fig