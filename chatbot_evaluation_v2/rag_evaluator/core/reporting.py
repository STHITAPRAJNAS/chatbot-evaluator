"""
Reporting module for the RAG evaluator.

This module provides utilities for generating reports and visualizations
of evaluation results.
"""

from typing import Dict, Any, List, Optional, Union
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO

class ReportGenerator:
    """Generator for evaluation reports."""
    
    def __init__(self, output_dir: str = None):
        """Initialize report generator.
        
        Args:
            output_dir: Directory to save report files.
        """
        self.output_dir = output_dir or os.path.join(os.getcwd(), "reports")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_summary_plots(
        self,
        summary: Dict[str, Dict[str, float]],
        thresholds: Dict[str, float] = None
    ) -> Dict[str, str]:
        """Generate summary plots for evaluation results.
        
        Args:
            summary: Summary statistics for evaluation results.
            thresholds: Threshold values for metrics.
            
        Returns:
            Dictionary mapping plot names to file paths.
        """
        plot_paths = {}
        
        # Create summary bar chart
        metrics = list(summary.keys())
        means = [summary[m]["mean"] for m in metrics]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics, means, alpha=0.7)
        
        # Add threshold lines if available
        if thresholds:
            for i, metric in enumerate(metrics):
                if metric in thresholds:
                    threshold = thresholds[metric]
                    plt.hlines(
                        threshold,
                        i - 0.4,
                        i + 0.4,
                        colors='red',
                        linestyles='dashed',
                        linewidth=2
                    )
        
        plt.xlabel('Metric')
        plt.ylabel('Mean Score')
        plt.title('Summary of Evaluation Metrics')
        plt.ylim(0, 1.1)  # Assuming scores are between 0 and 1
        plt.tight_layout()
        
        summary_path = os.path.join(self.output_dir, "metrics_summary.png")
        plt.savefig(summary_path)
        plt.close()
        
        plot_paths["summary"] = summary_path
        
        return plot_paths
    
    def generate_distribution_plots(
        self,
        results_df: pd.DataFrame,
        summary: Dict[str, Dict[str, float]],
        thresholds: Dict[str, float] = None
    ) -> Dict[str, str]:
        """Generate distribution plots for evaluation results.
        
        Args:
            results_df: DataFrame of evaluation results.
            summary: Summary statistics for evaluation results.
            thresholds: Threshold values for metrics.
            
        Returns:
            Dictionary mapping plot names to file paths.
        """
        plot_paths = {}
        
        if results_df.empty:
            return plot_paths
        
        # Generate score distribution plots for each metric
        for metric_name in results_df["metric_name"].unique():
            metric_df = results_df[results_df["metric_name"] == metric_name]
            
            plt.figure(figsize=(10, 6))
            sns.histplot(metric_df["score"], bins=10, kde=True)
            
            if metric_name in summary:
                plt.axvline(
                    summary[metric_name]["mean"],
                    color='blue',
                    linestyle='dashed',
                    linewidth=2,
                    label=f'Mean: {summary[metric_name]["mean"]:.2f}'
                )
            
            if thresholds and metric_name in thresholds:
                threshold = thresholds[metric_name]
                plt.axvline(
                    threshold,
                    color='red',
                    linestyle='dashed',
                    linewidth=2,
                    label=f'Threshold: {threshold:.2f}'
                )
            
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {metric_name} Scores')
            plt.legend()
            plt.tight_layout()
            
            plot_path = os.path.join(self.output_dir, f"{metric_name}_distribution.png")
            plt.savefig(plot_path)
            plt.close()
            
            plot_paths[metric_name] = plot_path
        
        return plot_paths
    
    def generate_heatmap(
        self,
        results_df: pd.DataFrame,
        metric_name: str = None
    ) -> Optional[str]:
        """Generate heatmap of sample scores.
        
        Args:
            results_df: DataFrame of evaluation results.
            metric_name: Name of metric to use for heatmap.
            
        Returns:
            Path to the generated heatmap.
        """
        if results_df.empty:
            return None
        
        if metric_name:
            metric_df = results_df[results_df["metric_name"] == metric_name]
        else:
            metric_df = results_df
        
        if "sample_id" not in metric_df.columns or "score" not in metric_df.columns:
            return None
        
        # Pivot the data to create a matrix of sample_id x metric_name
        pivot_df = metric_df.pivot_table(
            index="sample_id",
            columns="metric_name",
            values="score",
            aggfunc="mean"
        )
        
        if pivot_df.empty:
            return None
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Evaluation Scores Heatmap")
        plt.tight_layout()
        
        heatmap_path = os.path.join(self.output_dir, "scores_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        
        return heatmap_path
    
    def generate_html_report(
        self,
        title: str,
        description: str,
        summary: Dict[str, Dict[str, float]],
        threshold_results: Dict[str, bool],
        plot_paths: Dict[str, str],
        results_df: pd.DataFrame = None
    ) -> str:
        """Generate HTML report.
        
        Args:
            title: Report title.
            description: Report description.
            summary: Summary statistics for evaluation results.
            threshold_results: Results of threshold checks.
            plot_paths: Paths to generated plots.
            results_df: DataFrame of evaluation results.
            
        Returns:
            Path to the generated HTML report.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Convert images to base64 for embedding in HTML
        embedded_images = {}
        for name, path in plot_paths.items():
            if os.path.exists(path):
                with open(path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    embedded_images[name] = f"data:image/png;base64,{img_data}"
        
        # Simple HTML report template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
                .visualization {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>{description}</p>
            
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Std Dev</th>
                    <th>Status</th>
                </tr>
        """
        
        # Add summary rows
        for metric_name, stats in summary.items():
            threshold_status = threshold_results.get(metric_name, False)
            status_class = "pass" if threshold_status else "fail"
            status_text = "PASS" if threshold_status else "FAIL"
            
            html_content += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{stats["mean"]:.4f}</td>
                    <td>{stats["median"]:.4f}</td>
                    <td>{stats["min"]:.4f}</td>
                    <td>{stats["max"]:.4f}</td>
                    <td>{stats["std"]:.4f}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Visualizations</h2>
        """
        
        # Add summary visualization
        if "summary" in embedded_images:
            html_content += f"""
            <div class="visualization">
                <h3>Metrics Summary</h3>
                <img src="{embedded_images['summary']}" alt="Metrics summary" width="800">
            </div>
            """
        
        # Add distribution visualizations
        for metric_name in summary.keys():
            if metric_name in embedded_images:
                html_content += f"""
                <div class="visualization">
                    <h3>{metric_name} Distribution</h3>
                    <img src="{embedded_images[metric_name]}" alt="{metric_name} distribution" width="800">
                </div>
                """
        
        # Add heatmap if available
        if "heatmap" in embedded_images:
            html_content += f"""
            <div class="visualization">
                <h3>Scores Heatmap</h3>
                <img src="{embedded_images['heatmap']}" alt="Scores heatmap" width="800">
            </div>
            """
        
        # Add detailed results table if available
        if results_df is not None and not results_df.empty:
            html_content += """
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Sample ID</th>
                    <th>Metric</th>
                    <th>Score</th>
                </tr>
            """
            
            for _, row in results_df.iterrows():
                html_content += f"""
                <tr>
                    <td>{row.get('sample_id', '')}</td>
                    <td>{row.get('metric_name', '')}</td>
                    <td>{row.get('score', 0.0):.4f}</td>
                </tr>
                """
            
            html_content += """
            </table>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
