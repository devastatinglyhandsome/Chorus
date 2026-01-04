# Interactive dashboard generator for benchmark results

import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from chorus.benchmarks.runner import BenchmarkMetrics


class DashboardGenerator:
    def __init__(self, results_path: str):
        with open(results_path) as f:
            data = json.load(f)

        self.results = [BenchmarkMetrics(**r) for r in data]

    def generate_dashboard(self, output_path: str = "dashboard.html"):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Latency Distribution",
                "Cost Analysis",
                "Individual Model Latencies",
                "Parallel vs Total Latency"
            )
        )

        total_latencies = [r.total_latency_ms for r in self.results]
        parallel_latencies = [r.parallel_latency_ms for r in self.results]

        fig.add_trace(
            go.Histogram(x=total_latencies, name="Total Latency", opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=parallel_latencies, name="Parallel Latency", opacity=0.7),
            row=1, col=1
        )

        costs = [r.cost_estimate_usd for r in self.results]
        fig.add_trace(
            go.Box(y=costs, name="Cost per Query"),
            row=1, col=2
        )

        if len(self.results) > 0 and len(self.results[0].individual_latencies) > 0:
            for i in range(len(self.results[0].individual_latencies)):
                latencies = [r.individual_latencies[i] for r in self.results]
                fig.add_trace(
                    go.Box(y=latencies, name=f"Model {i+1}"),
                    row=2, col=1
                )

        fig.add_trace(
            go.Scatter(
                x=parallel_latencies,
                y=total_latencies,
                mode='markers',
                name="Queries"
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Chorus LLM Performance Dashboard",
            showlegend=True,
            height=800,
        )

        fig.write_html(output_path)
        print(f"Dashboard saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate dashboard")
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--output", default="dashboard.html", help="Output HTML file")

    args = parser.parse_args()

    dashboard = DashboardGenerator(args.results)
    dashboard.generate_dashboard(args.output)


if __name__ == "__main__":
    main()

