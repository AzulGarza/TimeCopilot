from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.agent import AgentRunResult
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tsfeatures import (
    acf_features,
    arch_stat,
    crossing_points,
    entropy,
    flat_spots,
    heterogeneity,
    holt_parameters,
    hurst,
    hw_parameters,
    lumpiness,
    nonlinearity,
    pacf_features,
    series_length,
    stability,
    stl_features,
    unitroot_kpss,
    unitroot_pp,
)
from tsfeatures.tsfeatures import _get_feats

from .models.benchmarks import (
    ADIDA,
    IMAPA,
    AutoARIMA,
    AutoCES,
    AutoETS,
    CrostonClassic,
    DOTheta,
    HistoricAverage,
    SeasonalNaive,
    Theta,
    ZeroModel,
)
from .utils.experiment_handler import ExperimentDataset, ExperimentDatasetParser

MODELS = {
    "ADIDA": ADIDA(),
    "AutoARIMA": AutoARIMA(),
    "AutoCES": AutoCES(),
    "AutoETS": AutoETS(),
    "CrostonClassic": CrostonClassic(),
    "DOTheta": DOTheta(),
    "HistoricAverage": HistoricAverage(),
    "IMAPA": IMAPA(),
    "SeasonalNaive": SeasonalNaive(),
    "Theta": Theta(),
    "ZeroModel": ZeroModel(),
}

TSFEATURES: dict[str, Callable] = {
    "acf_features": acf_features,
    "arch_stat": arch_stat,
    "crossing_points": crossing_points,
    "entropy": entropy,
    "flat_spots": flat_spots,
    "heterogeneity": heterogeneity,
    "holt_parameters": holt_parameters,
    "lumpiness": lumpiness,
    "nonlinearity": nonlinearity,
    "pacf_features": pacf_features,
    "stl_features": stl_features,
    "stability": stability,
    "hw_parameters": hw_parameters,
    "unitroot_kpss": unitroot_kpss,
    "unitroot_pp": unitroot_pp,
    "series_length": series_length,
    "hurst": hurst,
}


class ForecastAgentOutput(BaseModel):
    tsfeatures_results: list[str] = Field(
        description=(
            "The time series features that were considered as a list of strings of "
            "feature names and their values separated by colons."
        )
    )
    tsfeatures_analysis: str = Field(
        description=(
            "Analysis of what the time series features reveal about the data "
            "and their implications for forecasting."
        )
    )
    selected_model: str = Field(
        description="The model that was selected for the forecast"
    )
    model_details: str = Field(
        description=(
            "Technical details about the selected model including its assumptions, "
            "strengths, and typical use cases."
        )
    )
    cross_validation_results: list[str] = Field(
        description=(
            "The cross-validation results as a string of model names "
            "and their scores separated by colons."
        )
    )
    model_comparison: str = Field(
        description=(
            "Detailed comparison of model performances, explaining why certain "
            "models performed better or worse on this specific time series."
        )
    )
    is_better_than_seasonal_naive: bool = Field(
        description="Whether the selected model is better than the seasonal naive model"
    )
    reason_for_selection: str = Field(
        description="Explanation for why the selected model was chosen"
    )
    forecast: list[str] = Field(
        description=(
            "The forecasted values for the time series as a list of strings of "
            "periods and their values separated by colons."
        )
    )
    forecast_analysis: str = Field(
        description=(
            "Detailed interpretation of the forecast, including trends, patterns, "
            "and potential problems."
        )
    )
    user_query_response: str | None = Field(
        description=(
            "The response to the user's query, if any. "
            "If the user did not provide a query, this field will be None."
        )
    )

    def prettify(self, console: Console | None = None) -> None:
        """Pretty print the forecast results using rich formatting."""
        console = console or Console()

        # Create header with title and overview
        header = Panel(
            f"[bold cyan]{self.selected_model}[/bold cyan] forecast analysis\n"
            f"[{'green' if self.is_better_than_seasonal_naive else 'red'}]"
            f"{'✓ Better' if self.is_better_than_seasonal_naive else '✗ Not better'} "
            "than Seasonal Naive[/"
            f"{'green' if self.is_better_than_seasonal_naive else 'red'}]",
            title="[bold blue]TimeCopilot Forecast[/bold blue]",
            style="blue",
        )

        # Time Series Analysis Section
        ts_features = Table(
            title="Time Series Features",
            show_header=True,
            title_style="bold cyan",
            header_style="bold magenta",
        )
        ts_features.add_column("Feature", style="cyan")
        ts_features.add_column("Value", style="magenta")

        # Group features by category for better organization
        for feature in self.tsfeatures_results:
            feature_name, feature_value = feature.split(":")
            ts_features.add_row(
                feature_name.strip(), f"{float(feature_value.strip()):.3f}"
            )

        ts_analysis = Panel(
            f"{self.tsfeatures_analysis}",
            title="[bold cyan]Feature Analysis[/bold cyan]",
            style="blue",
        )

        # Model Selection Section
        model_details = Panel(
            f"[bold]Technical Details[/bold]\n{self.model_details}\n\n"
            f"[bold]Selection Rationale[/bold]\n{self.reason_for_selection}",
            title="[bold green]Model Information[/bold green]",
            style="green",
        )

        # Model Comparison Table
        model_scores = Table(
            title="Model Performance", show_header=True, title_style="bold yellow"
        )
        model_scores.add_column("Model", style="yellow")
        model_scores.add_column("MASE", style="cyan", justify="right")

        # Sort models by performance
        cv_results = []
        for result in self.cross_validation_results:
            model, score = result.split(":")
            cv_results.append((model.strip(), float(score.strip())))

        cv_results.sort(key=lambda x: x[1])  # Sort by score
        for model, score in cv_results:  # type: ignore
            model_scores.add_row(model, f"{score:.2f}")

        model_analysis = Panel(
            self.model_comparison,
            title="[bold yellow]Performance Analysis[/bold yellow]",
            style="yellow",
        )

        # Forecast Results Section
        forecast_table = Table(
            title="Forecast Values", show_header=True, title_style="bold magenta"
        )
        forecast_table.add_column("Period", style="magenta")
        forecast_table.add_column("Value", style="cyan", justify="right")

        # Show all individual values with period indicators
        for fcst in self.forecast:
            period, value = fcst.split(":")
            forecast_table.add_row(f"{period}", f"{value}")

        # Add note about number of periods if many
        if len(self.forecast) > 12:
            forecast_table.caption = (
                f"[dim]Showing all {len(self.forecast)} forecasted periods. "
                "Use aggregation functions for summarized views.[/dim]"
            )

        forecast_analysis = Panel(
            self.forecast_analysis,
            title="[bold magenta]Forecast Analysis[/bold magenta]",
            style="magenta",
        )

        # Optional user response section
        user_response = None
        if self.user_query_response:
            user_response = Panel(
                self.user_query_response,
                title="[bold]Response to Query[/bold]",
                style="cyan",
            )

        # Print all sections with clear separation
        console.print("\n")
        console.print(header)

        console.print("\n[bold]1. Time Series Analysis[/bold]")
        console.print(ts_features)
        console.print(ts_analysis)

        console.print("\n[bold]2. Model Selection[/bold]")
        console.print(model_details)
        console.print(model_scores)
        console.print(model_analysis)

        console.print("\n[bold]3. Forecast Results[/bold]")
        console.print(forecast_table)
        console.print(forecast_analysis)

        if user_response:
            console.print("\n[bold]4. Additional Information[/bold]")
            console.print(user_response)

        console.print("\n")


class TimeCopilot:
    def __init__(
        self,
        llm: str,
        **kwargs: Any,
    ):
        """
        Args:
            llm: The LLM to use.
            **kwargs: Additional keyword arguments to pass to the agent.
        """

        self.system_prompt = f"""
        You're a forecasting expert. You will be given a time series 
        as a list of numbers
        and your task is to determine the best forecasting model for that series. 
        You have access to the following tools:

        1. tsfeatures_tool: Calculates time series features to help 
        with model selection.
        Available features are: {", ".join(TSFEATURES.keys())}

        2. cross_validation_tool: Performs cross-validation for one or more models.
        Takes a list of model names and returns their cross-validation results.
        Available models are: {", ".join(MODELS.keys())}

        3. forecast_tool: Generates forecasts using a selected model.
        Takes a model name and returns forecasted values.

        Your task is to provide a comprehensive analysis following these steps:

        1. Time Series Feature Analysis:
           - Calculate a focused set of key time series features
           - Quickly identify the main characteristics (trend, seasonality, 
                stationarity, etc.)
           - Use these insights to guide efficient model selection
           - Avoid over-analysis - focus on features that directly inform model choice

        2. Model Selection and Evaluation:
           - Start with simple models that can potentially beat seasonal naive
           - Select additional candidate models based on the time series 
                values and features
           - Document each model's technical details and assumptions
           - Explain why these models are suitable for the identified features
           - Perform cross-validation to evaluate performance
           - Compare models quantitatively and qualitatively against seasonal naive
           - If initial models don't beat seasonal naive, try more complex models
           - Prioritize finding a model that outperforms seasonal naive benchmark
           - Balance model complexity with forecast accuracy

        3. Final Model Selection and Forecasting:
           - Choose the best performing model with clear justification
           - Generate and analyze the forecast
           - Interpret trends and patterns in the forecast
           - Discuss reliability and potential uncertainties
           - Address any specific aspects from the user's prompt

        The evaluation will use MASE (Mean Absolute Scaled Error) by default.
        Use at least one cross-validation window for evaluation.
        The seasonality will be inferred from the date column.

        Your output must include:
        - Comprehensive feature analysis with clear implications
        - Detailed model comparison and selection rationale
        - Technical details of the selected model
        - Clear interpretation of cross-validation results
        - Analysis of the forecast and its implications
        - Response to any user queries

        Focus on providing:
        - Clear connections between features and model choices
        - Technical accuracy with accessible explanations
        - Quantitative support for decisions
        - Practical implications of the forecast
        - Thorough responses to user concerns
        """

        if "model" in kwargs:
            raise ValueError(
                "model is not allowed to be passed as a keyword argument"
                "use `llm` instead"
            )

        self.forecasting_agent = Agent(
            deps_type=ExperimentDataset,
            output_type=ForecastAgentOutput,
            system_prompt=self.system_prompt,
            model=llm,
            **kwargs,
        )

        @self.forecasting_agent.system_prompt
        async def add_time_series(ctx: RunContext[ExperimentDataset]) -> str:
            output = (
                f"The time series is: {ctx.deps.df['y'].tolist()}, "
                f"the date column is: {ctx.deps.df['ds'].tolist()}"
            )
            return output

        @self.forecasting_agent.tool
        async def tsfeatures_tool(
            ctx: RunContext[ExperimentDataset],
            features: list[str],
        ) -> str:
            callable_features = []
            for feature in features:
                if feature not in TSFEATURES:
                    raise ModelRetry(
                        f"Feature {feature} is not available. Available features are: "
                        f"{', '.join(TSFEATURES.keys())}"
                    )
                callable_features.append(TSFEATURES[feature])
            features_df = _get_feats(
                index=ctx.deps.df["unique_id"].iloc[0],
                ts=ctx.deps.df,
                features=callable_features,
                freq=ctx.deps.seasonality,
            )
            return ",".join(
                [f"{col}: {features_df[col].iloc[0]}" for col in features_df.columns]
            )

        @self.forecasting_agent.tool
        async def cross_validation_tool(
            ctx: RunContext[ExperimentDataset],
            models: list[str],
        ) -> str:
            models_fcst_cv = None
            callable_models = []
            for str_model in models:
                if str_model not in MODELS:
                    raise ModelRetry(
                        f"Model {str_model} is not available. Available models are: "
                        f"{', '.join(MODELS.keys())}"
                    )
                callable_models.append(MODELS[str_model])
            for model in callable_models:
                fcst_cv = model.cross_validation(
                    df=ctx.deps.df,
                    h=ctx.deps.h,
                    freq=ctx.deps.freq,
                )
                if models_fcst_cv is None:
                    models_fcst_cv = fcst_cv
                else:
                    models_fcst_cv = models_fcst_cv.merge(
                        fcst_cv.drop(columns=["y"]),
                        on=["unique_id", "cutoff", "ds"],
                    )
            eval_df = ctx.deps.evaluate_forecast_df(
                forecast_df=models_fcst_cv,
                models=[model.alias for model in callable_models],
            )
            eval_df = eval_df.groupby(
                ["metric"],
                as_index=False,
            ).mean(numeric_only=True)
            return ", ".join(
                [
                    f"{model.alias}: {eval_df[model.alias].iloc[0]}"
                    for model in callable_models
                ]
            )

        @self.forecasting_agent.tool
        async def forecast_tool(ctx: RunContext[ExperimentDataset], model: str) -> str:
            callable_model = MODELS[model]
            fcst_df = callable_model.forecast(
                df=ctx.deps.df,
                h=ctx.deps.h,
                freq=ctx.deps.freq,
            )
            output = ",".join(
                [
                    f"{row['ds'].strftime('%Y-%m-%d')}: {row[model]}"
                    for _, row in fcst_df.iterrows()
                ]
            )
            return output

        @self.forecasting_agent.output_validator
        async def validate_best_model(
            ctx: RunContext[ExperimentDataset],
            output: ForecastAgentOutput,
        ) -> ForecastAgentOutput:
            if not output.is_better_than_seasonal_naive:
                raise ModelRetry(
                    "The selected model is not better than the seasonal naive model. "
                    "Please try again with a different model."
                    "The cross-validation results are: "
                    "{output.cross_validation_results}"
                )
            return output

    def forecast(
        self,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
    ) -> AgentRunResult[ForecastAgentOutput]:
        """Generate forecast and analysis.

        Args:
            df: The time-series data. Can be one of:

                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer
                it via :pyfunc:`~timecopilot.models.utils.forecaster.get_seasonality`.
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments.

        Returns:
            pydantic_ai.AgentRunResult[~timecopilot.agent.ForecastAgentOutput]:
                A rich result object whose ``output`` attribute is a fully
                populated :class:`~timecopilot.agent.ForecastAgentOutput`
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """
        query = f"User query: {query}" if query else None
        experiment_dataset_parser = ExperimentDatasetParser(
            model=self.forecasting_agent.model,
        )
        dataset = experiment_dataset_parser.parse(
            df,
            freq,
            h,
            seasonality,
            query,
        )
        result = self.forecasting_agent.run_sync(
            user_prompt=query,
            deps=dataset,
        )

        return result
