"""
Reusable UI components for the Dash web application.
"""
from dash import dcc, html


def make_slider(
    component_id,
    min_val,
    max_val,
    step_val,
    default_val,
    marks=None,
    tooltip_always_visible=True,
):
    """
    Create a labeled slider with common styling.

    Args:
        component_id: Unique ID for the slider
        min_val: Minimum value
        max_val: Maximum value
        step_val: Step size
        default_val: Default value
        marks: Optional dict of specific marks {value: label}
        tooltip_always_visible: Whether to always show tooltip

    Returns:
        Dash HTML component
    """
    if marks is None:
        marks = {}

    return html.Div(
        [
            dcc.Slider(
                id=component_id,
                min=min_val,
                max=max_val,
                step=step_val,
                value=default_val,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": tooltip_always_visible},
            )
        ]
    )


def make_checklist(options, default_values, inline=True):
    """
    Create a checklist with common styling.

    Args:
        options: List of (label, value) tuples
        default_values: List of default checked values
        inline: Whether to display inline

    Returns:
        Dash Checklist component
    """
    return dcc.Checklist(
        options=[{"label": f" {label}", "value": value} for label, value in options],
        value=default_values,
        inline=inline,
    )


def make_number_input(
    component_id, value, min_val=0, step_val=1, style=None
):
    """
    Create a number input field.

    Args:
        component_id: Unique ID for the input
        value: Default value
        min_val: Minimum value
        step_val: Step size
        style: Optional additional CSS styles

    Returns:
        Dash Input component
    """
    default_style = {"width": "80px"}
    if style:
        default_style.update(style)

    return dcc.Input(
        id=component_id,
        type="number",
        value=value,
        min=min_val,
        step=step_val,
        style=default_style,
    )
