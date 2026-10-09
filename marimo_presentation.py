import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    layout_file="layouts/marimo_presentation.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    from vega_datasets import data

    return alt, data, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Marimo: um notebook python reativo
    """)
    return


@app.cell
def _():
    a = 10
    return (a,)


@app.cell
def _():
    b = 50
    return (b,)


@app.cell
def _(a, b):
    a + b
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # com INTERATIVIDADE
    """)
    return


@app.cell
def _(mo):
    slider_exemplo1 = mo.ui.slider(
        start=0,
        stop=10,
        step=0.5,
        label='Slider de exemplo', value=7)
    slider_exemplo1

    return (slider_exemplo1,)


@app.cell
def _(slider_exemplo1):
    slider_exemplo1.value
    return


@app.cell(hide_code=True)
def _(mo):
    slider_shape = mo.ui.slider(start=0, stop=5, step=0.5, label='Slider do parâmetro shape', value=2.5)
    slider_shape
    return (slider_shape,)


@app.cell(hide_code=True)
def _(mo):
    slider_scale = mo.ui.slider(start=0, stop=5, step=0.5, label='Slider do parâmetro scale', value=2.5)
    slider_scale
    return (slider_scale,)


@app.cell(hide_code=True)
def _(alt, np, pl, slider_scale, slider_shape):
    np.random.seed(42)
    _norm = np.random.gamma(shape=slider_shape.value,scale=slider_scale.value,size=1000)
    _norm_dist = pl.DataFrame(data=_norm,schema=['valor'])
    _norm_dist
    chart = alt.Chart(_norm_dist).mark_bar().encode(
        x=alt.X('valor:Q', bin=True).bin(step=.5),
        y=alt.Y('count()')
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exemplo dataset CARS
    """)
    return


@app.cell
def _(data, pl):
    df_cars = pl.from_pandas(data.cars())
    return (df_cars,)


@app.cell
def _(df_cars):
    df_cars
    return


@app.cell
def _(df_cars, mo):
    origin_selector = mo.ui.dropdown(options=df_cars['Origin'].unique())
    origin_selector
    return (origin_selector,)


@app.cell(hide_code=True)
def _(alt, df_cars, origin_selector, pl):
    # replace _df with your data source

    _df = df_cars

    if origin_selector.selected_key != None:
        _df = df_cars.filter(pl.col('Origin') == origin_selector.selected_key)


    _chart = (
        alt.Chart(_df)
        .mark_point()
        .encode(
            x=alt.X(field='Horsepower', type='quantitative'),
            y=alt.Y(field='Weight_in_lbs', type='quantitative', aggregate='mean'),
            color=alt.Color('Origin'),
            tooltip=[
                alt.Tooltip(field='Horsepower', format=',.2f'),
                alt.Tooltip(field='Weight_in_lbs', aggregate='mean', format=',.0f')
            ]
        )
        .properties(
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': True
                }
            }
        )
    )
    _chart
    return


if __name__ == "__main__":
    app.run()
