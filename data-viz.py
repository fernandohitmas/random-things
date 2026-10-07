import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import polars as pl
    import altair as alt
    import numpy as np
    import vega_datasets as data

    return alt, mo, pl


@app.cell
def _(mo):
    file2_ui = mo.ui.file(
        filetypes=[".csv"],
        multiple=False,
        kind="area",
        label="Arraste o arquivo CSV aqui ou clique para selecionar",
    )
    file2_ui
    return (file2_ui,)


@app.cell
def _(file2_ui, pl):
    df_teste = pl.read_csv(file2_ui.contents())
    df_teste = df_teste.filter(pl.col('Valor gasto (BRL)') > 0)
    df_teste = df_teste.with_columns(
        mes=pl.col('Dia').str.strptime(pl.Date, format='%Y-%m-%d').dt.month(),
        dia=pl.col('Dia').str.strptime(pl.Date, format='%Y-%m-%d').dt.day(),
        dia_semana=pl.col('Dia').str.strptime(pl.Date, format='%Y-%m-%d').dt.weekday()
    )
    return (df_teste,)


@app.cell
def _(df_teste):
    df_teste.head()
    return


@app.cell
def _(df_teste):
    df_teste.columns
    return


@app.cell
def _(alt, df_teste):
    chart = (
        alt.Chart(df_teste)
        .encode(
            y=alt.Y('Resultados:Q'),
            x=alt.X('Valor gasto (BRL):Q'),
            color=alt.Color('mes:N').scale(scheme='lightgreyred'),
            # row=alt.Row('mes')
            # size=alt.Size('dia_semana:Q')
        )
        .mark_point(filled=True, opacity=.7, size=100)
        .facet(
            facet=alt.Facet('dia_semana:N'),
            columns=3
        )
    )
    chart.spec.width=250
    chart.spec.height=200
    chart
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
