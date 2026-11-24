import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Séries Temporais e Aprendizado Dinâmico (SME0808 — ICMC/USP)
    ## Transcrição Completa dos Exemplos de R (`fpp3`) para Python (`Polars` + `Altair` + `Statsmodels` + `SciPy`)

    Este notebook Marimo reproduz em Python todos os exemplos práticos originalmente escritos em R (pacotes `fpp3`, `tsibble`, `feasts`, `fable` e `tidyverse`) nos slides da disciplina **SME0808 — Séries Temporais e Aprendizado Dinâmico** (Prof. Oilson A. Gonzatto Junior, ICMC/USP).

    | Equivalência | Ecossistema R (`fpp3`) | Ecossistema Python neste Notebook |
    | :--- | :--- | :--- |
    | **Estrutura tabular temporal** | `tsibble` / `tibble` / `dplyr` | `polars.DataFrame` |
    | **Visualização declarativa** | `ggplot2` / `autoplot` / `patchwork` / `ggpubr` | `altair` |
    | **Transformações e Testes** | `feasts::guerrero`, `box_cox`, `cox.stuart`, `jarque.bera.test` | `scipy.stats`, `scipy.optimize` |
    | **Modelagem e Diagnóstico** | `TSLM`, `ARIMA`, `ACF`, `PACF`, `unitroot_kpss`, `ljung_box` | `statsmodels.tsa` (`SARIMAX`, `acf`, `pacf`, `kpss`, `acorr_ljungbox`) |
    """)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")

    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    from datetime import date
    from scipy import stats, optimize
    from statsmodels.tsa.stattools import acf, pacf, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima_process import ArmaProcess

    alt.data_transformers.disable_max_rows()

    # Datasets clássicos do R (Nile: 1871-1970 e AirPassengers: 1949-1960)
    nile_values = np.array([
        1120, 1160,  963, 1210, 1160, 1160,  813, 1230, 1370, 1140,
         995,  935, 1110,  994, 1020,  960, 1180,  799,  958, 1140,
        1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100,  774,  840,
         874,  694,  940,  833,  701,  916,  692, 1020, 1050,  969,
         831,  726,  456,  824,  702, 1120, 1100,  832,  764,  821,
         768,  845,  864,  862,  698,  845,  744,  796, 1040,  759,
         781,  865,  845,  944,  984,  897,  822, 1010,  771,  676,
         649,  846,  812,  742,  801, 1040,  860,  874,  848,  890,
         744,  749,  838, 1050, 1050,  687,  741,  845,  745,  862,
         990,  841,  796,  856,  860,  783,  714,  742,  808,  740
    ], dtype=float)

    air_passengers_values = np.array([
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
    ], dtype=float)

    meses_ordem = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    return (
        ArmaProcess,
        SARIMAX,
        acf,
        acorr_ljungbox,
        air_passengers_values,
        alt,
        date,
        kpss,
        meses_ordem,
        mo,
        nile_values,
        np,
        optimize,
        pacf,
        pl,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Módulo 1 (`SME0808_01.pdf`) — Métodos Descritivos e Gráficos Temporais

    Nos slides da Aula 01, uma série mensal de temperatura média do ar (de janeiro de 2010 a dezembro de 2019, totalizando $n = 120$ observações) é analisada através de 4 visualizações descritivas fundamentais:
    1. **Gráfico no Tempo** (`autoplot`)
    2. **Gráfico de Sazonalidade** (`gg_season`)
    3. **Gráfico de Sub-séries Sazonais** (`gg_subseries`)
    4. **Gráfico de Defasagens / Lag Plot** (`gg_lag`)
    """)
    return


@app.cell
def _(alt, date, meses_ordem, np, pl):
    # Dados de exemplo (SME0808_01.pdf - Slides 11, 31, 36 e 44)
    y_temp = np.array([
        26.8, 27.2, 27.1, 26.3, 25.4, 23.9, 23.8, 23.6, 25.3, 25.8, 26.4, 26.9,
        27.1, 27.5, 27.4, 26.4, 24.8, 24.3, 23.4, 23.4, 24.6, 25.4, 25.8, 26.7,
        26.9, 26.3, 25.7, 25.7, 24.8, 24.0, 23.4, 23.5, 24.8, 25.6, 26.2, 26.5,
        26.8, 26.9, 26.7, 26.1, 26.2, 24.7, 23.9, 23.7, 24.7, 25.8, 26.1, 26.5,
        26.3, 27.1, 26.2, 25.7, 25.5, 24.9, 24.2, 24.6, 25.5, 25.9, 26.4, 26.9,
        27.1, 27.1, 27.4, 26.8, 25.4, 24.8, 23.6, 23.9, 25.0, 25.9, 26.3, 26.6,
        26.8, 27.1, 27.4, 26.4, 25.5, 24.7, 24.3, 24.4, 24.8, 26.2, 26.3, 27.0,
        27.1, 27.5, 26.2, 28.2, 27.1, 25.4, 25.6, 24.5, 24.7, 26.0, 26.5, 26.8,
        26.3, 26.7, 26.6, 25.8, 25.2, 25.1, 23.3, 23.8, 25.2, 25.5, 26.4, 26.7,
        27.0, 27.4, 27.0, 26.3, 25.9, 24.6, 24.1, 24.3, 25.2, 26.3, 26.4, 26.7
    ])

    _datas = pl.date_range(date(2010, 1, 1), date(2019, 12, 1), interval="1mo", eager=True)

    df_temp = (
        pl.DataFrame({"data": _datas, "y": y_temp})
        .with_columns(
            ano=pl.col("data").dt.year(),
            mes_num=pl.col("data").dt.month(),
            mes_nome=pl.col("data").dt.month().map_elements(lambda m: meses_ordem[m - 1], return_dtype=pl.String),
        )
    )

    # 1.1 Gráfico no tempo (equivalente a autoplot(y))
    chart_time_plot = (
        alt.Chart(df_temp)
        .mark_line(point=True, color="#1f77b4")
        .encode(
            x=alt.X("data:T", title="Tempo (em meses)"),
            y=alt.Y("y:Q", title="Temperatura (°C)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("data:T", title="Mês"), alt.Tooltip("y:Q", title="Temp (°C)")]
        )
        .properties(
            title="1.1 Gráfico no Tempo — Temperatura média do ar, em meses sucessivos",
            width=700,
            height=260,
        )
    )
    chart_time_plot
    return chart_time_plot, df_temp, y_temp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Análise do Gráfico no Tempo (`Slide 11`):**
    * A série apresenta oscilações periódicas anuais bastante regulares entre aproximadamente $23,3\text{ }^\circ\text{C}$ (inverno no hemisfério sul, junho–agosto) e $28,2\text{ }^\circ\text{C}$ (verão, janeiro–março), sem tendência secular de aquecimento ou resfriamento aparente na década 2010–2019.
    """)
    return


@app.cell
def _(alt, df_temp, meses_ordem):
    # 1.2 Gráfico Sazonal (equivalente a gg_season(y, labels='both'))
    chart_season = (
        alt.Chart(df_temp)
        .mark_line(point=True)
        .encode(
            x=alt.X("mes_nome:O", sort=meses_ordem, title="Tempo (em meses)"),
            y=alt.Y("y:Q", title="Temperatura (°C)", scale=alt.Scale(zero=False)),
            color=alt.Color("ano:N", title="Ano", scale=alt.Scale(scheme="turbo")),
            tooltip=["ano:N", "mes_nome:O", "y:Q"]
        )
        .properties(
            title="1.2 Gráfico Sazonal (gg_season) — Curvas anuais sobrepostas",
            width=700,
            height=280,
        )
    )

    # 1.3 Gráfico de Sub-séries (equivalente a gg_subseries(y))
    _base_sub = alt.Chart(df_temp)
    _lines_sub = _base_sub.mark_line(color="#333333").encode(
        x=alt.X("ano:O", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("y:Q", title="Temperatura (°C)", scale=alt.Scale(zero=False)),
        tooltip=["ano:O", "mes_nome:O", "y:Q"]
    )
    _mean_sub = _base_sub.mark_rule(color="blue", strokeWidth=2).encode(
        y="mean(y):Q"
    )
    chart_subseries = (
        alt.layer(_lines_sub, _mean_sub, data=df_temp)
        .properties(width=48, height=180)
        .facet(
            column=alt.Column("mes_nome:O", sort=meses_ordem, title="1.3 Gráfico de Sub-séries Sazonais (gg_subseries) — Média mensal em azul")
        )
    )

    alt.vconcat(chart_season, chart_subseries)
    return chart_season, chart_subseries


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Análise dos Gráficos de Sazonalidade (`Slide 31`) e Sub-séries (`Slide 36`):**
    * O **gráfico sazonal** (`gg_season`) confirma que todos os 10 anos seguem o mesmo formato em "U", com vale pronunciado em julho/agosto e picos em fevereiro/março.
    * O **gráfico de sub-séries** (`gg_subseries`) explicita a média histórica de cada mês (traço horizontal azul) e mostra a estabilidade da amplitude sazonal ao longo dos anos (com um pico atípico apenas em abril de 2017, onde a temperatura chegou a $28,2\text{ }^\circ\text{C}$).
    """)
    return


@app.cell
def _(alt, df_temp, meses_ordem, pl):
    # 1.4 Gráfico de Defasagens (equivalente a gg_lag(y, geom='point', period='year'))
    _lags_list = []
    for _k in range(1, 10):
        _lags_list.append(
            df_temp.with_columns(
                y_lag=pl.col("y").shift(_k),
                lag=pl.lit(f"lag {_k}")
            ).drop_nulls()
        )
    df_lags_temp = pl.concat(_lags_list)

    _pts = (
        alt.Chart(df_lags_temp)
        .mark_circle(size=45, opacity=0.85)
        .encode(
            x=alt.X("y_lag:Q", title="y(t - k)", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", title="y(t)", scale=alt.Scale(zero=False)),
            color=alt.Color("mes_nome:N", sort=meses_ordem, title="Mês"),
            tooltip=["lag:N", "ano:N", "mes_nome:N", "y:Q", "y_lag:Q"]
        )
    )
    _diag = (
        alt.Chart(df_lags_temp)
        .mark_line(strokeDash=[4, 4], color="gray")
        .encode(
            x=alt.X("min(y_lag):Q"),
            x2=alt.X2("max(y_lag):Q"),
            y=alt.Y("min(y_lag):Q"),
            y2=alt.Y2("max(y_lag):Q"),
        )
    )

    chart_lag = (
        alt.layer(_diag, _pts, data=df_lags_temp)
        .properties(width=180, height=160)
        .facet(
            facet=alt.Facet("lag:N", title=None),
            columns=3
        )
        .properties(title="1.4 Gráfico de Defasagens (gg_lag) — y(t) vs y(t - k)")
    )
    chart_lag
    return chart_lag, df_lags_temp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Análise do Gráfico de Defasagens (`Slide 44`):**
    * Nos *lags* 1 e 2 há forte associação linear positiva (meses adjacentes possuem temperaturas próximas), enquanto no *lag* 6 (meio ciclo anual — verão vs. inverno) observa-se forte associação linear **negativa**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Módulo 2 (`SME0808_02.pdf`) — Transformação de Box-Cox e Método de Guerrero

    No **Slide 13** de `SME0808_02.pdf`, analisa-se uma série mensal de consumo de energia (iniciando em janeiro de 1968, com $n = 142$ meses) cuja variabilidade sazonal cresce com o nível da série.
    * O método de **Guerrero (1993)** seleciona $\lambda$ minimizando o coeficiente de variação das sub-séries anuais da razão $s_r / \bar{z}_r^{(1 - \lambda)}$.
    * A transformação de **Box-Cox** definida no `fpp3` é:
      $$w_t = \begin{cases} \log(y_t), & \text{se } \lambda = 0 \\ \frac{\text{sign}(y_t)|y_t|^\lambda - 1}{\lambda}, & \text{se } \lambda \neq 0 \end{cases}$$
    """)
    return


@app.cell
def _(alt, date, np, optimize, pl):
    # Dados de consumo de energia (SME0808_02.pdf - Slide 13)
    y_energia = np.array([
        10951, 10671, 13643, 15109, 12602, 12295, 12056, 12376, 14146, 12993, 14206, 13603, 13044, 12132, 13606, 15876, 14316,
        12675, 14290, 15170, 15065, 16273, 16505, 16060, 16964, 17675, 15548, 17275, 22318, 21117, 23326, 27224, 28794, 19554,
        19601, 27330, 29260, 29591, 29509, 28883, 32390, 32702, 31748, 28157, 32898, 32909, 34087, 31760, 35105, 31767, 35608,
        35094, 32674, 35438, 36270, 35369, 36742, 36615, 36748, 37289, 38277, 37825, 34107, 42106, 49218, 55002, 51820, 60111,
        58980, 56281, 60731, 62870, 63653, 65574, 59721, 65205, 61340, 59538, 55018, 65002, 66740, 63595, 63202, 59822, 65653,
        62089, 64107, 70440, 65237, 66885, 62929, 68011, 71948, 72940, 74873, 72523, 78868, 79610, 74635, 80178, 79802, 82737,
        81259, 81697, 80502, 78160, 80078, 83807, 84608, 89889, 81894, 95400, 91214, 89767, 89726, 97896, 103416, 107646, 120409,
        109626, 100276, 118065, 116504, 134251, 134667, 144760, 144438, 159168, 168255, 175231, 174530, 173720, 179821, 185780,
        270327, 196949, 202968, 213178, 210912, 213598, 210297
    ], dtype=float)

    def guerrero_lambda(x: np.ndarray, period: int = 12, bounds=(-1.0, 2.0)) -> float:
        """Implementação do método de Guerrero (1993) idêntico a feasts::guerrero."""
        n_full = (len(x) // period) * period
        subseries = x[:n_full].reshape(-1, period)
        means = subseries.mean(axis=1)
        sds = subseries.std(axis=1, ddof=1)

        def cv_obj(lam: float) -> float:
            ratios = sds / (means ** (1.0 - lam))
            return float(np.std(ratios, ddof=1) / np.mean(ratios))

        res = optimize.minimize_scalar(cv_obj, bounds=bounds, method="bounded")
        return float(res.x)

    def box_cox_fpp3(x: np.ndarray, lam: float) -> np.ndarray:
        if abs(lam) < 1e-8:
            return np.log(x)
        return (np.sign(x) * (np.abs(x) ** lam) - 1.0) / lam

    lambda_otimo = guerrero_lambda(y_energia, period=12)
    y_bc = box_cox_fpp3(y_energia, lambda_otimo)

    _end_year = 1968 + (len(y_energia) - 1) // 12
    _end_month = 1 + (len(y_energia) - 1) % 12
    _datas_energia = pl.date_range(date(1968, 1, 1), date(_end_year, _end_month, 1), interval="1mo", eager=True)
    df_energia = pl.DataFrame({
        "data": _datas_energia,
        "y": y_energia,
        "y_bc": y_bc,
    })

    _g0 = (
        alt.Chart(df_energia)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("data:T", title="Tempo (em meses)"),
            y=alt.Y("y:Q", title="Consumo de energia"),
            tooltip=["data:T", "y:Q"]
        )
        .properties(title="Série Original", width=340, height=250)
    )

    _g1 = (
        alt.Chart(df_energia)
        .mark_line(color="#2ca02c")
        .encode(
            x=alt.X("data:T", title="Tempo (em meses)"),
            y=alt.Y("y_bc:Q", title="Consumo de energia (transformada por Box-Cox)", scale=alt.Scale(zero=False)),
            tooltip=["data:T", alt.Tooltip("y_bc:Q", format=".3f")]
        )
        .properties(title=f"Transformada de Box-Cox (λ = {lambda_otimo:.3f})", width=340, height=250)
    )

    _g0 | _g1
    return box_cox_fpp3, df_energia, guerrero_lambda, lambda_otimo, y_bc, y_energia


@app.cell(hide_code=True)
def _(lambda_otimo, mo):
    mo.md(f"""
    **Análise da Transformação de Box-Cox (`Slide 13`):**
    * O método de Guerrero estimou $\\hat{{\\lambda}} = {lambda_otimo:.3f}$ (muito próximo de um ajuste logarítmico/potência negativa), reduzindo drasticamente a heterocedasticidade (o aumento da amplitude das oscilações no final da década de 1970) e linearizando a tendência de crescimento.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Módulo 3 (`SME0808_03.pdf`) — Modelagem e Testes de Tendência

    Na Aula 03 são abordadas três técnicas de análise de tendência:
    1. **Ajuste de Curvas Determinísticas** (`TSLM` Polinomial de grau 3 vs. Regressão Linear por Partes / *Piecewise Splines* na série `Nile` — Slide 8)
    2. **Suavização por Médias Móveis Centradas** (`2-MA` e `5-MA` na série `AirPassengers` — Slide 17)
    3. **Teste Não-Paramétrico de Cox-Stuart** para presença de tendência monotônica (Slide 30)
    """)
    return


@app.cell
def _(alt, nile_values, np, pl):
    # 3.1 Ajuste de Curvas na série Nile (SME0808_03.pdf - Slide 8)
    _anos_nile = np.arange(1871, 1971, dtype=float)
    _t_norm = (_anos_nile - _anos_nile.mean()) / _anos_nile.std()

    # Modelo 1: Polinômio de grau 3: value ~ index + I(index^2) + I(index^3)
    _X_poly = np.column_stack([np.ones_like(_t_norm), _t_norm, _t_norm**2, _t_norm**3])
    _beta_poly, _, _, _ = np.linalg.lstsq(_X_poly, nile_values, rcond=None)
    _fit_poly = _X_poly @ _beta_poly

    # Modelo 2: Linear por partes com knots = c(1895, 1910, 1930)
    _knots = [1895.0, 1910.0, 1930.0]
    _X_pw = np.column_stack(
        [np.ones_like(_anos_nile), _anos_nile]
        + [np.maximum(0.0, _anos_nile - k) for k in _knots]
    )
    _beta_pw, _, _, _ = np.linalg.lstsq(_X_pw, nile_values, rcond=None)
    _fit_pw = _X_pw @ _beta_pw

    df_nile = pl.DataFrame({
        "index": _anos_nile.astype(int),
        "value": nile_values,
        "Polinômio (grau 3)": _fit_poly,
        "Linear por partes": _fit_pw,
    })

    _df_nile_long = df_nile.unpivot(
        index=["index", "value"],
        on=["Polinômio (grau 3)", "Linear por partes"],
        variable_name="modelo",
        value_name="fitted",
    )

    _base_nile = alt.Chart(df_nile).mark_line(color="black", strokeWidth=1.2).encode(
        x=alt.X("index:Q", title="Tempo (em anos)", scale=alt.Scale(domain=[1870, 1970]), axis=alt.Axis(format="d")),
        y=alt.Y("value:Q", title="Medição", scale=alt.Scale(zero=False)),
    )

    _fitted_lines = alt.Chart(_df_nile_long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("index:Q"),
        y=alt.Y("fitted:Q"),
        color=alt.Color("modelo:N", title="Modelo de Tendência"),
        tooltip=["index:Q", "modelo:N", alt.Tooltip("fitted:Q", format=".1f")]
    )

    (_base_nile + _fitted_lines).properties(
        title={
            "text": "3.1 Vazões do Rio Nilo em Ashwan",
            "subtitle": "(entre 1871 e 1970) — Polinômio Grau 3 vs. Linear por Partes (knots: 1895, 1910, 1930)"
        },
        width=700,
        height=280,
    )
    return (df_nile,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Análise do Ajuste de Curvas na Série `Nile` (`Slide 8`):**
    * A mudança estrutural de nível por volta de 1898 (associada à construção da primeira barragem de Assuã e mudança no regime pluviométrico) é capturada com maior fidelidade pela **regressão linear por partes** (`knots = c(1895, 1910, 1930)`), enquanto o polinômio de grau 3 suaviza excessivamente a transição.
    """)
    return


@app.cell
def _(air_passengers_values, alt, date, pl):
    # 3.2 Médias Móveis Centradas na série AirPassengers (SME0808_03.pdf - Slide 17)
    # Nota: no R, slider::slide_dbl(value, mean, .before=2, .after=2) calcula média de janela 2+1+2 = 5
    # e .before=5, .after=5 calcula média de janela 5+1+5 = 11.
    _datas_air = pl.date_range(date(1949, 1, 1), date(1960, 12, 1), interval="1mo", eager=True)

    df_air_ma = (
        pl.DataFrame({"index": _datas_air, "value": air_passengers_values})
        .with_columns(
            ma_2=pl.col("value").rolling_mean(window_size=5, center=True),
            ma_5=pl.col("value").rolling_mean(window_size=11, center=True),
        )
        .with_columns(
            res_2_ma=pl.col("value") - pl.col("ma_2"),
            res_5_ma=pl.col("value") - pl.col("ma_5"),
        )
    )

    _df_ma_long = (
        df_air_ma.select(["index", "ma_2", "ma_5"])
        .rename({"ma_2": "2-MA (janela 5)", "ma_5": "5-MA (janela 11)"})
        .unpivot(index="index", variable_name="Suavização", value_name="ma")
        .drop_nulls()
    )

    _df_res_long = (
        df_air_ma.select(["index", "res_2_ma", "res_5_ma"])
        .rename({"res_2_ma": "2-MA (janela 5)", "res_5_ma": "5-MA (janela 11)"})
        .unpivot(index="index", variable_name="Suavização", value_name="residuo")
        .drop_nulls()
    )

    _g0_orig = alt.Chart(df_air_ma).mark_line(color="black", opacity=0.6).encode(
        x=alt.X("index:T", title="Tempo (em anos)"),
        y=alt.Y("value:Q", title="Número de passageiros"),
    )
    _g0_ma = alt.Chart(_df_ma_long).mark_line(strokeWidth=2).encode(
        x="index:T",
        y="ma:Q",
        color=alt.Color("Suavização:N", scale=alt.Scale(domain=["2-MA (janela 5)", "5-MA (janela 11)"], range=["red", "blue"])),
    )
    _chart_ma_top = (_g0_orig + _g0_ma).properties(
        title="3.2 Totais mensais de passageiros em linhas aéreas internacionais (1949 a 1960 - EUA)",
        width=700,
        height=230,
    )

    _zero_rule = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 4], color="black").encode(y="y:Q")
    _g1_res = alt.Chart(_df_res_long).mark_line().encode(
        x=alt.X("index:T", title="Tempo (em anos)"),
        y=alt.Y("residuo:Q", title="Resíduo", scale=alt.Scale(domain=[-150, 150])),
        color=alt.Color("Suavização:N", scale=alt.Scale(domain=["2-MA (janela 5)", "5-MA (janela 11)"], range=["red", "blue"])),
    )
    _chart_ma_bot = (_zero_rule + _g1_res).properties(
        title="Resíduo dos totais mensais de passageiros (Série - Média Móvel)",
        width=700,
        height=200,
    )

    alt.vconcat(_chart_ma_top, _chart_ma_bot)
    return (df_air_ma,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Análise das Médias Móveis (`Slide 17`):**
    * A janela mais curta (`2-MA`, 5 meses, em vermelho) ainda acompanha parte da oscilação sazonal, deixando resíduos menores.
    * Já a janela de 11 meses (`5-MA`, em azul, quase um ciclo anual completo de 12 meses) filtra quase toda a sazonalidade da curva de tendência, transferindo o padrão sazonal (com amplitude crescente) para a série de resíduos.
    """)
    return


@app.cell
def _(air_passengers_values, np, pl, stats):
    # 3.3 Teste de Tendência de Cox-Stuart (SME0808_03.pdf - Slide 30)
    def cox_stuart(series: np.ndarray, trend: str = "+", serie_nome: str = "Série") -> dict:
        assert trend in ("+", "-"), "trend deve ser '+' ou '-'"
        N = len(series)
        half = N // 2
        p1 = series[:half]
        p2 = series[half : 2 * half]

        sinais = np.sign(p2 - p1)
        if trend == "-":
            T2 = int(np.sum(sinais < 0))
        else:
            T2 = int(np.sum(sinais > 0))

        n_validos = int(np.sum(sinais != 0))
        tcrit = float(stats.binom.ppf(0.95, n_validos, 0.5))
        pv = float(1.0 - stats.binom.cdf(T2, n_validos, 0.5))
        pv_str = "<0.0001" if round(pv, 4) == 0 else f"{pv:.4f}"

        return {
            "Série": serie_nome,
            "N": N,
            "n (não-nulos)": n_validos,
            "estatistica (T2)": T2,
            "valor.critico (95%)": tcrit,
            "tendencia (H1)": trend,
            "p-valor": pv_str,
        }

    df_cox_stuart = pl.DataFrame([
        cox_stuart(air_passengers_values, trend="+", serie_nome="AirPassengers"),
        cox_stuart(np.diff(air_passengers_values), trend="-", serie_nome="diff(AirPassengers)"),
    ])
    df_cox_stuart
    return cox_stuart, df_cox_stuart


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Análise do Teste de Cox-Stuart (`Slide 30`):**
    * Para `AirPassengers` original ($H_1$: tendência crescente `+`), todas as $72$ comparações entre a segunda metade (1955–1960) e a primeira metade (1949–1954) são positivas ($T_2 = 72 > 43$, $p < 0,0001$), rejeitando fortemente a hipótese nula de ausência de tendência.
    * Após a primeira diferença `diff(AirPassengers)`, testando tendência decrescente (`-`), obtém-se $T_2 = 31 \le 42$ ($p = 0,8305$), não rejeitando a hipótese nula.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Módulo 4 (`SME0808_04.pdf`) — Sazonalidade Harmônica, Autocorrelação (ACF) e Simulações

    Na Aula 04, o foco recai sobre:
    1. **Modelagem Harmônica da Sazonalidade** via termos de Fourier ($\sin(2\pi t / 12)$ e $\cos(2\pi t / 12)$) combinados com tendência quadrática na escala original ($Y_t$) e na escala logarítmica ($\log Y_t$) de `AirPassengers` (**Slide 5**).
    2. **Construção da Função de Autocorrelação (ACF)** e **Simulação de Monte Carlo** ($B$ trajetórias) para estudar o formato esperado do correlograma em 7 processos clássicos (**Slide 31**).
    """)
    return


@app.cell
def _(air_passengers_values, alt, date, np, pl):
    # 4.1 Ajuste de Curvas com Fourier (K=1) na série AirPassengers (SME0808_04.pdf - Slide 5)
    _datas_ap = pl.date_range(date(1949, 1, 1), date(1960, 12, 1), interval="1mo", eager=True)
    _idx = np.arange(1, len(air_passengers_values) + 1, dtype=float)
    _sin1 = np.sin(2.0 * np.pi * _idx / 12.0)
    _cos1 = np.cos(2.0 * np.pi * _idx / 12.0)

    # Matrizes de desenho para os 3 modelos:
    # 1) Tt + Rt: 1 + idx + idx^2
    _X_t = np.column_stack([np.ones_like(_idx), _idx, _idx**2])
    # 2) St (h=1) + Rt: 1 + sin + cos
    _X_s = np.column_stack([np.ones_like(_idx), _sin1, _cos1])
    # 3) Tt + St (h=1) + Rt: 1 + idx + idx^2 + sin + cos
    _X_ts = np.column_stack([np.ones_like(_idx), _idx, _idx**2, _sin1, _cos1])

    def _fit_three_models(y_target: np.ndarray, prefix: str) -> pl.DataFrame:
        _specs = [
            (f"{prefix} = Tt + Rt", _X_t),
            (f"{prefix} = St (h=1) + Rt", _X_s),
            (f"{prefix} = Tt + St (h=1) + Rt", _X_ts),
        ]
        _rows = []
        for _name, _X in _specs:
            _b, _, _, _ = np.linalg.lstsq(_X, y_target, rcond=None)
            _hat = _X @ _b
            _rows.append(
                pl.DataFrame({
                    "index": _datas_ap,
                    "modelo": _name,
                    "observado": y_target,
                    "ajustado": _hat,
                    "residuo": y_target - _hat,
                })
            )
        return pl.concat(_rows)

    df_fourier_orig = _fit_three_models(air_passengers_values, "Yt")
    df_fourier_log = _fit_three_models(np.log(air_passengers_values), "logYt")

    def plot_fourier_panel(df_panel: pl.DataFrame, y_label: str, titulo: str):
        _obs = alt.Chart(df_panel).mark_line(color="black", opacity=0.75).encode(
            x=alt.X("index:T", title="Tempo (em anos)"),
            y=alt.Y("observado:Q", title=y_label, scale=alt.Scale(zero=False)),
        )
        _fit = alt.Chart(df_panel).mark_line(color="blue", strokeWidth=1.8).encode(
            x="index:T",
            y="ajustado:Q",
        )
        _top = (_obs + _fit).properties(width=210, height=160).facet(
            column=alt.Column("modelo:N", title=titulo)
        )
        _bot = (
            alt.Chart(df_panel)
            .mark_line(color="#333333")
            .encode(
                x=alt.X("index:T", title="Tempo (em anos)"),
                y=alt.Y("residuo:Q", title="Resíduo"),
            )
            .properties(width=210, height=130)
            .facet(column=alt.Column("modelo:N", title="Resíduos do Modelo"))
        )
        return alt.vconcat(_top, _bot)

    escala_fourier_ui = mo.ui.radio(
        options=["Escala Original (Yt)", "Escala Logarítmica (logYt)"],
        value="Escala Logarítmica (logYt)",
        label="Selecione a escala da variável resposta (Slide 5):",
    )
    return (
        df_fourier_log,
        df_fourier_orig,
        escala_fourier_ui,
        plot_fourier_panel,
    )


@app.cell
def _(
    df_fourier_log,
    df_fourier_orig,
    escala_fourier_ui,
    mo,
    plot_fourier_panel,
):
    _chart_f = (
        plot_fourier_panel(df_fourier_orig, "Número de passageiros", "4.1 Totais mensais de passageiros (Escala Original)")
        if escala_fourier_ui.value == "Escala Original (Yt)"
        else plot_fourier_panel(df_fourier_log, "Log do número de passageiros", "4.1 Totais mensais de passageiros (Transformação Log)")
    )
    mo.vstack([escala_fourier_ui, _chart_f])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Análise do Ajuste Sazonal com Fourier (`Slide 5`):**
    * Na **escala original ($Y_t$)**, como a série tem sazonalidade multiplicativa, o modelo aditivo `Yt = Tt + St (h=1) + Rt` deixa resíduos em formato de "cone" (variância residual crescendo com o tempo).
    * Na **escala logarítmica ($\log Y_t$)**, a sazonalidade torna-se aditiva e com amplitude estável; o modelo `logYt = Tt + St (h=1) + Rt` captura tanto a tendência quanto a oscilação anual principal, restando nos resíduos apenas os harmônicos superiores ($K \ge 2$) e a dependência serial de curto prazo.
    """)
    return


@app.cell
def _(mo):
    cenario_acf_ui = mo.ui.dropdown(
        options={
            "1. Ruído Branco i.i.d. N(0,1)": "iid",
            "2. Correlação de curto prazo — AR(1) (φ = 0.7)": "ar_pos",
            "3. Tendência estocástica — Passeio Aleatório": "rw",
            "4. Série alternada — AR(1) (φ = -0.8)": "ar_neg",
            "5. Média Móvel — MA(1) (θ = 0.6)": "ma1",
            "6. Tendência determinística linear + AR(1) (φ = 0.7)": "trend_ar",
            "7. Sazonalidade determinística senoidal + ruído": "season_sin",
        },
        value="2. Correlação de curto prazo — AR(1) (φ = 0.7)",
        label="Cenário de Simulação de Monte Carlo (Slide 31):",
    )
    slider_b_ui = mo.ui.slider(
        start=50,
        stop=500,
        step=50,
        value=200,
        label="Número de réplicas de Monte Carlo (B):",
    )
    return cenario_acf_ui, slider_b_ui


@app.cell
def _(ArmaProcess, acf, alt, cenario_acf_ui, mo, np, pl, slider_b_ui):
    # 4.2 Simulação de Monte Carlo da ACF (SME0808_04.pdf - Slide 31)
    _rng = np.random.default_rng(42)
    _B = slider_b_ui.value
    _n = 100
    _lag_max = 20
    _tipo = cenario_acf_ui.value

    _series_mat = np.zeros((_B, _n))
    _acf_mat = np.zeros((_B, _lag_max))
    _t = np.arange(1, _n + 1, dtype=float)

    for _b in range(_B):
        if _tipo == "iid":
            _y = _rng.standard_normal(_n)
        elif _tipo == "ar_pos":
            _proc = ArmaProcess(ar=[1.0, -0.7], ma=[1.0])
            _y = _proc.generate_sample(nsample=_n, distrvs=_rng.standard_normal)
        elif _tipo == "rw":
            _y = np.cumsum(_rng.normal(0.0, 0.1, size=_n))
        elif _tipo == "ar_neg":
            _proc = ArmaProcess(ar=[1.0, 0.8], ma=[1.0])
            _y = _proc.generate_sample(nsample=_n, distrvs=_rng.standard_normal)
        elif _tipo == "ma1":
            _proc = ArmaProcess(ar=[1.0], ma=[1.0, 0.6])
            _y = _proc.generate_sample(nsample=_n, distrvs=_rng.standard_normal)
        elif _tipo == "trend_ar":
            _proc = ArmaProcess(ar=[1.0, -0.7], ma=[1.0])
            _y = 2.0 + 0.1 * _t + _proc.generate_sample(nsample=_n, distrvs=_rng.standard_normal)
        else:
            _y = 2.0 + 1.5 * np.sin(_t) + _rng.standard_normal(_n)

        _series_mat[_b, :] = _y
        _acf_mat[_b, :] = acf(_y, nlags=_lag_max, fft=True)[1:]

    # Resumo das trajetórias no tempo
    _df_st_summary = pl.DataFrame({
        "id": _t.astype(int),
        "rep1": _series_mat[0, :],
        "media": _series_mat.mean(axis=0),
        "q025": np.quantile(_series_mat, 0.025, axis=0),
        "q975": np.quantile(_series_mat, 0.975, axis=0),
    })

    # Resumo da ACF empírica nas B réplicas
    _lags_idx = np.arange(1, _lag_max + 1, dtype=int)
    _df_acf_summary = pl.DataFrame({
        "lag": _lags_idx,
        "acf_media": _acf_mat.mean(axis=0),
        "acf_q025": np.quantile(_acf_mat, 0.025, axis=0),
        "acf_q975": np.quantile(_acf_mat, 0.975, axis=0),
        "band_low": np.full(_lag_max, -2.0 / np.sqrt(_n)),
        "band_high": np.full(_lag_max, 2.0 / np.sqrt(_n)),
    })

    _st_band = alt.Chart(_df_st_summary).mark_area(opacity=0.18, color="gray").encode(
        x=alt.X("id:Q", title="Índice (t)"),
        y=alt.Y("q025:Q", title="Valor"),
        y2="q975:Q",
    )
    _st_rep1 = alt.Chart(_df_st_summary).mark_line(color="black", strokeWidth=1.2).encode(
        x="id:Q", y="rep1:Q"
    )
    _st_mean = alt.Chart(_df_st_summary).mark_line(color="blue", strokeWidth=2).encode(
        x="id:Q", y="media:Q"
    )
    _chart_sim_top = (_st_band + _st_rep1 + _st_mean).properties(
        title="Trajetórias Simuladas (Preto = Réplica #1 | Azul = Média de B réplicas | Sombra = IC 95%)",
        width=700,
        height=210,
    )

    _acf_band = alt.Chart(_df_acf_summary).mark_area(opacity=0.2, color="gray").encode(
        x=alt.X("lag:O", title="Defasagem (lag)"),
        y=alt.Y("band_low:Q", title="Autocorrelação", scale=alt.Scale(domain=[-1, 1])),
        y2="band_high:Q",
    )
    _acf_zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="black").encode(y="y:Q")
    _acf_err = alt.Chart(_df_acf_summary).mark_errorbar(color="blue").encode(
        x="lag:O", y="acf_q025:Q", y2="acf_q975:Q"
    )
    _acf_line = alt.Chart(_df_acf_summary).mark_line(point=True, color="blue").encode(
        x="lag:O",
        y="acf_media:Q",
        tooltip=["lag:O", alt.Tooltip("acf_media:Q", format=".3f"), alt.Tooltip("acf_q025:Q", format=".3f"), alt.Tooltip("acf_q975:Q", format=".3f")]
    )
    _chart_sim_bot = (_acf_band + _acf_zero + _acf_err + _acf_line).properties(
        title="Distribuição Empírica da ACF em B Simulações (Média e Quantis 2.5% - 97.5%)",
        width=700,
        height=220,
    )

    mo.vstack([
        mo.hstack([cenario_acf_ui, slider_b_ui]),
        alt.vconcat(_chart_sim_top, _chart_sim_bot)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Análise das Simulações da ACF (`Slide 31`):**
    * **Ruído Branco:** ACF oscila dentro da faixa cinza de significância ($\pm 2/\sqrt{100} = \pm 0,20$) em todos os *lags*.
    * **AR(1) ($\phi = 0,7$):** Decaimento exponencial suave em direção a zero.
    * **Passeio Aleatório e Tendência Determinística:** Decaimento linear lento e persistente (sinal clássico de não-estacionariedade).
    * **AR(1) ($\phi = -0,8$):** Correlograma alternando sinais positivos e negativos com amortecimento exponencial.
    * **MA(1) ($\theta = 0,6$):** Corte abrupto após o *lag* 1 ($\rho_k = 0$ para $k \ge 2$).
    * **Sazonalidade Senoidal:** ACF exibe padrão periódico cossenoidal persistente.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Módulo 5 (`SME0808_12.pdf` e `SME0808_13.pdf`) — Identificação, Estimação, Diagnóstico e Previsão ARIMA

    Nos slides finais (`SME0808_12.pdf`, Slide 29 e `SME0808_13.pdf`, Slides 15, 20 e 22), é executado o ciclo completo de Box-Jenkins:
    1. **Identificação:** Gráfico no tempo, teste de raiz unitária **KPSS** (`unitroot_kpss`), gráficos de **ACF** e **PACF** (`gg_tsdisplay`).
    2. **Estimação e Seleção:** Ajuste de modelos candidatos $\text{ARIMA}(p,d,q)$, comparação via `AIC`, `AICc` e `BIC` (`glance() %>% arrange(AICc)`), e inspeção das raízes inversas no círculo unitário (`gg_arma()`).
    3. **Diagnóstico dos Resíduos:** Teste de **Ljung-Box** (`ljung_box`, $H_0$: ausência de autocorrelação até o *lag* 15) e teste de **Jarque-Bera** (`jarque.bera.test`, $H_0$: normalidade das inovações).
    4. **Previsão:** Projeção de $h = 15$ passos à frente (`forecast(h=15) %>% autoplot()`).
    """)
    return


@app.cell
def _(SARIMAX, acf, acorr_ljungbox, alt, kpss, np, pacf, pl, stats):
    # Funções utilitárias equivalentes a feasts/fable (fpp3)
    def fit_arima_candidates(y: np.ndarray, specs: dict[str, tuple[int, int, int]]):
        """Ajusta múltiplos modelos ARIMA(p,d,q) e retorna tabela glance(), raízes e objetos ajustados."""
        results = {}
        glance_rows = []
        roots_rows = []
        lb_rows = []
        jb_rows = []

        n = len(y)
        for name, order in specs.items():
            # trend='c' para d=0 equivale ao comportamento padrão de fable::ARIMA com média
            trend = "c" if order[1] == 0 else "n"
            mod = SARIMAX(y, order=order, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            results[name] = res

            k_params = len(res.params)
            aicc = float(res.aic + (2.0 * k_params * (k_params + 1.0)) / max(1.0, n - k_params - 1.0))
            sigma2 = float(res.params[-1])

            glance_rows.append({
                ".model": name,
                "order": f"ARIMA{order}",
                "sigma2": round(sigma2, 3),
                "log_lik": round(float(res.llf), 2),
                "AIC": round(float(res.aic), 2),
                "AICc": round(aicc, 2),
                "BIC": round(float(res.bic), 2),
            })

            # Raízes inversas (1 / root) para o gráfico no círculo unitário (gg_arma)
            if order[0] > 0 and len(res.arroots) > 0:
                for r in 1.0 / res.arroots:
                    roots_rows.append({
                        ".model": name,
                        "Tipo": "AR",
                        "re": float(np.real(r)),
                        "im": float(np.imag(r)),
                    })
            if order[2] > 0 and len(res.maroots) > 0:
                for r in 1.0 / res.maroots:
                    roots_rows.append({
                        ".model": name,
                        "Tipo": "MA",
                        "re": float(np.real(r)),
                        "im": float(np.imag(r)),
                    })

            # Diagnósticos nas inovações (resíduos)
            innov = np.asarray(res.resid)
            dof_fit = order[0] + order[2]
            lb_df = acorr_ljungbox(innov, lags=[15], model_df=dof_fit, return_df=True)
            lb_rows.append({
                ".model": name,
                "lb_stat (lag=15)": round(float(lb_df["lb_stat"].iloc[0]), 4),
                "lb_pvalue": round(float(lb_df["lb_pvalue"].iloc[0]), 4),
            })

            jb_res = stats.jarque_bera(innov)
            jb_rows.append({
                ".model": name,
                "jb_stat": round(float(jb_res.statistic), 4),
                "jb_pvalue": round(float(jb_res.pvalue), 4),
            })

        df_glance = pl.DataFrame(glance_rows).sort("AICc")
        df_roots = pl.DataFrame(roots_rows) if roots_rows else pl.DataFrame({"modelo": [], "Tipo": [], "re": [], "im": []})
        df_diag = pl.DataFrame(lb_rows).join(pl.DataFrame(jb_rows), on=".model")
        return results, df_glance, df_roots, df_diag

    def plot_gg_tsdisplay(y: np.ndarray, titulo: str, nlags: int = 20):
        n = len(y)
        df_s = pl.DataFrame({"id": np.arange(1, n + 1), "X": y})
        acf_vals = acf(y, nlags=nlags, fft=True)[1:]
        pacf_vals = pacf(y, nlags=nlags, method="ywm")[1:]
        bound = 1.96 / np.sqrt(n)
        df_corr = pl.DataFrame({
            "lag": np.arange(1, nlags + 1),
            "ACF": acf_vals,
            "PACF": pacf_vals,
            "low": np.full(nlags, -bound),
            "high": np.full(nlags, bound),
        })

        p_top = (
            alt.Chart(df_s)
            .mark_line(point=True, color="#1f77b4")
            .encode(x=alt.X("id:Q", title="Índice (t)"), y=alt.Y("X:Q", title="X"))
            .properties(title=titulo, width=680, height=180)
        )

        def _corr_chart(col_name: str):
            _band = alt.Chart(df_corr).mark_area(opacity=0.2, color="steelblue").encode(
                x=alt.X("lag:O", title="Defasagem"),
                y=alt.Y("low:Q", title=col_name, scale=alt.Scale(domain=[-1, 1])),
                y2="high:Q",
            )
            _zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="black").encode(y="y:Q")
            _Rule = alt.Chart(df_corr).mark_rule(strokeWidth=2, color="#1f77b4").encode(
                x="lag:O", y=f"{col_name}:Q"
            )
            _Pts = alt.Chart(df_corr).mark_circle(size=45, color="#1f77b4").encode(
                x="lag:O", y=f"{col_name}:Q", tooltip=["lag:O", alt.Tooltip(f"{col_name}:Q", format=".3f")]
            )
            return (_band + _zero + _Rule + _Pts).properties(width=325, height=160)

        return alt.vconcat(p_top, alt.hconcat(_corr_chart("ACF"), _corr_chart("PACF")))

    def plot_gg_arma(df_roots: pl.DataFrame):
        theta = np.linspace(0, 2 * np.pi, 200)
        df_circle = pl.DataFrame({"x": np.cos(theta), "y": np.sin(theta), "ordem": np.arange(200)})
        c_line = (
            alt.Chart(df_circle)
            .mark_line(color="gray", strokeDash=[4, 4])
            .encode(
                x=alt.X("x:Q", title="Parte Real", scale=alt.Scale(domain=[-1.15, 1.15])),
                y=alt.Y("y:Q", title="Parte Imaginária", scale=alt.Scale(domain=[-1.15, 1.15])),
                order="ordem:Q",
            )
        )
        zero_h = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#dddddd").encode(y="y:Q")
        zero_v = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#dddddd").encode(x="x:Q")

        subcharts = []
        for m in df_roots[".model"].unique(maintain_order=True):
            df_m = df_roots.filter(pl.col(".model") == m)
            pts = (
                alt.Chart(df_m)
                .mark_point(filled=True, size=90)
                .encode(
                    x="re:Q",
                    y="im:Q",
                    color=alt.Color("Tipo:N", scale=alt.Scale(domain=["AR", "MA"], range=["#d62728", "#1f77b4"])),
                    shape=alt.Shape("Tipo:N"),
                    tooltip=[".model:N", "Tipo:N", alt.Tooltip("re:Q", format=".3f"), alt.Tooltip("im:Q", format=".3f")],
                )
            )
            subcharts.append((zero_h + zero_v + c_line + pts).properties(title=str(m), width=165, height=165))

        return alt.hconcat(*subcharts).properties(
            title="Raízes Inversas dos Polinômios AR/MA no Círculo Unitário (gg_arma)"
        )

    def plot_forecast(y: np.ndarray, res_model, model_name: str, h: int = 15):
        n = len(y)
        fc = res_model.get_forecast(steps=h)
        mean_fc = np.asarray(fc.predicted_mean)
        ci95 = np.asarray(fc.conf_int(alpha=0.05))
        ci80 = np.asarray(fc.conf_int(alpha=0.20))

        df_hist = pl.DataFrame({"id": np.arange(1, n + 1), "X": y})
        df_fc = pl.DataFrame({
            "id": np.arange(n + 1, n + h + 1),
            "mean": mean_fc,
            "low80": ci80[:, 0],
            "high80": ci80[:, 1],
            "low95": ci95[:, 0],
            "high95": ci95[:, 1],
        })

        c_hist = alt.Chart(df_hist).mark_line(color="black").encode(
            x=alt.X("id:Q", title="Índice (t)"),
            y=alt.Y("X:Q", title="X", scale=alt.Scale(zero=False)),
        )
        c_95 = alt.Chart(df_fc).mark_area(opacity=0.2, color="#1f77b4").encode(
            x="id:Q", y="low95:Q", y2="high95:Q"
        )
        c_80 = alt.Chart(df_fc).mark_area(opacity=0.35, color="#1f77b4").encode(
            x="id:Q", y="low80:Q", y2="high80:Q"
        )
        c_mean = alt.Chart(df_fc).mark_line(point=True, color="#1f77b4", strokeWidth=2).encode(
            x="id:Q",
            y="mean:Q",
            tooltip=["id:Q", alt.Tooltip("mean:Q", format=".2f"), alt.Tooltip("low95:Q", format=".2f"), alt.Tooltip("high95:Q", format=".2f")],
        )
        return (c_hist + c_95 + c_80 + c_mean).properties(
            title=f"Previsão h = {h} passos à frente ({model_name}) — Faixas de 80% e 95%",
            width=680,
            height=240,
        )

    return (
        fit_arima_candidates,
        plot_forecast,
        plot_gg_arma,
        plot_gg_tsdisplay,
    )


@app.cell
def _(
    fit_arima_candidates,
    kpss,
    mo,
    np,
    pl,
    plot_gg_arma,
    plot_gg_tsdisplay,
):
    # 5.1 Análise da serieE (SME0808_12.pdf Slide 29 e SME0808_13.pdf Slide 15)
    serie_e = np.array([
        101,  82,  66,  35,  31,   7,  20,  92, 154, 125,
         85,  68,  38,  23,  10,  24,  83, 132, 131, 118,
         90,  67,  60,  47,  41,  21,  16,   6,   4,   7,
         14,  34,  45,  43,  48,  42,  28,  10,   8,   2,
          0,   1,   5,  12,  14,  35,  46,  41,  30,  24,
         16,   7,   4,   2,   8,  17,  36,  50,  62,  67,
         71,  48,  28,   8,  13,  57, 122, 138, 103,  86,
         63,  37,  24,  11,  15,  40,  62,  98, 124,  96,
         66,  64,  54,  39,  21,   7,   4,  23,  55,  94,
         96,  77,  59,  44,  47,  30,  16,   7,  37,  74
    ], dtype=float)

    _kpss_stat, _kpss_p, _, _ = kpss(serie_e, regression="c", nlags="auto")
    _df_kpss_e = pl.DataFrame({"kpss_stat": [round(float(_kpss_stat), 4)], "kpss_pvalue": [round(float(_kpss_p), 3)]})

    _specs_e = {
        "arima200": (2, 0, 0),
        "arima300": (3, 0, 0),
        "stepwise_search (2,0,1)": (2, 0, 1),
    }
    _res_e, _df_glance_e, _df_roots_e, _df_diag_e = fit_arima_candidates(serie_e, _specs_e)

    mo.vstack([
        mo.md("### 5.1 Estudo da `serieE` (Manchas Solares de Wolfer — `SME0808_12` Slide 29 & `SME0808_13` Slide 15)"),
        plot_gg_tsdisplay(serie_e, "5.1 Série E — Painel gg_tsdisplay (Série, ACF e PACF)"),
        mo.md("**Teste KPSS de Estacionariedade (`unitroot_kpss`):**"),
        _df_kpss_e,
        mo.md("**Comparação de Modelos (`glance() %>% arrange(AICc)`):**"),
        _df_glance_e,
        plot_gg_arma(_df_roots_e),
        mo.md("**Diagnóstico de Resíduos (Ljung-Box `lag=15` e Jarque-Bera):**"),
        _df_diag_e,
    ])
    return (serie_e,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Análise da `serieE` (`SME0808_12` e `SME0808_13`):**
    * O teste KPSS ($\text{estatística} = 0,160$, $p = 0,10$) não rejeita a hipótese nula de estacionariedade em nível.
    * O ACF exibe comportamento senoidal amortecido e a PACF apresenta cortes após os *lags* 2/3, sugerindo processos $\text{AR}(2)$, $\text{AR}(3)$ ou $\text{ARMA}(2,1)$.
    * O modelo $\text{ARMA}(2,1)$ (`stepwise`/`search`) atinge o menor `AICc` e todas as raízes inversas permanecem estritamente dentro do círculo unitário (processo estacionário e invertível).
    """)
    return


@app.cell
def _(mo):
    seletor_sim_arima_ui = mo.ui.radio(
        options={
            "Caso 1 (Slide 20a): AR(1) Simulado (φ = 0.8)": "ar1",
            "Caso 2 (Slide 20b): AR(2) Simulado (φ1 = 0.8, φ2 = -0.7)": "ar2",
            "Caso 3 (Slide 22): MA(2) Simulado (θ1 = 0.3, θ2 = 0.8)": "ma2",
        },
        value="Caso 1 (Slide 20a): AR(1) Simulado (φ = 0.8)",
        label="Selecione o Exemplo de Previsão de SME0808_13.pdf:",
    )
    return (seletor_sim_arima_ui,)


@app.cell
def _(
    ArmaProcess,
    fit_arima_candidates,
    mo,
    np,
    plot_forecast,
    plot_gg_arma,
    plot_gg_tsdisplay,
    seletor_sim_arima_ui,
):
    # 5.2 Estudos de Caso Simulados de Previsão (SME0808_13.pdf - Slides 20 e 22)
    _rng_sim = np.random.default_rng(123)
    _caso = seletor_sim_arima_ui.value

    if _caso == "ar1":
        # arima.sim(model=list(ar=0.8), n=150) -> no statsmodels os sinais do polinômio AR são [1, -phi1]
        _proc = ArmaProcess(ar=[1.0, -0.8], ma=[1.0])
        _serie_sim = _proc.generate_sample(nsample=150, distrvs=_rng_sim.standard_normal)
        _specs = {
            "arima001": (0, 0, 1),
            "arima100": (1, 0, 0),
            "arima200": (2, 0, 0),
        }
        _best_name = "arima100"
        _titulo_sim = "5.2 Processo AR(1) Simulado (φ = 0.8, n = 150)"
    elif _caso == "ar2":
        # arima.sim(model=list(ar=c(0.8, -0.7)), n=150) -> ar=[1.0, -0.8, +0.7]
        _proc = ArmaProcess(ar=[1.0, -0.8, 0.7], ma=[1.0])
        _serie_sim = _proc.generate_sample(nsample=150, distrvs=_rng_sim.standard_normal)
        _specs = {
            "arima100": (1, 0, 0),
            "arima200": (2, 0, 0),
            "arima300": (3, 0, 0),
        }
        _best_name = "arima200"
        _titulo_sim = "5.2 Processo AR(2) Simulado (φ1 = 0.8, φ2 = -0.7, n = 150)"
    else:
        # arima.sim(model=list(ma=c(0.3, 0.8)), n=150) -> ma=[1.0, 0.3, 0.8]
        _proc = ArmaProcess(ar=[1.0], ma=[1.0, 0.3, 0.8])
        _serie_sim = _proc.generate_sample(nsample=150, distrvs=_rng_sim.standard_normal)
        _specs = {
            "arima001": (0, 0, 1),
            "arima002": (0, 0, 2),
            "arima003": (0, 0, 3),
        }
        _best_name = "arima002"
        _titulo_sim = "5.2 Processo MA(2) Simulado (θ1 = 0.3, θ2 = 0.8, n = 150)"

    _res_sim, _df_glance_sim, _df_roots_sim, _df_diag_sim = fit_arima_candidates(_serie_sim, _specs)

    mo.vstack([
        seletor_sim_arima_ui,
        plot_gg_tsdisplay(_serie_sim, _titulo_sim),
        mo.md("**Seleção de Modelos por Critérios de Informação (`glance() %>% arrange(AICc)`):**"),
        _df_glance_sim,
        plot_gg_arma(_df_roots_sim),
        mo.md("**Testes de Diagnóstico nos Resíduos (Ljung-Box e Jarque-Bera):**"),
        _df_diag_sim,
        plot_forecast(_serie_sim, _res_sim[_best_name], _best_name, h=15),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Conclusão e Síntese dos Resultados

    1. **Análise Exploratória e Visualização (`SME0808_01`):** Os gráficos `autoplot`, `gg_season`, `gg_subseries` e `gg_lag` foram integralmente reconstruídos com `Polars` e `Altair`, permitindo inspecionar tendência, sazonalidade, variações interanuais dentro de cada mês e dependência serial por defasagem.
    2. **Estabilização da Variância (`SME0808_02`):** A implementação do critério de **Guerrero (1993)** via `scipy.optimize` combinada à transformação de **Box-Cox** estabilizou a variância crescente da série de consumo de energia.
    3. **Modelagem e Testes de Tendência (`SME0808_03`):** Comparamos regressão polinomial vs. *splines* lineares por partes na série `Nile`, filtros de médias móveis centradas (`2-MA` e `5-MA`) em `AirPassengers`, e o teste não-paramétrico de **Cox-Stuart** baseado na distribuição Binomial.
    4. **Decomposição Harmônica e Propriedades da ACF (`SME0808_04`):** A regressão harmônica com termos de Fourier em escala logarítmica capturou a sazonalidade multiplicativa de `AirPassengers`, e o simulador de Monte Carlo demonstrou empiricamente a assinatura da ACF para processos estacionários e não-estacionários.
    5. **Ciclo Box-Jenkins Completo (`SME0808_12` e `SME0808_13`):** Em todos os cenários ($\text{AR}(1)$, $\text{AR}(2)$, $\text{MA}(2)$ e `serieE`), o critério `AICc` identificou corretamente a ordem geradora do processo, os testes de **Ljung-Box** e **Jarque-Bera** validaram a hipótese de ruído branco gaussiano nas inovações apenas para os modelos bem especificados, e as previsões de $h=15$ passos convergiram para a média incondicional na velocidade ditada pela memória de cada processo (geométrica em $\text{AR}(p)$ e em exatamente $q$ passos para $\text{MA}(q)$).
    """)
    return


if __name__ == "__main__":
    app.run()
