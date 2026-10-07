import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import requests
    import marimo as mo
    import polars as pl

    return mo, pl, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Select api key file
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    file_api_key = mo.ui.file(
        filetypes=[".txt"],
        multiple=False,
        kind="area",
        label="Arraste o arquivo CSV aqui ou clique para selecionar",
    )
    file_api_key
    return (file_api_key,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Select places_id file
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    file_places_ui = mo.ui.file(
        filetypes=[".csv"],
        multiple=False,
        kind="area",
        label="Arraste o arquivo CSV aqui ou clique para selecionar",
    )
    file_places_ui
    return (file_places_ui,)


@app.cell
def _(file_places_ui, mo, pl):
    try:
        df_csv = pl.read_csv(file_places_ui.contents() , columns=["place_id"])
        unique_ids = df_csv["place_id"].drop_nulls().unique().to_list()
        csv_status = mo.callout(
            mo.md(f"✅ Loaded **{len(unique_ids)} unique place IDs** from `per_store.csv`"),
            kind="success",
        )
    except Exception as _e:
        unique_ids = []
        csv_status = mo.callout(mo.md(f"❌ Could not read CSV: {_e}"), kind="danger")

    ids_preview = mo.ui.text_area(
        label=f"Place IDs loaded from CSV ({len(unique_ids)} unique)",
        value="\n".join(unique_ids),
        rows=8,
        disabled=True,
    )

    mo.vstack([csv_status, ids_preview])
    return (unique_ids,)


@app.cell
def _(file_api_key, mo, requests, unique_ids):
    BASE_URL = "https://places.googleapis.com/v1/places/{place_id}"
    FIELD_MASK = "displayName,formattedAddress,addressComponents"

    def fetch_place(place_id: str, api_key: str) -> dict:
        url = BASE_URL.format(place_id=place_id.strip())
        headers = {
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": FIELD_MASK,
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def extract_city(address_components: list, formatted_address: str = "") -> str:
        """Extract city from address components, trying multiple type fallbacks.

        Priority order:
          1. locality              – standard city in most countries
          2. administrative_area_level_2 – city/municipality in BR and others
          3. sublocality_level_1   – district used as city in some regions
          4. Parse formattedAddress – last resort: second-to-last comma-segment
        """
        for city_type in ("locality", "administrative_area_level_2", "sublocality_level_1"):
            for component in address_components:
                if city_type in component.get("types", []):
                    return component.get("longText", "")

        # Fallback: formattedAddress is typically "Street, City, State, Country"
        if formatted_address:
            parts = [p.strip() for p in formatted_address.split(",")]
            if len(parts) >= 2:
                return parts[-2]  # second-to-last segment is usually the city

        return ""

    api_key = file_api_key.contents().strip()
    place_ids = unique_ids  # sourced directly from the CSV

    results = []
    errors = []

    mo.stop(not api_key or not place_ids, mo.callout(
        mo.md("**Enter your API key above — place IDs are loaded automatically from the CSV.**"),
        kind="info",
    ))

    with mo.status.progress_bar(
        total=len(place_ids),
        title="Fetching places…",
        subtitle="Calling Google Places API",
        completion_title="Done!",
    ) as bar:
        for pid in place_ids:
            try:
                data = fetch_place(pid, api_key)
                name = data.get("displayName", {}).get("text", "N/A")
                address = data.get("formattedAddress", "N/A")
                city = extract_city(data.get("addressComponents", []), address)
                results.append({
                    "Place ID": pid,
                    "Name": name,
                    "City": city,
                    "Address": address,
                })
            except Exception as e:
                errors.append({"Place ID": pid, "Error": str(e)})
            bar.update()

    summary = mo.callout(
        mo.md(
            f"✅ **{len(results)} successful** out of {len(place_ids)} requests"
            + (f" — ❌ {len(errors)} failed" if errors else "")
        ),
        kind="success" if not errors else "warn",
    )
    summary
    return errors, results


@app.cell
def _(errors, mo, results):
    elements = []

    if results:
        elements.append(mo.md("### ✅ Results"))
        elements.append(mo.ui.table(results))

    if errors:
        elements.append(mo.md("### ❌ Errors"))
        elements.append(mo.ui.table(errors))

    mo.vstack(elements) if elements else mo.callout(
        mo.md("No results yet."), kind="warn"
    )
    return


if __name__ == "__main__":
    app.run()
