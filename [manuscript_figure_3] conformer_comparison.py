import os
import sqlite3

import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

db_path = "./data/pac5_conformer_comparison.db"
EV_TO_KJ_MOL = 96.4853321233100184


def main():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT pubchem_cid,
                  num_atoms,
                  xtb_energy_rdkit_random_openff,
                  xtb_energy_rdkit_pubchem3d,
                  xtb_energy_rdkit_mmff94,
                  xtb_energy_rdkit_random_mmff94
           FROM conformers""",
        conn,
    )
    conn.close()

    if df.empty:
        print("No data found in the database.")
        os._exit(0)

    os.makedirs("./figures", exist_ok=True)
    x_max = df["num_atoms"].max() * 1.05

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        x_title="<b>Number of atoms</b>",
        y_title="<b>Δ energy (kJ/mol)</b>",
        horizontal_spacing=0.02,
    )

    default_colors = plotly.colors.qualitative.Plotly
    sources = [
        (
            "delta_pubchem3d",
            "xtb_energy_rdkit_pubchem3d",
            "PubChem3D",
            default_colors[0],
        ),
        ("delta_mmff94", "xtb_energy_rdkit_mmff94", "RDKit MMFF", default_colors[1]),
        (
            "delta_random_mmff94",
            "xtb_energy_rdkit_random_mmff94",
            "RDKit MMFF random",
            default_colors[2],
        ),
    ]

    letters = ["a", "b", "c"]

    for i, (delta_col, source_col, source_name, color) in enumerate(sources, start=1):
        df_pair = df.dropna(
            subset=["num_atoms", "xtb_energy_rdkit_random_openff", source_col]
        ).copy()

        if df_pair.empty:
            continue

        ref = df_pair["xtb_energy_rdkit_random_openff"]
        df_pair[delta_col] = (df_pair[source_col] - ref) * EV_TO_KJ_MOL

        mean_val = df_pair[delta_col].mean()

        fig.add_trace(
            go.Scattergl(  # If any problems, change to go.Scatter. Scattergl is much faster though
                x=df_pair["num_atoms"],
                y=df_pair[delta_col],
                mode="markers",
                name=source_name,
                text=df_pair["pubchem_cid"],
                marker=dict(color=color, opacity=0.6),
                showlegend=False,
            ),
            row=1,
            col=i,
        )

        # Baseline at 0
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=i)

        # Mean line
        fig.add_hline(y=mean_val, line_dash="solid", line_color=color, row=1, col=i)

        # Subplot letter
        fig.add_annotation(
            text=f"<b>{letters[i - 1]}</b>",
            xref="x domain",
            yref="y domain",
            x=0.03,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=28, color="black"),
            row=1,
            col=i,
        )

        # Upper favored annotation
        fig.add_annotation(
            text="RDKit OpenFF",
            xref="x domain",
            yref="y domain",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(size=24, color="black"),
            row=1,
            col=i,
        )

        # Mean annotation
        fig.add_annotation(
            text=f"Mean: {'+' if mean_val > 0 else ''}{mean_val:.2f} kJ/mol",
            xref="x domain",
            yref="y domain",
            x=0.98,
            y=0.25,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            font=dict(size=24),
            row=1,
            col=i,
        )

        # Lower favored annotation
        fig.add_annotation(
            text=f"{source_name}",
            xref="x domain",
            yref="y domain",
            x=0.98,
            y=0.10,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=24, color="black"),
            row=1,
            col=i,
        )

    fig.update_xaxes(
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="inside",
        showline=True,
        range=[0, x_max],
        dtick=50,
    )

    fig.update_yaxes(
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="inside",
        showline=True,
        title_standoff=100,
    )

    fig.update_annotations(
        selector=dict(text="<b>Number of atoms</b>"), font=dict(size=28)
    )
    fig.update_annotations(
        selector=dict(text="<b>Δ energy (kJ/mol)</b>"), font=dict(size=28), xshift=-75
    )

    fig.update_layout(
        template="plotly",
        height=600,
        width=1200,
        font=dict(size=24),
        margin=dict(t=40, b=90, l=120, r=20),
    )

    fig.write_image("./figures/manuscript_figure_3.png", scale=2)


if __name__ == "__main__":
    main()
