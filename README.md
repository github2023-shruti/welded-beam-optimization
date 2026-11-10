# Welded Beam Multi-Objective Optimization (Hybrid SMPSO → NSGA-II)

This project builds an interactive Streamlit GUI to solve the classic welded beam design as a 3-objective problem:

- Cost (minimize)
- Weight (minimize)
- Weld Area (minimize)

Subject to strength and deflection constraints.

A hybrid optimization pipeline is used:

1. SMPSO explores the search space and discovers diverse candidates quickly.
2. NSGA-II is then seeded with the SMPSO population to refine the Pareto front.

## Requirements

- Python 3.9+
- See `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Files

- `problem.py`: Defines the welded beam problem, variables, objectives, and constraints.
- `optimizer.py`: Implements the SMPSO → NSGA-II hybrid and utilities to extract top solutions.
- `plotting.py`: Plotly-based 3D and 2D Pareto visualizations.
- `app.py`: Streamlit UI to configure, run, and visualize results.

## Notes

- The Pareto front displayed prefers feasible solutions. If none are feasible, the UI will show the entire final population and warn you.
- Top solutions per objective report the best (min), worst (max), and average objective values along with the corresponding/average decision variables.
