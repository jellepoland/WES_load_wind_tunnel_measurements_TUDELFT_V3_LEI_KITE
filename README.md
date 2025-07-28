[![CODECHECK](https://codecheck.org.uk/img/codeworks-badge.svg)](https://doi.org/10.5281/zenodo.15603144)

# Load Balance Analysis
This repository contains code that transforms raw data measured into tables and plots used in a paper, titled "Wind Tunnel Load Measurements Of A Rigid Subscale Leading Edge Inflatable Kite" published Open-Source in Wind Energy science, [link].
The raw data is from a subscale rigid leading-edge inflatable kite model measured during a Wind Tunnel campaign in the Open-Jet Facility of the TU Delft in April 2024. The data is published Open-Source and available on [Zenodo](10.5281/zenodo.14288467) 

## Installation Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/jellepoland/load_balance_wind_tunnel_measurement_analysis_of_TUDELFT_V3_LEI_KITE_scale_model
    ```

2. Navigate to the repository folder:
    ```bash
    cd load_balance_wind_tunnel_measurement_analysis_of_TUDELFT_V3_LEI_KITE_scale_model
    ```
    
3. Create a virtual environment:
   
   Linux or Mac:
    ```bash
    python3 -m venv venv
    ```
    
    Windows:
    ```bash
    python -m venv venv
    ```
    
5. Activate the virtual environment:

   Linux or Mac:
    ```bash
    source venv/bin/activate
    ```

    Windows
    ```bash
    .\venv\Scripts\activate
    ```

6. Install the required dependencies:

   For users:
    ```bash
    pip install .
    ```
        
   For developers:
    ```bash
    pip install -e .[dev]
    ```

7. To deactivate the virtual environment:
    ```bash
    deactivate
    ```
### Dependencies
- numpy
- pandas>=1.5.3
- matplotlib>=3.7.1,
- ipykernel,
- statsmodels,
- VSM @ git+https://github.com/ocayon/Vortex-Step-Method.git@develop


## Usages
1. Follow installation instructions
2. Download data from [Insert Link](..) and place all data, with current names, inside the `data/` folder. The resulting folder should have `data/CFD_polar_data` and `data/normal` etc.
3. Run `scripts/main.py`

### Inner Logic
`scripts/main_process.py` calls all the processing scripts, a short listing and description follows:
- `process_raw_lvm_with_labbook_into_df`: reads the labbook in csv format and raw measurements in lvm format, and save them in csv format to `processed_data/without_csv` folder
- `process_support_struc_aero_interp_coeffs.py`: reads the processed_data and interpolates the support structure measurements, saves these as csv into `processed_data` folder
- `process_normal_csv`: Use the `process_raw_lvm_with_labbook_into_df` script to analyze the "normal" runs and save these into as csv in `processed_data/normal` folder
- `process_zigzag_csv`: Read out the zigzag specific measurements, and saved as csv in `processed_data/zigzag`
- `process_vsm`: Run the [Vortex-Step Method](https://github.com/ocayon/Vortex-Step-Method/tree/main/src/VSM) to produce aerodynamic results, that will be used in the comparison. 
- `process_bundling_beta_0`: bundles the results for the beta_0 case, such that it is easier to process

`scripts/main_plot.py` calls all the plotting and printing scripts, that generate all the tables (in latex format) and plots present in the manuscript.


## Citation
If you use this project in your research, please consider citing it. 
Citation details can be found in the [CITATION.cff](CITATION.cff) file included in this repository.
- UPDATE THE CITATION


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## :warning: License and Waiver

Specify the license under which your software is distributed and include the copyright notice:

> Technische Universiteit Delft hereby disclaims all copyright interest in the program “NAME PROGRAM” (one line description of the content or function) written by the Author(s).
> 
> Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering
> 
> Copyright (c) [YEAR] [NAME SURNAME].

## :gem: Help and Documentation
[AWE Group | Developer Guide](https://awegroup.github.io/developer-guide/)


