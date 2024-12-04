
# MyRA (My Research Assistant)
This is a Flask-based web application designed to help researchers and data analysts perform common econometric tasks. It allows users to upload datasets, visualize data, compute summary statistics, run econometric models, and perform power calculations.

## Features
### 1. **Upload Dataset**
- Upload datasets in CSV format.
- The uploaded dataset is stored globally for use across different tools.

### 2. **Data Viewer**
- View the uploaded dataset.
- Inspect variables and data types.

### 3. **Visualizations**
- Generate plots with various options (scatter, bar, box, line, etc.).
- Add titles, subtitles, and captions to visualizations.
- Export plots as JPEG or LaTeX-compatible figures.
- View Python code for reproducibility.

### 4. **Summary Statistics**
- Compute key statistics (mean, median, standard deviation, etc.).
- Export summary results as LaTeX tables.
- View Python code for reproducibility.

### 5. **Econometric Modeling**
- Build econometric models using Ordinary Least Squares (OLS).
- Supports robust standard errors.
- Preprocesses categorical variables with one-hot encoding.
- Diagnostics:
  - Variance Inflation Factor (VIF) for multicollinearity.
  - Breusch-Pagan test for heteroscedasticity.
  - White test for heteroscedasticity.
- Visualizations:
  - Residual plots.
  - Q-Q plots for residuals.
- View Python and LaTeX code for reproducibility.

### 6. **Power Calculations**
- Perform statistical power calculations for t-tests.
- Calculate:
  - Required sample size.
  - Power of the test.
  - Effect size.
- View Python code for reproducibility.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/myra.git
   ```
2. Navigate to the project directory:
   ```bash
   cd myra
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   flask run
   ```

---

## Usage
### Upload Dataset
1. Navigate to `/upload` to upload a dataset.
2. Ensure your dataset is a valid CSV file.

### Summary Statistics
1. Navigate to `/summary`.
2. Select variables and statistics to compute.
3. View results in a table, along with LaTeX and Python code.

### Visualizations
1. Navigate to `/visualize`.
2. Select the plot type, variables, and additional options.
3. Generate and download plots or view Python code.

### Econometric Modeling
1. Navigate to `/model`.
2. Select dependent and independent variables.
3. View results, diagnostics, and code for replication.

### Power Calculations
1. Navigate to `/power`.
2. Select calculation type, enter parameters, and compute results.

---

## Code Overview
### `/app.py`
- Contains all route definitions and core functionalities.
- Implements preprocessing, modeling, diagnostics, and visualization logic.

### `/templates/`
- HTML templates for rendering web pages.
- Includes:
  - `base.html`: Common layout template.
  - `visualize.html`: For visualizations.
  - `model.html`: For econometric modeling.
  - `summary.html`: For summary statistics.

### `/static/`
- Stores exported plots (JPEG format).

---

## License
This project is licensed under the MIT License.

---

## Contributions
Contributions are welcome! Feel free to fork the repository and create pull requests. For major changes, open an issue first to discuss what you'd like to change. 

For questions, contact [lance.pangilinan@yale.edu].
