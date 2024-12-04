# Flask-related imports
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
# Flask is a lightweight web framework in Python. These imports allow us to:
# - Create and manage a web application (`Flask`).
# - Render HTML templates (`render_template`).
# - Handle HTTP requests (`request`) and redirections (`redirect`).
# - Generate URLs dynamically (`url_for`).
# - Enable file handling (`send_file`).
# - Display feedback messages to users (`flash`).

# OS and file-related imports
import os  # For interacting with the operating system (e.g., file paths, environment variables).
import io  # Provides tools to work with in-memory file-like objects.
import base64  # Encodes/decodes data in Base64, useful for embedding binary data like images in HTML.

# Pandas and PyArrow for data manipulation
import pandas as pd  # High-level data manipulation library, especially useful for structured data (e.g., tables).
import pyarrow as pa  # In-memory data format for analytics (efficient data interchange).
import pyarrow.csv as pv  # High-performance CSV reader provided by PyArrow.

# Visualization libraries
import matplotlib
matplotlib.use('Agg')  # Ensures matplotlib uses a backend suitable for server environments (no GUI).
import matplotlib.pyplot as plt  # Provides a plotting interface for creating static, animated, and interactive visuals.
import seaborn as sns  # High-level API for creating attractive and informative statistical graphics.

# Statsmodels for statistical modeling
import statsmodels.api as sm  # Comprehensive library for statistical models and tests.
from statsmodels.formula.api import ols  # Formula-based API for ordinary least squares (OLS) regression.
from statsmodels.discrete.discrete_model import Logit  # Logit model for binary outcomes.
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Calculate VIF for multicollinearity.
from statsmodels.stats.diagnostic import het_breuschpagan, het_white  # Tests for heteroskedasticity.
from statsmodels.graphics.regressionplots import plot_ccpr, plot_leverage_resid2
# plot_ccpr: Conditional Component + Residual plots for regression diagnostics.
# plot_leverage_resid2: Leverage vs. residual-squared plot for assessing influence.
from statsmodels.stats.stattools import durbin_watson  # Test for autocorrelation in residuals.
from statsmodels.stats.power import TTestIndPower  # Power analysis for t-tests.

# Linearmodels for panel data analysis
from linearmodels.panel import PanelOLS, RandomEffects  # Models for panel data (e.g., repeated observations per entity).
from linearmodels.datasets import wage_panel  # Example dataset for panel data analysis.

# Miscellaneous
from scipy.stats import zscore  # Calculate the z-score for each value in a dataset (standardization).
import math  # Provides mathematical functions like square root, trigonometric functions, etc.
import textwrap  # Conveniently wraps text into lines of a specified width (useful for formatting).
import numpy as np


# Create an instance of the Flask application
app = Flask(__name__)

# Configure the folder where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = './uploads'

# Secret key for session management and security
# This is used by Flask to securely sign session cookies and flash messages
app.secret_key = 'LCP'

# Initialize a global variable to hold the currently uploaded dataset
# This allows the application to persist the dataset across different user actions
current_dataset = None

@app.route('/')
def index():
    """
    Home page of the application.

    This function handles requests to the root URL ('/') of the application.
    It renders the 'index.html' template, which is the front-end interface for the user.
    """
    return render_template('index.html')  # Render the homepage

def optimize_memory(df):
    """
    Optimize the memory usage of a DataFrame by adjusting data types.

    This function examines the columns of the input DataFrame and:
    - Converts object/string columns to the 'category' type if they have high cardinality.
    - Downcasts numeric columns to more efficient numeric types (e.g., float32).

    Args:
        df (pd.DataFrame): The DataFrame to optimize.

    Returns:
        pd.DataFrame: The optimized DataFrame with reduced memory usage.
    """
    # Iterate over object-type columns (e.g., strings) and convert to 'category' if appropriate
    for col in df.select_dtypes(include=['object']).columns:
        # Check if the column has high cardinality (unique values relative to total rows)
        if df[col].nunique() / len(df[col]) < 0.5:  # Threshold of 50% unique values
            df[col] = df[col].astype('category')  # Convert to 'category' type

    # Downcast numeric columns (int and float) to minimize memory usage
    for col in df.select_dtypes(include=['int', 'float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')  # Downcast to the smallest possible float type

    return df  # Return the optimized DataFrame

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    Handle the upload of datasets.

    This function processes file uploads via HTTP POST requests and renders the upload page via GET requests.
    It supports uploading and processing of `.csv` and `.dta` files. Data is loaded in chunks for memory efficiency,
    and global memory optimization is applied to the dataset before storing it in the global `current_dataset` variable.

    Returns:
        - On POST:
            * If successful, redirects to the 'data_view' route.
            * If no file is selected or an unsupported format is provided, returns an error message with a 400 status code.
            * If an error occurs during file processing, returns an error message with a 500 status code.
        - On GET:
            * Renders the 'upload.html' template for the upload form.
    """
    global current_dataset  # Use the global variable to store the uploaded dataset

    if request.method == 'POST':  # Handle file upload
        file = request.files['file']  # Access the uploaded file object

        if file.filename == '':  # Check if a file was selected
            return "No file selected", 400  # Respond with a "Bad Request" status if no file is selected

        # Save the uploaded file to the configured upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            if file.filename.endswith('.csv'):  # Handle CSV files
                # Load CSV data in chunks for memory efficiency
                chunk_size = 5000  # Define the number of rows per chunk
                current_dataset = pd.DataFrame()  # Initialize an empty DataFrame to hold the dataset
                reader = pv.read_csv(filepath, read_options=pv.ReadOptions(block_size=chunk_size))
                # Read CSV in batches and process each chunk
                for batch in reader.to_batches():
                    chunk = batch.to_pandas()  # Convert the batch to a Pandas DataFrame
                    chunk = optimize_memory(chunk)  # Optimize memory usage for the chunk
                    current_dataset = pd.concat([current_dataset, chunk], ignore_index=True)  # Append chunk
            elif file.filename.endswith('.dta'):  # Handle Stata (.dta) files
                # Load Stata data in chunks for memory efficiency
                chunks = pd.read_stata(filepath, chunksize=5000)
                current_dataset = pd.concat([optimize_memory(chunk) for chunk in chunks], ignore_index=True)
            else:
                # Unsupported file format
                return "Unsupported file format. Please upload a .csv or .dta file.", 400
        except Exception as e:
            # Handle errors during file processing
            return f"Error loading file: {str(e)}", 500

        # Redirect to the data view page after a successful upload and processing
        return redirect(url_for('data_view'))

    # Render the upload form for GET requests
    return render_template('upload.html')

@app.route('/data', methods=['GET'])
def data_view():
    """
    Displays the dataset with pagination (1000 rows per page).
    
    This function renders a paginated view of the dataset stored in the global variable `current_dataset`.
    It also allows filtering of the dataset based on user-specified conditions.
    """
    global current_dataset  # Access the globally stored dataset
    if current_dataset is None:
        # If no dataset is loaded, redirect the user to the upload page
        return redirect(url_for('upload'))

    # Get filtering parameters from the query string (if provided)
    filter_column = request.args.get('filter_column', None)  # The column to apply the filter on
    filter_condition = request.args.get('filter_condition', None)  # The condition to apply (e.g., >10, contains 'value')
    filtered_dataset = current_dataset.copy()  # Make a copy of the dataset for applying filters

    # Apply filtering logic if the user specifies a filter
    if filter_column and filter_condition:
        try:
            # Handle conditions for range queries (>, <, >=, <=, =)
            if '>' in filter_condition or '<' in filter_condition or '=' in filter_condition:
                filtered_dataset = filtered_dataset.query(f"{filter_column} {filter_condition}")
            # Handle string matching queries (e.g., "contains 'value'")
            elif "contains" in filter_condition:
                value = filter_condition.split("'")[1]  # Extract the value for string matching
                filtered_dataset = filtered_dataset[filtered_dataset[filter_column].astype(str).str.contains(value)]
        except Exception as e:
            # If an error occurs in filtering, display a flash message with the error details
            flash(f"Invalid filter condition: {e}", "danger")

    # Get the current page number from query parameters; default to page 1 if not specified
    page = int(request.args.get('page', 1))
    rows_per_page = 1000  # Define the number of rows to display per page

    # Calculate the start and end row indices for the current page
    start_row = (page - 1) * rows_per_page  # Starting index for the page
    end_row = start_row + rows_per_page  # Ending index for the page

    # Slice the filtered dataset to get the rows for the current page
    paginated_data = current_dataset.iloc[start_row:end_row]

    # Convert the paginated dataset into an HTML table with Bootstrap styling
    data_table = paginated_data.to_html(classes="table table-striped", index=False)

    # Calculate the total number of pages for pagination
    total_pages = (len(current_dataset) + rows_per_page - 1) // rows_per_page

    # Pagination logic for slicing filtered data
    page = int(request.args.get('page', 1))  # Current page number
    rows_per_page = 1000  # Rows displayed per page
    start_row = (page - 1) * rows_per_page  # Start index for pagination
    end_row = start_row + rows_per_page  # End index for pagination

    # Slice the filtered dataset for the current page
    paginated_data = filtered_dataset.iloc[start_row:end_row]

    # Generate an HTML table for the paginated data
    data_table = paginated_data.to_html(classes="table table-striped", index=False)

    # Calculate total pages for the filtered dataset
    total_pages = (len(filtered_dataset) + rows_per_page - 1) // rows_per_page

    # Render the `data.html` template, passing pagination details and the HTML table
    return render_template(
        'data.html',
        data_table=data_table,  # HTML table of the current page's data
        current_page=page,  # Current page number
        total_pages=total_pages,  # Total number of pages
        total_observations=len(filtered_dataset),  # Total number of rows in the filtered dataset
        rows_per_page=rows_per_page,  # Number of rows displayed per page
        formatted_total_observations=f"{len(filtered_dataset):,}",  # Total observations formatted (e.g., "1,000")
        max=max,  # Pass the built-in max function for use in the template
        min=min,  # Pass the built-in min function for use in the template
        current_dataset=current_dataset  # Provide access to the full dataset (optional context)
    )

@app.route('/drop_missing', methods=['POST'])
def drop_missing():
    """
    Drops all missing values from the dataset and redirects to the dataset view.

    This route replaces placeholder missing values (e.g., 'missing', '') with NaN and
    then removes all rows containing NaN values from the dataset.
    """
    global current_dataset  # Access the globally stored dataset
    if current_dataset is not None:
        # Replace placeholders (e.g., 'missing', '') with NaN
        current_dataset.replace([''], np.nan, inplace=True)

        # Drop all rows with missing values from the dataset in place
        current_dataset.dropna(inplace=True)

        # Display a success message to the user
        flash("All missing values have been dropped.", "success")
    else:
        # Display an error message if no dataset is loaded
        flash("No dataset is currently loaded.", "danger")
    
    # Redirect to the dataset view page
    return redirect(url_for('data_view'))

@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    """
    Generate visualizations with options for title, subtitle, caption, export as JPEG/LaTeX, and show Python code.

    This route handles both GET (render visualization options) and POST (generate visualizations) requests.
    It supports various plot types (scatter, line, bar, etc.) and allows customization such as adding titles, 
    subtitles, captions, and log scales. Users can export visualizations as JPEG or generate LaTeX code 
    for embedding in documents.
    """
    global current_dataset  # Access the globally stored dataset
    if current_dataset is None:
        # Redirect to the upload page if no dataset is loaded
        return redirect(url_for('upload'))

    # Initialize variables for error handling and output
    error_message = None
    plot_url = None  # URL to the generated plot
    export_code = None  # Python code to recreate the plot
    latex_code = None  # LaTeX code to include the plot in a document

    # Guidance text for each plot type
    plot_guidance = {
        'scatter': 'Scatter Plot: Use numeric variables for X and Y axes. Optionally group by a categorical variable.',
        'line': 'Line Plot: Use numeric variables for X and Y axes. Optionally group by a categorical variable.',
        'bar': 'Bar Plot: Use a categorical variable for X-axis and a numeric variable for Y-axis.',
        'box': 'Box Plot: Use a categorical variable for X-axis and a numeric variable for Y-axis.',
        'violin': 'Violin Plot: Use a categorical variable for X-axis and a numeric variable for Y-axis.',
        'hist': 'Histogram: Use a numeric variable for X-axis.',
        'kde': 'KDE Plot: Use numeric variables for X-axis and optionally Y-axis.',
        'heatmap': 'Heatmap: Requires at least two numeric variables.',
        'pairplot': 'Pair Plot: Works with all numeric variables in the dataset.',
    }

    # Default plot type is scatter
    plot_type = "scatter"

    if request.method == 'POST':
        # Retrieve plot parameters from the form
        plot_type = request.form.get('plot_type', "scatter")  # Plot type (default: scatter)
        x_var = request.form.get('x_var')  # X-axis variable
        y_var = request.form.get('y_var') or None  # Y-axis variable (optional)
        hue_var = request.form.get('hue_var') or None  # Grouping variable (optional)
        title = request.form.get('title') or ''  # Plot title (optional)
        subtitle = request.form.get('subtitle') or ''  # Plot subtitle (optional)
        caption = request.form.get('caption') or ''  # Plot caption (optional)
        log_scale = 'log_scale' in request.form  # Boolean for log scaling

        # Validate user input
        if x_var and x_var not in current_dataset.columns:
            error_message = f"Invalid X-Axis variable: {x_var}"
        elif y_var and y_var not in current_dataset.columns:
            error_message = f"Invalid Y-Axis variable: {y_var}"

        if not error_message:
            # Start creating the plot
            plt.figure(figsize=(10, 6))  # Set figure size
            try:
                # Generate the appropriate plot based on the selected type
                if plot_type == 'scatter':
                    sns.scatterplot(data=current_dataset, x=x_var, y=y_var, hue=hue_var)
                elif plot_type == 'line':
                    sns.lineplot(data=current_dataset, x=x_var, y=y_var, hue=hue_var)
                elif plot_type == 'bar':
                    sns.barplot(data=current_dataset, x=x_var, y=y_var, hue=hue_var)
                elif plot_type == 'box':
                    sns.boxplot(data=current_dataset, x=x_var, y=y_var, hue=hue_var)
                elif plot_type == 'violin':
                    sns.violinplot(data=current_dataset, x=x_var, y=y_var, hue=hue_var)
                elif plot_type == 'hist':
                    sns.histplot(data=current_dataset, x=x_var, hue=hue_var, kde=True)
                elif plot_type == 'kde':
                    sns.kdeplot(data=current_dataset, x=x_var, y=y_var, hue=hue_var, fill=True)
                elif plot_type == 'heatmap':
                    numeric_data = current_dataset.select_dtypes(include=['number'])  # Select numeric columns
                    if numeric_data.shape[1] < 2:
                        raise ValueError("Heatmap requires at least two numeric columns.")
                    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')  # Correlation heatmap
                elif plot_type == 'pairplot':
                    sns.pairplot(current_dataset.select_dtypes(include=['number']))  # Pairwise plots for numeric data
                else:
                    raise ValueError("Invalid plot type selected. Please choose a valid option.")

                # Add optional title, subtitle, and caption
                if title:
                    plt.title(title, fontsize=16, weight='bold', pad=15)
                if subtitle:
                    plt.suptitle(subtitle, fontsize=12, style='italic', y=0.98)
                if caption:
                    plt.figtext(0.5, -0.1, caption, ha='center', fontsize=10, wrap=True)

                # Apply log scaling if selected
                if log_scale:
                    plt.xscale('log')  # Apply log scale to X-axis
                    if y_var:
                        plt.yscale('log')  # Apply log scale to Y-axis if available

                # Save the plot as a JPEG file
                if not os.path.exists('static'):  # Ensure the static folder exists
                    os.makedirs('static')
                plot_filename = 'visualization.jpeg'  # Define the filename for the plot
                plot_path = os.path.join('static', plot_filename)
                plt.savefig(plot_path, format='jpeg', bbox_inches='tight')  # Save the plot
                plot_url = '/' + plot_path  # Set the URL for the saved plot

                # Generate LaTeX code to embed the plot
                latex_code = f"""
                \\begin{{figure}}[H]
                    \\centering
                    \\includegraphics[width=0.8\\textwidth]{{visualization.jpeg}}
                    \\caption{{{caption if caption else f"Generated {plot_type.capitalize()} Plot"}}}
                \\end{{figure}}
                """

                # Generate Python code to recreate the plot
                export_code = f"""
import seaborn as sns
import matplotlib.pyplot as plt

sns.{plot_type}(data=data, x='{x_var}', y='{y_var}', hue='{hue_var}')
plt.title("{title}")
plt.suptitle("{subtitle}")
plt.figtext(0.5, -0.1, "{caption}", ha='center', fontsize=10, wrap=True)
{"plt.xscale('log')" if log_scale else ""}
{"plt.yscale('log')" if log_scale and y_var else ""}
plt.show()
                """

            except Exception as e:
                # Handle errors during plot generation
                error_message = f"Error generating {plot_type} plot: {e}"

            plt.close()  # Close the plot to release memory

    # Render the visualization page with plot details and options
    return render_template(
        'visualize.html',
        columns=current_dataset.columns,  # Dataset columns for plot variable selection
        plot_url=plot_url,  # URL to the generated plot
        error_message=error_message,  # Error message, if any
        export_code=export_code,  # Python code for plot recreation
        latex_code=latex_code,  # LaTeX code for plot embedding
        plot_guidance=plot_guidance,  # Guidance for different plot types
        selected_plot_type=plot_type  # The selected plot type
    )

@app.route('/model', methods=['GET', 'POST'])
def model():
    """
    Build and evaluate a statistical model using the uploaded dataset.

    This route supports Ordinary Least Squares (OLS) regression with options for robust standard errors.
    Users can specify a dependent variable, independent variables, and diagnostics are provided including 
    Variance Inflation Factor (VIF), Breusch-Pagan test, and White test. Visualization of residuals and Q-Q 
    plots is also included. Python and LaTeX code for replication are generated.
    """
    global current_dataset  # Access the globally stored dataset
    if current_dataset is None:
        # Redirect to the upload page if no dataset is loaded
        return redirect(url_for('upload'))

    # Initialize variables for results, diagnostics, and error handling
    model_results = None
    diagnostics = None
    error_message = None
    formula = None  # String representing the formula used in the model
    python_code = None  # Python code to recreate the model
    latex_code = None  # LaTeX code for presenting the results

    if request.method == 'POST':
        # Get user input from the form
        dependent_var = request.form.get('dependent_var')  # Dependent variable (Y)
        independent_vars = request.form.getlist('independent_vars')  # List of independent variables (X)
        model_type = request.form.get('model_type')  # Type of model (e.g., OLS)
        robust_errors = request.form.get('robust_errors') == 'true'  # Boolean for robust standard errors

        # Validate user input
        if not dependent_var:
            error_message = "Please select a dependent variable."
        elif not independent_vars:
            error_message = "Please select at least one independent variable."
        else:
            try:
                # Construct the formula for the model
                formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"

                # Fit the model using OLS
                model = sm.OLS.from_formula(formula, data=current_dataset)
                if robust_errors:
                    results = model.fit(cov_type='HC3')  # Fit with robust standard errors
                else:
                    results = model.fit()  # Fit without robust standard errors

                # Extract and format model results
                model_results = {
                    'summary': results.summary().as_html(),  # HTML summary for display
                    'coefficients': results.params.to_dict(),  # Coefficients as a dictionary
                    'pvalues': results.pvalues.to_dict(),  # P-values as a dictionary
                    'rsquared': results.rsquared,  # R-squared value
                    'adj_rsquared': results.rsquared_adj,  # Adjusted R-squared value
                }

                # Initialize diagnostics dictionary
                diagnostics = {}

                # Calculate Variance Inflation Factor (VIF)
                X = current_dataset[independent_vars]
                X = sm.add_constant(X)  # Add intercept term
                vif_data = {
                    col: variance_inflation_factor(X.values, i) for i, col in enumerate(X.columns)
                }
                diagnostics['vif'] = {
                    'data': vif_data,
                    'decision_rule': "A VIF > 10 indicates high multicollinearity, which may inflate standard errors and reduce model reliability."
                }

                # Perform Breusch-Pagan test for heteroscedasticity
                _, bp_pval, _, _ = het_breuschpagan(results.resid, results.model.exog)
                diagnostics['bp_test'] = {
                    'p_value': bp_pval,
                    'decision_rule': (
                        "If the p-value < 0.05, there is evidence of heteroscedasticity (non-constant variance), "
                        "which may affect the efficiency of OLS estimates."
                    ),
                    'interpretation': "Heteroscedasticity detected" if bp_pval < 0.05 else "No heteroscedasticity detected"
                }

                # Perform White test for heteroscedasticity
                _, white_pval, _, _ = het_white(results.resid, results.model.exog)
                diagnostics['white_test'] = {
                    'p_value': white_pval,
                    'decision_rule': (
                        "If the p-value < 0.05, there is evidence of heteroscedasticity. "
                        "The White test is a more general test for non-constant variance."
                    ),
                    'interpretation': "Heteroscedasticity detected" if white_pval < 0.05 else "No heteroscedasticity detected"
                }

                # Create Residual Plot
                plt.figure(figsize=(6, 4))
                plt.scatter(results.fittedvalues, results.resid)
                plt.axhline(0, color='red', linestyle='--', linewidth=1)
                plt.xlabel("Fitted Values")
                plt.ylabel("Residuals")
                plt.title("Residuals vs Fitted")
                residual_img = io.BytesIO()
                plt.savefig(residual_img, format="png")
                residual_img.seek(0)
                diagnostics['residual_plot'] = base64.b64encode(residual_img.getvalue()).decode()  # Encode as base64

                # Create Q-Q Plot
                plt.figure(figsize=(6, 4))
                sm.qqplot(results.resid, line='s', ax=plt.gca())
                plt.title("Q-Q Plot")
                qq_img = io.BytesIO()
                plt.savefig(qq_img, format="png")
                qq_img.seek(0)
                diagnostics['qq_plot'] = base64.b64encode(qq_img.getvalue()).decode()  # Encode as base64

                # Generate Python code for replication
                python_code = f"""
import pandas as pd
import statsmodels.api as sm

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Define the formula
formula = "{formula}"

# Fit the model
model = sm.OLS.from_formula(formula, data=data)
results = model.fit(cov_type='HC3' if {robust_errors} else 'nonrobust')

# Summary
print(results.summary())
                """

                # Generate LaTeX code for results
                latex_code = results.summary().as_latex()  # Convert summary to LaTeX
                latex_code = latex_code.replace('\\toprule', '\\hline').replace('\\bottomrule', '\\hline')  # Format LaTeX
                latex_code = latex_code.replace('\\midrule', '\\hline')  # Replace midrule with hline
            except Exception as e:
                error_message = f"Error running the model: {e}"

    # Render the model results and diagnostics in the template
    return render_template(
        'model.html',
        columns=current_dataset.columns,  # Pass dataset columns for variable selection
        model_results=model_results,  # Model results for display
        diagnostics=diagnostics,  # Diagnostic tests and plots
        error_message=error_message,  # Error message if any
        formula=formula,  # The model formula used
        python_code=python_code,  # Python code for replication
        latex_code=latex_code,  # LaTeX code for results
    )

@app.route('/summary', methods=['GET', 'POST'])
def summary():
    """
    Generate summary statistics for selected variables.
    
    Only numeric variables are allowed for summary statistics. If a categorical variable is selected,
    an error message is displayed.
    """
    global current_dataset  # Access the globally stored dataset
    if current_dataset is None:
        # Redirect to the upload page if no dataset is loaded
        return redirect(url_for('upload'))

    # Define statistic titles for numeric variables
    stat_titles = {
        'mean': 'Mean',
        'median': 'Median',
        'std': 'Standard Deviation',
        'min': 'Minimum',
        'max': 'Maximum',
        'sum': 'Sum',
        'count': 'Count',
    }

    summary_results = {}  # Store processed summary statistics
    error_message = None  # Store error messages if any

    if request.method == 'POST':
        # Get user-selected variables and statistics
        selected_vars = request.form.getlist('variables')  # Selected variables
        stats_to_compute = request.form.getlist('stats')  # Selected statistics

        if not selected_vars:
            error_message = "Please select at least one variable."
        elif not stats_to_compute:
            error_message = "Please select at least one statistic."
        else:
            try:
                # Check for categorical variables in the selection
                non_numeric_vars = [
                    var for var in selected_vars
                    if var in current_dataset.select_dtypes(include=['object', 'category']).columns
                ]
                if non_numeric_vars:
                    error_message = (
                        f"The following variables are not numeric and cannot be summarized: {', '.join(non_numeric_vars)}."
                    )
                else:
                    # Process numeric variables only
                    for var in selected_vars:
                        stats_map = {
                            'mean': current_dataset[var].mean(),
                            'median': current_dataset[var].median(),
                            'std': current_dataset[var].std(),
                            'min': current_dataset[var].min(),
                            'max': current_dataset[var].max(),
                            'sum': current_dataset[var].sum(),
                            'count': current_dataset[var].count(),
                        }
                        # Filter statistics based on user selection
                        summary_results[var] = {stat: stats_map.get(stat, "") for stat in stats_to_compute}
            except Exception as e:
                # Catch and store any errors during processing
                error_message = f"Error generating summary: {e}"

    # Render the summary template with results
    return render_template(
        'summary.html',
        columns=current_dataset.columns,  # Pass dataset columns for selection
        summary_results=summary_results,  # Pass computed statistics
        stat_titles=stat_titles,  # Pass statistic titles for display
        error_message=error_message,  # Pass any error messages
    )


def generate_latex_code(summary_results, stat_titles):
    """
    Generate a LaTeX table for the summary statistics.

    This function takes the computed summary statistics and generates a properly formatted
    LaTeX table, which can be included in a LaTeX document. The table includes all the 
    selected statistics for each variable.

    Args:
        summary_results (dict): A dictionary containing the computed statistics for each variable.
            The keys are variable names, and the values are dictionaries of statistics.
        stat_titles (dict): A dictionary mapping statistic keys (e.g., 'mean', 'median') to 
            their user-friendly titles (e.g., 'Mean', 'Median').

    Returns:
        str: A string containing the LaTeX code for the summary table.
    """
    # Initialize LaTeX table structure with a floating table environment
    latex = "\\begin{table}[htbp]\n"
    latex += "\\hspace*{-4cm}\n"  # Add horizontal spacing for alignment
    latex += "\\centering\n"  # Center the table on the page
    latex += "\\renewcommand{\\arraystretch}{1.2} % Adjust row spacing\n"  # Increase row spacing for readability
    latex += "\\setlength{\\tabcolsep}{10pt} % Adjust column spacing\n"  # Increase column spacing

    # Define the table format: 'l' for the first column (left-aligned) and 'c' for statistic columns (center-aligned)
    latex += "\\begin{tabular}{|l" + "c" * len(stat_titles) + "|}\n"
    latex += "\\hline\n"  # Add a horizontal line at the top of the table

    # Add the header row with variable name and statistic titles
    latex += "\\textbf{Variable} & " + " & ".join(f"\\textbf{{{title}}}" for title in stat_titles.values()) + " \\\\\n"
    latex += "\\hline\n"  # Add a horizontal line below the header

    # Populate the table with data rows
    for var, stats in summary_results.items():
        # Construct a row with the variable name and its statistics
        row = " & ".join(
            f"{stats.get(stat, ''):.2f}" if isinstance(stats.get(stat, ''), (int, float)) else str(stats.get(stat, '')) 
            for stat in stat_titles.keys()
        )
        latex += f"{var} & {row} \\\\\n"

    latex += "\\hline\n"  # Add a horizontal line at the end of the data rows

    # Close the table environment
    latex += "\\end{tabular}\n"
    latex += "\\caption{Summary Statistics}\n"  # Add a caption to describe the table
    latex += "\\label{tab:summary_statistics}\n"  # Add a label for referencing the table
    latex += "\\end{table}\n"

    return latex  # Return the LaTeX code as a string

@app.route('/power', methods=['GET', 'POST'])
def power():
    """
    Perform power analysis for hypothesis testing.

    This route allows users to calculate sample size, power, or effect size for a two-sample t-test.
    Users can input parameters such as alpha, power, effect size, mean difference, and standard deviation.
    The results are displayed, and error handling ensures users are guided on valid input requirements.

    Returns:
        - Rendered HTML template 'power.html' with power analysis results or error messages.
    """
    # Initialize variables for storing results and error messages
    power_results = None
    error_message = None

    if request.method == 'POST':
        # Retrieve input values from the form
        calc_type = request.form.get('calc_type')  # Type of calculation ('sample_size', 'power', or 'effect_size')
        alpha = request.form.get('alpha')  # Significance level (alpha)
        power = request.form.get('power')  # Desired statistical power
        effect_size = request.form.get('effect_size')  # Effect size
        nobs = request.form.get('nobs')  # Number of observations
        mean_diff = request.form.get('mean_diff')  # Mean difference
        std_dev = request.form.get('std_dev')  # Standard deviation

        try:
            # Convert inputs to appropriate data types
            alpha = float(alpha) if alpha else None
            power = float(power) if power else None
            effect_size = float(effect_size) if effect_size else None
            nobs = int(nobs) if nobs else None
            mean_diff = float(mean_diff) if mean_diff else None
            std_dev = float(std_dev) if std_dev else None

            # Calculate effect size if mean difference and standard deviation are provided
            if mean_diff and std_dev:
                effect_size = mean_diff / std_dev

            # Initialize the power analysis object
            analysis = TTestIndPower()

            # Perform the requested calculation
            if calc_type == 'sample_size':
                # Calculate the required sample size
                if effect_size is None or power is None or alpha is None:
                    raise ValueError("Effect size, power, and alpha are required for sample size calculation.")
                nobs = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha)
                power_results = {'calculation': 'Sample Size', 'value': math.ceil(nobs)}  # Round up sample size
            elif calc_type == 'power':
                # Calculate the statistical power
                if effect_size is None or nobs is None or alpha is None:
                    raise ValueError("Effect size, sample size, and alpha are required for power calculation.")
                power = analysis.solve_power(effect_size=effect_size, nobs=nobs, alpha=alpha)
                power_results = {'calculation': 'Power', 'value': power}
            elif calc_type == 'effect_size':
                # Calculate the effect size
                if nobs is None or power is None or alpha is None:
                    raise ValueError("Sample size, power, and alpha are required for effect size calculation.")
                effect_size = analysis.solve_power(nobs=nobs, power=power, alpha=alpha)
                power_results = {'calculation': 'Effect Size', 'value': effect_size}
            else:
                # Handle invalid calculation type
                raise ValueError("Invalid calculation type selected.")
        except Exception as e:
            # Catch and display errors during power calculation
            error_message = f"Error in power calculation: {e}"

    # Render the power analysis template with results and error messages
    return render_template('power.html', power_results=power_results, error_message=error_message)

if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)