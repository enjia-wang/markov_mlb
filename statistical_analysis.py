import pandas as pd
import matplotlib.pyplot as plt
import statistics

def create_and_save_histogram(data: list[int], filename: str) -> None:
    """
    Creates a histogram from a list of discrete integers, displays the mean 
    and standard deviation in a corner box, and saves it as an image.

    Args:
        data (list[int]): A list of discrete integer values.
        filename (str): The name of the output image file (e.g., 'histogram.png').
    """
    if not data:
        print("Error: The data list is empty.")
        return

    # 1. Calculate statistics
    mean_val = statistics.mean(data)
    # Standard deviation requires at least two data points
    if len(data) > 1:
        std_val = statistics.stdev(data)
        stats_text = f"Mean: {mean_val:.2f}\nStd Dev: {std_val:.2f}"
    else:
        stats_text = f"Mean: {mean_val:.2f}\nStd Dev: N/A"

    # 2. Define bins for discrete integers
    min_val = min(data)
    max_val = max(data)
    bins = range(min_val, max_val + 2)

    # 3. Set up and draw the plot
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, align='left', edgecolor='black', color='skyblue')

    # 4. Add formatting and labels
    plt.title('Histogram of Discrete Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xticks(range(min_val, max_val + 1))

    # 5. Add the statistics box in the top-right corner
    # bbox creates the physical box around the text. 
    # transAxes uses relative coordinates (0 to 1) rather than data coordinates.
    box_properties = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
    plt.gca().text(0.95, 0.95, stats_text, 
                   transform=plt.gca().transAxes, 
                   fontsize=11, 
                   verticalalignment='top', 
                   horizontalalignment='right', 
                   bbox=box_properties)

    # 6. Save and close
    plt.savefig(filename, bbox_inches='tight')
    print(f"Histogram successfully saved to '{filename}'")
    plt.close()

def get_column_as_list(csvpath: str, column_name: str) -> list:
    """
    Reads a CSV file using pandas and returns a specific column as a Python list.

    Args:
        csvpath (str): The file path to the CSV.
        column_name (str): The name of the column to extract.

    Returns:
        list: A list containing the values from the specified column. 
              Returns an empty list if the file or column is not found.
    """
    try:
        # 1. Load the CSV into a pandas DataFrame
        df = pd.read_csv(csvpath)
        
        # 2. Extract the column and convert it directly to a list
        return df[column_name].tolist()
        
    except FileNotFoundError:
        print(f"Error: The file at '{csvpath}' could not be found.")
        return []
    except KeyError:
        print(f"Error: The column '{column_name}' does not exist in the CSV.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def plot_predicted_vs_real(pred_csv: str, real_csv: str, cols: tuple):
    id_col, val_col = cols
    
    df_pred = pd.read_csv(pred_csv)
    df_real = pd.read_csv(real_csv)
    
    # --- THE FIX: Force columns to numeric ---
    # The errors='coerce' argument turns any text it can't parse into NaN
    df_pred[val_col] = pd.to_numeric(df_pred[val_col], errors='coerce')
    df_real[val_col] = pd.to_numeric(df_real[val_col], errors='coerce')
    
    # Drop those newly created NaN rows so they don't break the plot
    df_pred = df_pred.dropna(subset=[val_col])
    df_real = df_real.dropna(subset=[val_col])
    # -----------------------------------------
    
    # Merge on the ID column
    df_merged = pd.merge(
        df_real, 
        df_pred, 
        on=id_col, 
        how='inner', 
        suffixes=('_real', '_pred')
    )
    
    if df_merged.empty:
        print("Warning: No matching numeric trials found between the two CSV files.")
        return

    real_vals = df_merged[f"{val_col}_real"]
    pred_vals = df_merged[f"{val_col}_pred"]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(real_vals, pred_vals, alpha=0.7, edgecolors='k', label='Data points')
    
    # This will now work perfectly because both arrays are strictly numbers
    min_val = min(real_vals.min(), pred_vals.min())
    max_val = max(real_vals.max(), pred_vals.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
    
    plt.title('Predicted vs. Real Values', fontsize=14)
    plt.xlabel('Real Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.show()

if __name__ == "__main__":

    import sys
    if len(sys.argv) > 2: 
        csv_path = sys.argv[1]
        column = sys.argv[2]
        plot_name = column + ".png"
    
        # get list of data values 
        extracted_list = get_column_as_list(csv_path, column)

        # plot the histogram
        create_and_save_histogram(extracted_list,plot_name)
    else:
        if sys.argv[1] == "vs":
            plot_predicted_vs_real("ATC_Pitchers_2026.csv","ATC_Pitchers_may29.csv",("Name","SO%"))
            sys.exit()
        innings = [1,2,3,4,5,6,7,8,9]
        category = "SO"
        name_list = []
        for i in innings: 
            name_list.append("Home_" + category + "_In" + str(i))
            name_list.append("Away_" + category + "_In" + str(i))
        
        for i in name_list: 
            extracted_list = get_column_as_list("mlb_archive.csv", i)
            plot_name = i + ".png"
            create_and_save_histogram(extracted_list,plot_name)


