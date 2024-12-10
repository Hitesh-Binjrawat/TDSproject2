# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "matplotlib",
#   "requests",
#   "seaborn",
#   "charset_normalizer"
# ]
# ///
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import os
from charset_normalizer import detect
# Function for Generic Visualization of Data
def visualize_data(data):
    """Generate visualizations for the data."""
    images = []
    
    # Correlation heatmap (if numerical columns exist)
    if data.select_dtypes(include=["number"]).shape[1] > 1:
        plt.figure(figsize=(10, 8))
        corr = data.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        heatmap_path = "correlation_heatmap.png"
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_path)
        plt.close()
        images.append(heatmap_path)
    
    # Distribution of numerical columns
    for column in data.select_dtypes(include=["number"]).columns[:2]:  # Limit to 2 plots for simplicity
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True, bins=30)
        hist_path = f"{column}_distribution.png"
        figsize = (5.12, 5.12)
        dpi=100
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.savefig(hist_path,dpi=dpi)
        plt.close()
        images.append(hist_path)
    
    return images

# Create a Function to create a Readme File 
def create_readme(data, analysis, images,api_key):
    """Generate a README.md file containing the story."""
    columns = ", ".join(analysis["columns"])
    shape = f"{analysis['shape'][0]} rows and {analysis['shape'][1]} columns"
    missing_values = pd.DataFrame.from_dict(analysis["missing_values"], orient="index", columns=["Missing Values"])

    # Ask LLM for a narrative
    prompt = f"""
    I have analyzed a dataset with the following details:
    - Shape: {shape}
    - Columns: {columns}
    - Missing Values: {missing_values.to_string()}
    Can you narrate a story about this dataset and the insights gained from the analysis?
    """
    story = ask_llm(prompt,api_key)
    
    # Write the README.md
    with open("README.md", "w") as f:
        f.write("# Automated Data Analysis\n")
        f.write("## Dataset Overview\n")
        f.write(f"- Shape: {shape}\n")
        f.write(f"- Columns: {columns}\n")
        f.write("\n## Insights\n")
        f.write(story)
        f.write("\n## Visualizations\n")
        for img in images:
            f.write(f"![{img}]({img})\n")


def ask_llm(prompt,api_key):
    """Make an API request to OpenAI to generate a response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raises an error for bad responses
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        sys.exit(1)
    
def perform_analysis(data):
    """Perform basic analysis on the dataset."""
    analysis = {
        "shape": data.shape,
        "columns": list(data.columns),
        "missing_values": data.isnull().sum().to_dict(),
        "summary_statistics": data.describe(include="all").to_dict(),
    }
    return analysis

def read_csv_with_fallback(file_path):
    """
    Reads a CSV file, handling unknown encodings dynamically.
    """
    try:
        # Attempt to read with utf-8 encoding
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # Detect encoding using charset_normalizer
            with open(file_path, 'rb') as file:
                detected_encoding = detect(file.read())['encoding']
            
            # Read the file with detected encoding
            return pd.read_csv(file_path, encoding=detected_encoding)
        except Exception as e:
            # Attempt fallback encodings if detection fails
            for encoding in ['latin1', 'cp1252', 'utf-16']:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except Exception:
                    continue
            
            # If all fails, raise the last exception
            raise Exception(f"Unable to read the file: {e}")

# Usage
    

# Main Function     
def main():
    file_name = sys.argv[1]
    try:
        data = read_csv_with_fallback(file_name)
        print("File loaded successfully!")
    except Exception as e:
        print(f"Error: {e}")
    key=os.environ.get('AIPROXY_TOKEN')
    visualize_data(data)
    analysis=perform_analysis(data)
    images=visualize_data(data)
    # creating a readme file 
    create_readme(data, analysis, images,key)
    


if __name__ == "__main__":
    main()
