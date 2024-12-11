# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "matplotlib",
#   "requests",
#   "seaborn",
#   "charset_normalizer",
#   "chardet",
# ]
# ///
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import os
import chardet
import pandas as pd
import base64


# Step 1: Try reading with default settings
import pandas as pd
import chardet

def prompt():
    return """You are a data storytelling assistant. Your task is to analyze data and provide insightful narratives, clearly structured as Markdown content. 

    Your output should always include:
    1. **Title**: Begin with an engaging title.
    2. **Introduction**: Briefly introduce the topic or dataset.
    3. **Key Insights**: Highlight the most critical findings with bullet points or tables where necessary.
    4. **Visualization Suggestions**: Provide recommendations for visualizations (e.g., bar charts, line graphs) to represent the data effectively.
    5. **Conclusion**: Summarize the story and its implications.

    Always format your response with proper Markdown syntax, including headings (`#`, `##`, etc.), bullet points (`-`), tables, and inline code when appropriate. Example:

    # Title
    ## Introduction
    - Key point 1
    - Key point 2

    ## Key Insights
    | Column A | Column B |
    |----------|----------|
    | Value 1  | Value 2  |

    ## Visualization Suggestions
    - Bar chart for X vs. Y
    - Line graph to show trends

    ## Conclusion
    Summarize the findings here.

    Respond only in Markdown.
    """
def read_unknown_csv(file_path):
    # Step 1: Detect file encoding
    try:
        with open(file_path, 'rb') as file:
            result = chardet.detect(file.read())
            encoding = result['encoding']
        print(f"Detected encoding: {encoding}")
    except Exception as e:
        print(f"Failed to detect encoding: {e}")
        encoding = 'utf-8'  # Default fallback encoding

    # Step 2: Try reading the CSV with inferred delimiter
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding=encoding)
        print("Successfully read CSV with inferred delimiter.")
        return df
    except Exception as e:
        print(f"Failed to read CSV with inferred delimiter: {e}")

    # Step 3: Fall back to default delimiter (',') as a last resort
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print("Successfully read CSV with default delimiter (',').")
        return df
    except Exception as e:
        print(f"Failed to read CSV with default delimiter: {e}")

    # If all attempts fail
    print("Failed to read the CSV file.")
    return None

# Usage Example:
# df = read_unknown_csv('unknown_data.csv')
# if df is not None:
#     print(df.head())

# Function for Generic Visualization of Data
def visualize_data(data,file_name):
    """Generate visualizations for the data."""
    images = []
    
    # Correlation heatmap (if numerical columns exist)
    if data.select_dtypes(include=["number"]).shape[1] > 1:
        plt.figure(figsize=(10, 8))
        corr = data.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        heatmap_path = "correlation_heatmap.png"
        plt.title("Correlation Heatmap")
        plt.savefig(f"./{heatmap_path}")
        plt.close()
        images.append(f"./{heatmap_path}")
    
    # Distribution of numerical columns
    for column in data.select_dtypes(include=["number"]).columns[:2]:  # Limit to 2 plots for simplicity
        plt.figure(figsize=(5.12,5.12))
        sns.histplot(data[column], kde=True, bins=30)
        plt.plot()
        hist_path = f"./{column}_distribution.png"
        dpi=100
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.savefig(hist_path,dpi=dpi)
        plt.close()
        images.append(hist_path)
    
    return images

# Create a Function to create a Readme File 
import os
import requests
import pandas as pd
import json


def create_readme(analysis, api_key, folder_path):
    """Generate a README.md file containing the story and include image content."""
    # Extract dataset details
    columns = ", ".join(analysis["columns"])
    shape = f"{analysis['shape'][0]} rows and {analysis['shape'][1]} columns"
    missing_values = pd.DataFrame.from_dict(analysis["missing_values"], orient="index", columns=["Missing Values"])

    # Prepare the prompt for LLM
    prmpt = f"""
    I have analyzed a dataset with the following details:
    - Shape: {shape}
    - Columns: {columns}
    - Missing Values: {missing_values.to_string()}
    Additionally, I am attaching image data related to this dataset. 
    Please include their insights in the narrative.
    """

    # Check if the folder exists
    # folder_path = f"./{folder_path.rstrip('/')}"  # Ensure no trailing slash
    # if not os.path.isdir(folder_path):
    #     print(f"Error: Folder '{folder_path}' does not exist.")
    #     return

    # Prepare image contents as Base64
    image_prompts = []
    image_links = []
    for file_name in os.listdir("."):
        if file_name.endswith(".png"):
            # file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_name, "rb") as file:
                    # Encode image in Base64
                    encoded_image = base64.b64encode(file.read()).decode("utf-8")
                    # Add image information to prompt
                    image_links.append(f"![{file_name}](./{file_name})")

                    image_prompts.append(
                        f"Image: {encoded_image}\n\n(Base64-encoded content omitted for brevity)\n"
                    )
            except Exception as e:
                print(f"Error reading file '{file_name}': {e}")

    # Append image descriptions to the prompt
    # prmpt += "\n\nImages analyzed:\n" + "\n".join(image_prompts)

    # Headers for API request
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # URL for LLM storytelling API
    url_story = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    data_story = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prmpt}
        ]
    }

    # Request narrative from LLM
    try:
        response_story = requests.post(url_story, headers=headers, json=data_story)
        response_story.raise_for_status()
        story_content = response_story.json()["choices"][0]["message"]["content"]

        # Write the story to a README file
        with open("README.md", "w") as readme_file:
            readme_file.write("# Dataset Story\n\n")
            readme_file.write(story_content)
            readme_file.write("\n\nImages analyzed:\n" + "\n".join(image_links))

            print("README.md file created successfully!")
    except requests.exceptions.RequestException as e:
        print(f"Error generating story: {e}")

def ask_llm(prompt,api_key,file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
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
# Main Function     
def main():
    file_name = sys.argv[1]
    data = read_unknown_csv(file_name)
    # data=pd.read_csv(file_name)
    print(data,file_name)
    key=os.getenv("AIPROXY_TOKEN")
    # print(os.environ.get('AIPROXY_TOKEN'))
    analysis=perform_analysis(data)
    visualize_data(data,file_name)
    # # creating a readme file 
    create_readme(analysis,key,file_name[:-4])
    # print(data)
    # create_readme(data, analysis, images,api_key,folder_path)


if __name__ == "__main__":
    main()
