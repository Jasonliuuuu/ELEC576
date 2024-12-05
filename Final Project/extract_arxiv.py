import json
import csv
import os

def extract_arxiv_metadata(json_file_path, output_csv_path):
    """
    Extracts 'title', 'abstract', and 'categories' fields from the arXiv metadata JSON file
    and saves the results into a CSV file.
    
    Args:
        json_file_path (str): Path to the arXiv metadata JSON file.
        output_csv_path (str): Path to save the resulting CSV file.
    """
    try:
        # Check if the input JSON file exists
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")

        print(f"Extracting data from {json_file_path}...")

        # Open the JSON file and the output CSV file
        with open(json_file_path, 'r', encoding='utf-8') as infile, \
             open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
            
            # Set up CSV writer
            fieldnames = ["title", "abstract", "categories"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()  # Write CSV header
            
            # Process JSON file line by line
            for i, line in enumerate(infile, start=1):
                try:
                    record = json.loads(line)
                    writer.writerow({
                        "title": record.get("title", "").strip(),
                        "abstract": record.get("abstract", "").strip(),
                        "categories": record.get("categories", "").strip()
                    })

                except json.JSONDecodeError:
                    print(f"Error decoding JSON on line {i}. Skipping...")
                
                # Print progress every 100,000 lines
                if i % 100000 == 0:
                    print(f"Processed {i} records...")

        print(f"Extraction complete. Data saved to {output_csv_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Main entry point
if __name__ == "__main__":
    # Input JSON file path
    json_file_path = r"C:\Users\jason\OneDrive - Rice University\Rice_U\ELEC 576 Intro to ML\Final project\arxiv-metadata-oai-snapshot.json"
    
    # Output CSV file path
    output_csv_path = r"C:\Users\jason\OneDrive - Rice University\Rice_U\ELEC 576 Intro to ML\Final project\arxiv_subset.csv"
    
    # Run the extraction
    extract_arxiv_metadata(json_file_path, output_csv_path)
