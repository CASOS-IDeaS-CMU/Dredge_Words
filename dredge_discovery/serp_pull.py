import WebSearcher as ws
import pandas as pd
import tqdm
import time
import numpy as np
import os
import argparse

se = ws.SearchEngine()

def search(query, output_folder):
    se = ws.SearchEngine()
    se.search(query)
    se.parse_results()
    out = pd.DataFrame(se.results)
    out['qry'] = query
    # time.sleep(10 + np.random.uniform(high = 10, low=1))
    out.to_csv(f'{output_folder}/qry_{query}.csv', index=False)

def get_serp(query, output_folder, wait_time=1):
    try:
        search(query, output_folder)
    except:
        # exponential backoff
        if wait_time < 10:
            print(f'Timeout on {query}, retrying in {wait_time} seconds')
            time.sleep(wait_time)
            get_serp(query, output_folder, wait_time=wait_time * 2)
        else:
            print(f'Timeout on {query}, skipping')

def main(input_folder, output_folder):
    for file in tqdm.tqdm(os.listdir(input_folder)):
        queries = pd.read_csv(os.path.join(input_folder, file), on_bad_lines='skip', encoding='latin1')['Keyword']
        queries.drop_duplicates(inplace=True)

        for query in queries.to_list():
            get_serp(query, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_folder', type=str, help='The input folder containing keyword files')
    parser.add_argument('output_folder', type=str, help='The output folder to save results')

    args = parser.parse_args()
    main(args.input_folder, args.output_folder)