import json

def print_summary(run_path, run_name):
    json_file = json.load(run_path / f"{run_name}.json")
    print("RUN SUMMARY")
    print(json_file)