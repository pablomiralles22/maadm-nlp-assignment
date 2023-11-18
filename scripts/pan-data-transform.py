import argparse
import json
import os
import re


def transform(source_path, target_path):
    filenames = os.listdir(source_path)

    lines_by_id = dict()
    truths_by_id = dict()

    for filename in filenames:
        is_truth = filename.startswith("truth")
        problem_id = re.search(r"\d+", filename).group(0)
        filepath = os.path.join(source_path, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            if is_truth is True:
                truths_by_id[problem_id] = json.load(f)["changes"]
            else:
                lines_by_id[problem_id] = f.readlines()

    cnt = 0
    for problem_id, lines in lines_by_id.items():
        truths = truths_by_id[problem_id]
        for text1, text2, label in zip(lines, lines[1:], truths):
            example = {
                "text1": text1,
                "text2": text2,
                "label": label,
            }
            example_path = os.path.join(target_path, f"{cnt}.json")
            with open(example_path, "w", encoding="utf-8") as f:
                json.dump(example, f, ensure_ascii=False)
            cnt += 1



def main():
    parser = argparse.ArgumentParser(
        description="PAN-23 Data Transformation Tool",
        epilog=(
            "This tool helps in transforming data for PAN-23 by taking original "
            "data from a specified directory and saving the transformed data "
            "into a different directory."
        ),
    )

    # Argument for the directory of the original data
    parser.add_argument(
        "--source-dir",
        type=str,
        help="Path to the directory where the original data is stored.",
    )

    # Argument for the directory to store the transformed data
    parser.add_argument(
        "--target-dir",
        type=str,
        help="Path to the directory where the transformed data will be stored.",
    )

    args = parser.parse_args()
    for task in range(1, 4):
        for set_type in ["train", "validation"]:
            source_path = os.path.join(args.source_dir, f"pan23-multi-author-analysis-dataset{task}/pan23-multi-author-analysis-dataset{task}-{set_type}/")
            target_path = os.path.join(args.target_dir, f"pan23-task{task}-{set_type}/")
            os.makedirs(target_path, exist_ok=True)
            transform(source_path, target_path)


if __name__ == "__main__":
    main()
