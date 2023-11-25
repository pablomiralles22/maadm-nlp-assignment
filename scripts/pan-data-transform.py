import argparse
import json
import os
import re
import itertools


def save_example(target_path, cnt, text_1, text_2, label):
    example = {
        "text1": text_1,
        "text2": text_2,
        "label": label,
    }
    example_path = os.path.join(target_path, f"{cnt}.json")
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(example, f, ensure_ascii=False)

def files_to_dicts(source_path):
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

    return lines_by_id, truths_by_id

def transform_with_augmentation(source_path, target_path):
    lines_by_id, truths_by_id = files_to_dicts(source_path)

    cnt = 0
    for problem_id, lines in lines_by_id.items():
        truths = truths_by_id[problem_id]
        text_groups = [
            [lines[0]],
        ]  # group consecutive same author texts
        for text, label in zip(lines[1:], truths):
            if label == 1:
                text_groups.append([text])
            else:
                text_groups[-1].append(text)

        # load negative examples
        for group_1, group_2 in zip(text_groups, text_groups[1:]):
            for text_1, text_2 in itertools.product(group_1, group_2):
                save_example(target_path, cnt, text_1, text_2, 1)
                cnt += 1

        # load positive examples
        for group in text_groups:
            for text_1, text_2 in itertools.combinations(group, 2):
                save_example(target_path, cnt, text_1, text_2, 0)
                cnt += 1

def transform(source_path, target_path):
    lines_by_id, truths_by_id = files_to_dicts(source_path)

    cnt = 0
    for problem_id, lines in lines_by_id.items():
        truths = truths_by_id[problem_id]
        for text_1, text_2, label in zip(lines, lines[1:], truths):
            save_example(target_path, cnt, text_1, text_2, label)
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
            source_path = os.path.join(
                args.source_dir,
                f"pan23-multi-author-analysis-dataset{task}/pan23-multi-author-analysis-dataset{task}-{set_type}/",
            )
            target_path = os.path.join(args.target_dir, f"task{task}/{set_type}/")
            os.makedirs(target_path, exist_ok=True)
            transform_fn = transform_with_augmentation if set_type == "train" else transform
            transform_fn(source_path, target_path)


if __name__ == "__main__":
    main()
