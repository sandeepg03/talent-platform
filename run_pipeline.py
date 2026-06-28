import os
import subprocess
import time


def main():
    print("Orchestrator: Waiting for precompute artifacts to be ready...")

    # We poll for the existence of the FAISS index and candidate IDs npy
    faiss_path = os.path.join("artifacts", "faiss.index")
    ids_path = os.path.join("artifacts", "candidate_ids.npy")
    embeddings_path = os.path.join("artifacts", "embeddings.npy")

    attempts = 0
    max_attempts = 120  # 120 * 15s = 30 minutes

    while not (
        os.path.exists(faiss_path) and os.path.exists(ids_path) and os.path.exists(embeddings_path)
    ):
        attempts += 1
        if attempts > max_attempts:
            print("Error: Precomputation timed out (took >30 minutes).")
            return
        time.sleep(15)

    print("Orchestrator: Precompute artifacts detected! Launching ranker...")

    # Step 1: Run ranker to generate submission.csv
    rank_cmd = [
        "python",
        "-u",
        "rank.py",
        "--jd",
        "data/job_description.docx",
        "--candidates",
        "data/candidates.jsonl",
        "--artifacts",
        "artifacts",
        "--output",
        "submission.csv",
    ]

    print(f"Running: {' '.join(rank_cmd)}")
    result = subprocess.run(rank_cmd, capture_output=True, text=True)
    print("Ranker stdout:")
    print(result.stdout)
    print("Ranker stderr:")
    print(result.stderr)

    if result.returncode != 0:
        print("Error: Ranker execution failed.")
        return

    # Step 2: Run format converter to generate submission.xlsx
    convert_cmd = ["python", "convert_submission.py"]
    print(f"Running: {' '.join(convert_cmd)}")
    result_convert = subprocess.run(convert_cmd, capture_output=True, text=True)
    print(result_convert.stdout)

    if result_convert.returncode != 0:
        print("Error: XLSX conversion failed.")
        return

    print("Orchestrator: Pipeline complete! Ranked candidate XLSX file is ready.")


if __name__ == "__main__":
    main()
