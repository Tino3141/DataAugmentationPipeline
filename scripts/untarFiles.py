import argparse
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed

def unpack(file_path: str) -> None:
    if file_path.endswith(".tar"):
        dest = os.path.dirname(file_path)
        print(f"Unpacking {file_path}...")
        with tarfile.open(file_path, "r") as tar:
            tar.extractall(path=dest)
        os.remove(file_path)  # Remove the tar file after unpacking
        print(f"Unpacked {file_path} to {dest}")
    else:
        print(f"Skipping {file_path}, not a tar file.")

def main(args: argparse.Namespace) -> None:
    # Read list of files
    with open(args.filesList, "r") as f:
        files = [line.strip() for line in f if line.strip()]

    # Launch unpacking in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Option A: simple map
        # executor.map(unpack, files)

        # Option B: to catch and display exceptions as they occur
        futures = {executor.submit(unpack, fp): fp for fp in files}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error unpacking {futures[future]}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unpack tar files in parallel.")
    parser.add_argument(
        "--filesList",
        type=str,
        required=True,
        help="Path to a text file listing .tar files, one per line."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel worker processes."
    )
    args = parser.parse_args()
    main(args)