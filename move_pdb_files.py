import os
import shutil

source_dir = "./profile_dataset/pdb_others"  # Change this to your source directory
dest_dir = "./profile_dataset/pdb_files"  # Change this to your destination directory
num_files = 1000


# Get a list of files in the source directory
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Move the first 1000 files
for file in files[:1000]:
    shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))

print(f"Moved {len(files[:1000])} files from {source_dir} to {dest_dir}")

num_files = len([f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))])
print(f"Number of files in {source_dir}: {num_files}")
num_files = len([f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))])
print(f"Number of files in {dest_dir}: {num_files}")

