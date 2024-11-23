# Check large files

#%%

import os

def find_large_files(start_path='.', size_threshold_mb=96):
    size_threshold = size_threshold_mb * 1024 * 1024
    for dirpath, dirnames, filenames in os.walk(start_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                file_size = os.path.getsize(file_path)
                if file_size >= size_threshold:
                    print(f"{file_path} - {file_size / (1024 * 1024):.2f} MB")
            except (FileNotFoundError, PermissionError):
                continue

if __name__ == "__main__":
    find_large_files()

# %%
