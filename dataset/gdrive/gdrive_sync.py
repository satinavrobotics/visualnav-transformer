import os
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
from tqdm import tqdm

# Constants
CHUNK_SIZE = 16 * 1024 * 1024  # 4MB chunks for faster download

def get_drive_service():
    """
    Authenticate using Application Default Credentials (ADC) and return a Google Drive service object.
    """
    creds, _ = default(scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=creds)

def get_folder_name(service, folder_id):
    """
    Retrieve the folder name given its ID.
    """
    folder = service.files().get(fileId=folder_id, fields="name").execute()
    return folder["name"]

def get_file_content(service, file_id):
    """
    Download and return the content of a file from Google Drive as text.
    """
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    return fh.getvalue().decode('utf-8')

def list_drive_items(service, parent_id):
    """
    List all items (files, folders, shortcuts) under a given parent folder.
    We include shortcutDetails so we can see if a shortcut points to a folder or file.
    """
    query = f"'{parent_id}' in parents and trashed=false"
    items = []
    page_token = None

    while True:
        response = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, shortcutDetails)",
            pageToken=page_token
        ).execute()
        new_items = response.get("files", [])
        items.extend(new_items)
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return items

def list_files_in_folder(folder_id):
    """
    List files in a given Drive folder. A new service instance is used here.
    Ensures pagination is handled correctly and results are sorted by name.
    """
    service = get_drive_service()
    files = []
    page_token = None

    try:
        while True:
            response = service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, size)",
                orderBy="name",
                pageSize=1000,
                pageToken=page_token
            ).execute()

            files.extend(response.get('files', []))
            page_token = response.get('nextPageToken')

            if not page_token:
                break

    except HttpError as e:
        print(f"Error listing folder {folder_id}: {e}")
        
    return files
            

def dataset_upload(local_root, drive_parent_id, max_workers=16, max_retries=3):
    """
    Uploads all files in local_root to the Google Drive folder identified by drive_parent_id,
    replicating the local folder structure (excluding the top-level directory).
    Uses concurrent uploads with retries and skips duplicates.
    """
    # Global service for folder queries and caching
    global_service = get_drive_service()
    folder_cache = {}

    def get_remote_folder_id(relative_path):
        """Ensure remote folder exists for the given relative path; use a cache to reduce API calls."""
        if relative_path in folder_cache:
            return folder_cache[relative_path]
        current_id = drive_parent_id
        if relative_path == ".":
            folder_cache[relative_path] = current_id
            return current_id
        parts = relative_path.split(os.sep)
        for i, part in enumerate(parts):
            key = os.path.join(*parts[:i+1])
            if key in folder_cache:
                current_id = folder_cache[key]
                continue
            query = f"'{current_id}' in parents and name='{part}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            response = global_service.files().list(q=query, fields="files(id)", pageSize=1).execute()
            files = response.get("files", [])
            if files:
                current_id = files[0]["id"]
            else:
                folder_metadata = {"name": part,
                                   "mimeType": "application/vnd.google-apps.folder",
                                   "parents": [current_id]}
                folder = global_service.files().create(body=folder_metadata, fields="id").execute()
                current_id = folder["id"]
            folder_cache[key] = current_id
        return current_id

    # Gather upload tasks: (local_file, remote_folder_id, file_name)
    upload_tasks = []
    for root, dirs, files in os.walk(local_root):
        rel_folder = os.path.relpath(root, local_root)
        remote_folder = drive_parent_id if rel_folder == "." else get_remote_folder_id(rel_folder)
        for file in files:
            local_file = os.path.join(root, file)
            upload_tasks.append((local_file, remote_folder, file))
    total_files = len(upload_tasks)
    print(f"Uploading {total_files} files to Drive...")

    from googleapiclient.http import MediaFileUpload

    def upload_single_file(task):
        local_file, remote_folder, file_name = task
        attempt = 0
        while attempt < max_retries:
            try:
                # Each thread gets its own service instance to avoid sharing issues
                service = get_drive_service()
                # Check for duplicate
                query = f"'{remote_folder}' in parents and name='{file_name}' and trashed=false"
                try:
                    existing = service.files().list(q=query, fields="files(id)", pageSize=1).execute()
                    if existing.get("files"):
                        print(f"Skipping duplicate: {local_file}")
                        return True
                except Exception as dup_err:
                    print(f"Warning: duplicate check failed for {local_file}: {dup_err}")
                # Prepare upload
                media = MediaFileUpload(local_file, resumable=True)
                file_metadata = {"name": file_name, "parents": [remote_folder]}
                service.files().create(body=file_metadata, media_body=media, fields="id").execute()
                print(f"Uploaded: {local_file}")
                return True
            except Exception as e:
                attempt += 1
                sleep_time = 2 ** attempt
                print(f"Error uploading {local_file} (attempt {attempt}): {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        print(f"Failed to upload {local_file} after {max_retries} attempts.")
        return False

    successes = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(upload_single_file, task) for task in upload_tasks]
        for future in tqdm(as_completed(futures), total=total_files, desc="Uploading files"):
            if future.result():
                successes += 1
    print(f"Uploaded {successes}/{total_files} files successfully.")
    
    
    
def sync_mlflow_artifacts(local_path, drive_folder_id):
    """
    Sync MLflow artifacts from Google Drive to a local path.
    """
    drive_service = get_drive_service()
    os.makedirs(local_path, exist_ok=True)

    # Get list of files from Google Drive folder
    files = list_drive_items(drive_service, drive_folder_id)

    # Download each file
    for file in files:
        download_file(drive_service, file["id"], file["name"], local_path)

    print(f"✅ Successfully synced artifacts from Drive folder {drive_folder_id} to {local_path}")
    

def download_file(file_id, local_path, retries=3):
    """
    Download a file from Google Drive and save it locally.
    A new service instance is created for thread-safety.
    """
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        service = get_drive_service()
        request = service.files().get_media(fileId=file_id)
        with open(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=CHUNK_SIZE)
            done = False
            while not done:
                status, done = downloader.next_chunk()
    except HttpError as e:
        if retries > 0:
            time.sleep(2)
            download_file(file_id, local_path, retries - 1)
        else:
            print(f"❌ HTTP error {e.resp.status} for {local_path}")
    except Exception as e:
        if retries > 0:
            time.sleep(2)
            download_file(file_id, local_path, retries - 1)
        else:
            print(f"❌ Error downloading {local_path}: {e}")

def dataset_download(parent_folder_id, local_root):
    """
    Download dataset images using metadata from a JSON file named 
    'huron_dataset_metadata.json' located in the given parent folder.
    
    Parameters:
      parent_folder_id (str): The Google Drive folder ID where the metadata file is located.
      local_root (str): Local directory to download images into.
    """
    service = get_drive_service()
    
    # Search for the metadata JSON file in the parent folder
    response = service.files().list(
        q=f"'{parent_folder_id}' in parents and trashed=false and name='huron_dataset_metadata.json'",
        fields="files(id, name)",
        pageSize=1
    ).execute()
    
    files = response.get("files", [])
    if not files:
        print("Metadata file 'huron_dataset_metadata.json' not found in the specified folder.")
        return
    
    metadata_file_id = files[0]["id"]
    metadata_content = get_file_content(service, metadata_file_id)
    metadata = json.loads(metadata_content)

    # List files in each trajectory concurrently
    trajectory_files = []  # List of (file_id, local_path) tuples
    trajectories = metadata.get("trajectories", [])
    print(f"Found {len(trajectories)} trajectory folders in metadata.")

    def process_trajectory(trajectory):
        files_list = []
        if trajectory["image_count"] <= 0:
            return files_list
        traj_local_path = os.path.join(local_root, trajectory["path"])
        os.makedirs(traj_local_path, exist_ok=True)
        files = list_files_in_folder(trajectory["id"])
        for f in files:
            local_file_path = os.path.join(traj_local_path, f["name"])
            # Only add the file if it doesn't exist or is empty.
            if not os.path.exists(local_file_path) or os.path.getsize(local_file_path) == 0:
                files_list.append((f["id"], local_file_path))
            else:
                print(f"Skipping already downloaded file: {local_file_path}")
        return files_list

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_traj = {executor.submit(process_trajectory, traj): traj for traj in trajectories}
        for future in as_completed(future_to_traj):
            trajectory_files.extend(future.result())

    total_files = len(trajectory_files)
    print(f"Total files to download: {total_files}")

    if total_files == 0:
        print("All files already downloaded.")
        return

    # Now download files concurrently
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=16) as executor:
        pbar = tqdm(total=total_files, desc="Downloading files")
        futures = [executor.submit(download_file, file_id, local_path) for file_id, local_path in trajectory_files]
        for _ in as_completed(futures):
            pbar.update(1)
        pbar.close()

    print("✅ Download completed successfully!")
    

def index_drive_folder(service, folder_id, path=""):
    """
    Recursively index a folder (or a folder shortcut) in Google Drive.
    Returns a dictionary representing the folder hierarchy.
    """
    if not path:
        folder_name = get_folder_name(service, folder_id)
        path = folder_name
    else:
        folder_name = os.path.basename(path)

    folder_meta = {
        "id": folder_id,
        "name": folder_name,
        "path": path,
        "subfolders": [],
        "files": []
    }

    items = list_drive_items(service, folder_id)
    for item in items:
        mime = item["mimeType"]
        if mime == "application/vnd.google-apps.folder":
            sub_path = os.path.join(path, item["name"])
            sub_meta = index_drive_folder(service, item["id"], sub_path)
            folder_meta["subfolders"].append(sub_meta)
        elif mime == "application/vnd.google-apps.shortcut":
            shortcut_info = item.get("shortcutDetails", {})
            target_id = shortcut_info.get("targetId")
            target_mime = shortcut_info.get("targetMimeType")
            if target_mime == "application/vnd.google-apps.folder":
                sub_path = os.path.join(path, item["name"])
                sub_meta = index_drive_folder(service, target_id, sub_path)
                folder_meta["subfolders"].append(sub_meta)
            else:
                folder_meta["files"].append(item)
        else:
            folder_meta["files"].append(item)
    return folder_meta

def add_leaf_metadata(folder_meta):
    """
    If a folder is a leaf (has no subfolders), count the number of image files
    based on common image extensions and store the count in 'image_count'.
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
    if not folder_meta.get("subfolders"):
        count = sum(1 for f in folder_meta.get("files", [])
                    if os.path.splitext(f["name"].lower())[1] in image_extensions)
        folder_meta["image_count"] = count
    else:
        for sub in folder_meta["subfolders"]:
            add_leaf_metadata(sub)


def count_total_folders(folder_meta):
    """
    Recursively count total number of folders in the hierarchy.
    """
    count = 1  # count current folder
    for sub in folder_meta.get("subfolders", []):
        count += count_total_folders(sub)
    return count

def collect_trajectory_folders(folder_meta):
    """
    Return a list of leaf folders (assumed trajectories) with id, path, and image_count.
    Only includes folders with image_count > 0.
    """
    trajectories = []
    if not folder_meta.get("subfolders"):
        if folder_meta.get("image_count", 0) > 0:
            trajectories.append({
                "id": folder_meta["id"],
                "path": folder_meta["path"],
                "image_count": folder_meta.get("image_count", 0)
            })
    else:
        for sub in folder_meta["subfolders"]:
            trajectories.extend(collect_trajectory_folders(sub))
    return trajectories

def extract_datasets_info(metadata):
    """
    Extract immediate dataset folder information from the parent folder metadata.
    For each dataset folder, include its id and name.
    """
    datasets = []
    for dataset in metadata.get("subfolders", []):
        datasets.append({
            "id": dataset["id"],
            "name": dataset["name"]
        })
    return datasets

def upload_file_to_drive(service, local_file, drive_folder_id, file_name):
    """
    Upload a local file to the specified Drive folder.
    """
    media = MediaFileUpload(local_file, mimetype='application/json', resumable=True)
    file_metadata = {"name": file_name, "parents": [drive_folder_id]}
    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return uploaded_file["id"]

def metadata_main(drive_folder_id, output_path):
    """
    Generate metadata for a Google Drive folder and save it to a JSON file.
    The output includes:
      - dataset_parent_folder_id and parent_folder_name,
      - immediate dataset folders (id and name),
      - total folder counts and trajectories (leaf folders with id, path, image_count).
    """
    service = get_drive_service()
    parent_folder_name = get_folder_name(service, drive_folder_id)
    metadata = index_drive_folder(service, drive_folder_id)
    add_leaf_metadata(metadata)
    total_folders = count_total_folders(metadata)
    trajectories = collect_trajectory_folders(metadata)
    datasets = extract_datasets_info(metadata)

    summary = {
        "dataset_parent_folder_id": drive_folder_id,
        "parent_folder_name": parent_folder_name,
        "datasets": datasets,
        "total_folders": total_folders,
        "total_trajectory_folders": len(trajectories),
        "trajectories": trajectories,
    }

    output_file = os.path.join(output_path, "huron_dataset_metadata.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Metadata generated and written to {output_file}")
    
    # Upload the metadata file to the parent Drive folder
    drive_file_id = upload_file_to_drive(service, output_file, drive_folder_id, os.path.basename(output_file))
    print(f"✅ Metadata file also uploaded to Drive with file ID: {drive_file_id}")


if __name__ == "__main__":
    FOLDER_ID = "10q5pNVGx3c2HegH5RNsPbCGw1KqvhgGc"     # Sati_data folder ID
    test = "10PAqz6gz8EFzxeN0DYZzYHvcjQenqs-b"         # Test folder ID
    Stanford320240 = "18hq1Lg1EuFgbSKGf2i7htfiyDSFBZI-f" # stanford 
    output_path = "/app/Sati_data/Go-Stanford_320x240"
    MuSuHu_320x240 = "1RL9JNBO1pYoH8AoE3JsM9WumU8nMKuHL"
    LOCAL_FOLDER = "/app/Sati_data/MuSuHu_320x240"
    metadata_main(MuSuHu_320x240, LOCAL_FOLDER)

    # Sync the entire Drive folder to a local path
    Recon_folder_ID = "1bgBfBQdqZlnWJ-M166YkRSkGj6jAjeSL"     # Recon folder ID
    Recon_local_path = "/app/recon_processe"
    UPLODAD_FOLDER_ID = "10qjBXf9hg-J5AOpYar-21OpudkTA3h0k"     # Uplodad ID
    Scand_folder_ID = "1Yfh9aF0lMpI9OsnDMIWLxVbs0lEFyKnA"
    Huron_dataset = "10q5pNVGx3c2HegH5RNsPbCGw1KqvhgGc"
    
    test = "/app/Sati_data/test"
    UPLOAD_FOLDER = "/app/Sati_data/Huron_320x240"
    
    # dataset_upload(Recon_local_path, Recon_folder_ID)

    dataset_download(MuSuHu_320x240, LOCAL_FOLDER)
    