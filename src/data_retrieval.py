from math import isclose
import os
from azure.storage.blob import BlobServiceClient, ContainerClient, StorageStreamDownloader
from dotenv import load_dotenv
from pathlib import Path
from config.definitions import DOTENV_PATH, ROOT_DIRECTORY

def download_preprocessed_data(mode="mean"):
    try:
        load_dotenv(DOTENV_PATH)
        connect_str = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')

        # Create the BlobServiceClient object which will be used to create a container client
        blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(connect_str)
        container_name = "amex"
        container_client: ContainerClient = blob_service_client.get_container_client(container=container_name) 

        # Create a local path if not already exist
        abs_path = Path.joinpath(ROOT_DIRECTORY, "dataset")

        if mode == "mean":
            local_file_name = "train_data_mean.csv"
            download_file_path = Path.joinpath(abs_path, local_file_name)
            blob_name = "fillmeanmode.csv"
        else:
            local_file_name = "junk.csv"
            download_file_path = Path.joinpath(abs_path, local_file_name)
            blob_name = "junk.csv"
        
        # make sure that the directory 'dataset' exist
        download_file_path.parent.mkdir(parents=True, exist_ok=True)

        # if the file exists, delete it
        if download_file_path.is_file():
            os.remove(download_file_path)
        
        # download the blob and write the content to a local file
        segment_size = 1 * 1024 * 1024 # 1 MB 
        blob_client = container_client.get_blob_client(blob_name)
        blob_properties = blob_client.get_blob_properties()
        blob_length_remaining = blob_properties["size"]
        start = 0
        no_MiB = 0
        while blob_length_remaining > 0:
            blockSize = min(blob_length_remaining, segment_size)
            stream: StorageStreamDownloader = blob_client.download_blob(start, blockSize)
            with open(download_file_path, "ab") as download_file:
                download_file.write(stream.readall())
            print_download_progress(start + blockSize, blob_properties["size"], no_MiB)
            start += blockSize
            blob_length_remaining -= blockSize
            no_MiB += 1

    except Exception as ex:
        print('Exception:')
        print(ex)


def print_download_progress(current_size, max_size, no_MiB):
    progress = (current_size / max_size) * 100

    if no_MiB % 1000 == 0:
        print(f"Downloaded {no_MiB // 1000} GB of data. {progress: .2f}% completed.")
