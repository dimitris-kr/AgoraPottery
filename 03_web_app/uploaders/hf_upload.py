import os
from dotenv import load_dotenv
from huggingface_hub import login, upload_folder

load_dotenv()

if __name__ == "__main__":
    # Login with Hugging Face credentials
    login(token=os.getenv("HF_TOKEN"))

    upload = input("Upload:")

    if upload == "images":
        # Push Image Dataset
        upload_folder(
            repo_id="dimitriskr/agora_pottery_images",
            folder_path="../../data/images",
            path_in_repo="images",
            repo_type="dataset"
        )

    if upload in ["images"]:
        print("Uploading images finished successfully")
    else:
        print("No upload")