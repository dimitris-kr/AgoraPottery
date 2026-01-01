import os
from dotenv import load_dotenv
from huggingface_hub import login, upload_folder, upload_file

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

        print("Uploading images finished successfully")
    elif upload == "features":

        BASE_PATH = "../../data/features2/tensors"
        TFIDF_REPO = "dimitriskr/agora_pottery_tfidf"
        VIT_REPO = "dimitriskr/agora_pottery_vit"

        VERSION = "v1"

        for filename in os.listdir(BASE_PATH):
            file_path = os.path.join(BASE_PATH, filename)

            if not os.path.isfile(file_path):
                continue

            if "test" in filename.lower():
                continue

            # Decide target repo
            if "tfidf" in filename.lower():
                repo_id = TFIDF_REPO
            elif "vit" in filename.lower():
                repo_id = VIT_REPO
            else:
                continue

            print(f"Uploading {filename} → {repo_id}")

            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=f"{VERSION}/{filename}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {filename} ({VERSION})"
            )

        print("Uploading feature sets finished successfully")
    else:
        print("No upload")