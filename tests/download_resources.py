import os
import shutil
import requests
import logging
import traceback
import zipfile


def download_resources(test_dir: str):
    try:
        test1_image_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsRADSLib-UnitTest1.zip'
        test2_image_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsRADSLib-UnitTest2.zip'
        test3_image_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsRADSLib-UnitTest3-Mediastinum.zip'
        dest_dir = os.path.join(test_dir, "patients")
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'unittest1_volume.zip')
        headers = {}
        response = requests.get(test1_image_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'unittest2_volume.zip')
        headers = {}
        response = requests.get(test2_image_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'unittest3_medi_volume.zip')
        headers = {}
        response = requests.get(test3_image_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

    except Exception as e:
        logging.error(f"Error during resources download with: {e} \n {traceback.format_exc()}.\n")
        shutil.rmtree(dest_dir)
        raise ValueError("Error during resources download.\n")
    try:
        seq_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_SequenceClassifier-v13.zip'
        brain_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_Brain-v13.zip'
        tumorcore_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_TumorCore-v13.zip'
        tumorce_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_TumorCE_Postop-v13.zip'
        flairchanges_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_FLAIRChanges-v13.zip'
        lungs_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-CT_Lungs-v13.zip'
        medi_tumor_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-CT_Tumor-v13.zip'
        dest_dir = os.path.join(test_dir, "models")
        os.makedirs(dest_dir, exist_ok=True)

        archive_dl_dest = os.path.join(dest_dir, 'brain_model.zip')
        headers = {}
        response = requests.get(brain_model_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'seq_model.zip')
        headers = {}
        response = requests.get(seq_model_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'tumorce_model.zip')
        headers = {}
        response = requests.get(tumorce_model_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'flairchanges_model.zip')
        headers = {}
        response = requests.get(flairchanges_model_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'tumorcore_model.zip')
        headers = {}
        response = requests.get(tumorcore_model_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'lungs_model.zip')
        headers = {}
        response = requests.get(lungs_model_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'medi_tumor_model.zip')
        headers = {}
        response = requests.get(medi_tumor_model_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

    except Exception as e:
        logging.error(f"Error during resources download with: \n {e}\n{traceback.format_exc()}")
        shutil.rmtree(dest_dir)
        raise ValueError("Error during resources download.\n")
    try:
        dest_dir = os.path.join(test_dir, "results")
        os.makedirs(dest_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Error during resources download with: \n {e}\n{traceback.format_exc()}")
        shutil.rmtree(dest_dir)
        raise ValueError("Error during resources download.\n")
    return dest_dir