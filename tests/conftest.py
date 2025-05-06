import os
import shutil
import pytest
import logging
from tests.download_resources import download_resources


@pytest.fixture(scope="session")
def test_dir():
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.listdir(test_dir):
        download_resources(test_dir=test_dir)
    yield test_dir
    logging.info(f"Removing the temporary directory for tests.")
    shutil.rmtree(test_dir)

# @pytest.fixture(scope="session")
# def input_data_dir(temp_dir):
#     try:
#         test1_image_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsRADSLib-UnitTest1-v13.zip'
#         test2_image_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsRADSLib-UnitTest2-v13.zip'
#         dest_dir = os.path.join(temp_dir, "patients")
#         if os.path.exists(dest_dir):
#             shutil.rmtree(dest_dir)
#         os.makedirs(dest_dir)
#
#         archive_dl_dest = os.path.join(dest_dir, 'unittest1_volume.zip')
#         headers = {}
#         response = requests.get(test1_image_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#
#         archive_dl_dest = os.path.join(dest_dir, 'unittest2_volume.zip')
#         headers = {}
#         response = requests.get(test2_image_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#
#     except Exception as e:
#         logging.error("Error during resources download with: \n {}.\n".format(traceback.format_exc()))
#         shutil.rmtree(dest_dir)
#         raise ValueError("Error during resources download.\n")
#     return dest_dir
#
# @pytest.fixture(scope="session")
# def input_models_dir(temp_dir):
#     try:
#         seq_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_SequenceClassifier-v13.zip'
#         brain_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_Brain-v13.zip'
#         tumorce_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_TumorCE_Postop-v13.zip'
#         dest_dir = os.path.join(temp_dir, "models")
#         os.makedirs(dest_dir, exist_ok=True)
#
#         archive_dl_dest = os.path.join(dest_dir, 'brain_model.zip')
#         headers = {}
#         response = requests.get(brain_model_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#
#         archive_dl_dest = os.path.join(dest_dir, 'seq_model.zip')
#         headers = {}
#         response = requests.get(seq_model_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#
#         archive_dl_dest = os.path.join(dest_dir, 'tumorce_model.zip')
#         headers = {}
#         response = requests.get(tumorce_model_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#
#     except Exception as e:
#         logging.error(f"Error during resources download with: \n {e}\n{traceback.format_exc()}")
#         shutil.rmtree(dest_dir)
#         raise ValueError("Error during resources download.\n")
#     return dest_dir
#
# @pytest.fixture(scope="session")
# def results_dir(temp_dir):
#     try:
#         dest_dir = os.path.join(temp_dir, "results")
#         os.makedirs(dest_dir, exist_ok=True)
#     except Exception as e:
#         logging.error(f"Error during resources download with: \n {e}\n{traceback.format_exc()}")
#         shutil.rmtree(dest_dir)
#         raise ValueError("Error during resources download.\n")
#     return dest_dir