import os
import json
import shutil
import requests
import logging
import traceback
import zipfile


"""
The archives are only downloaded again when they changed remotely (or their URL changed), by comparing the ETag (or the
Last-Modified date and size when no ETag is given) of each release asset with the one stored in resources_manifest.json
when it was last downloaded. The folders and files extracted from each archive are also listed in the manifest, so that
an archive is downloaded again if any of them was removed, and so that they are removed before extracting a new version.
Without network access, the resources already on disk are used as they are.
"""
RESOURCES = {"patients": [['https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsRADSLib-UnitTest1-v1.3.3.zip', 'unittest1_volume.zip'],
                          ['https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsRADSLib-UnitTest2-v1.3.1.zip', 'unittest2_volume.zip'],
                          ['https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsRADSLib-UnitTest3-Mediastinum-v1.3.1.zip', 'unittest3_medi_volume.zip']],
             "models": [['https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_Brain-v13.zip', 'brain_model.zip'],
                        ['https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_SequenceClassifier-v13.zip', 'seq_model.zip'],
                        ['https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_TumorCE_Postop-v13.zip', 'tumorce_model.zip'],
                        ['https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_FLAIRChanges-v13.zip', 'flairchanges_model.zip'],
                        ['https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_TumorCore-v13.zip', 'tumorcore_model.zip'],
                        ['https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-CT_Lungs-v13.zip', 'lungs_model.zip'],
                        ['https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-CT_Tumor-v13.zip', 'medi_tumor_model.zip']]}


def get_remote_version(url: str) -> str:
    """
    Identifier of the current version of the remote file, from the headers only (the release assets are redirected,
    the headers of the final location are used).
    """
    response = requests.head(url, allow_redirects=True, timeout=30)
    response.raise_for_status()
    if "ETag" in response.headers.keys():
        return response.headers["ETag"]
    last_modified = ""
    if "Last-Modified" in response.headers.keys():
        last_modified = response.headers["Last-Modified"]
    content_length = ""
    if "Content-Length" in response.headers.keys():
        content_length = response.headers["Content-Length"]
    return last_modified + "_" + content_length


def is_up_to_date(manifest: dict, url: str, remote_version: str, dest_dir: str) -> bool:
    if url not in manifest.keys():
        return False
    if remote_version is not None and manifest[url]["version"] != remote_version:
        return False
    for name in manifest[url]["extracted"]:
        if not os.path.exists(os.path.join(dest_dir, name)):
            return False
    return True


def download_resources(test_dir: str):
    manifest_filename = os.path.join(test_dir, "resources_manifest.json")
    manifest = {}
    if os.path.exists(manifest_filename):
        try:
            with open(manifest_filename, 'r') as infile:
                manifest = json.load(infile)
        except Exception as e:
            logging.warning(f"Could not read the resources manifest, downloading everything again ({e}).")
            manifest = {}

    for folder in RESOURCES.keys():
        dest_dir = os.path.join(test_dir, folder)
        os.makedirs(dest_dir, exist_ok=True)
        for url, archive_name in RESOURCES[folder]:
            try:
                remote_version = None
                try:
                    remote_version = get_remote_version(url=url)
                except Exception as e:
                    # No network access, the resources already on disk are used as they are
                    if is_up_to_date(manifest=manifest, url=url, remote_version=None, dest_dir=dest_dir):
                        logging.warning(f"Could not check the remote version of {url}, using the local copy ({e}).")
                        continue
                    raise ValueError(f"Could not reach {url} with {e}")

                if is_up_to_date(manifest=manifest, url=url, remote_version=remote_version, dest_dir=dest_dir):
                    logging.info(f"Test resource {archive_name} already up to date.")
                    continue

                # Removing what was extracted from the previous version of the archive
                if url in manifest.keys():
                    for name in manifest[url]["extracted"]:
                        previous_fn = os.path.join(dest_dir, name)
                        if os.path.isdir(previous_fn):
                            shutil.rmtree(previous_fn)
                        elif os.path.exists(previous_fn):
                            os.remove(previous_fn)
                    del manifest[url]

                logging.info(f"Downloading test resource {archive_name}.")
                archive_dl_dest = os.path.join(dest_dir, archive_name)
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(archive_dl_dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1048576):
                        f.write(chunk)
                extracted = []
                with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        top_name = member.split('/')[0]
                        if top_name != "" and top_name not in extracted:
                            extracted.append(top_name)
                    zip_ref.extractall(dest_dir)
                os.remove(archive_dl_dest)

                manifest[url] = {"version": remote_version, "extracted": extracted}
                # Saved after each archive, so that an interrupted download does not invalidate the others
                with open(manifest_filename, 'w') as outfile:
                    json.dump(manifest, outfile, indent=4)
            except Exception as e:
                logging.error(f"Error during resources download with: {e} \n {traceback.format_exc()}.\n")
                archive_dl_dest = os.path.join(dest_dir, archive_name)
                if os.path.exists(archive_dl_dest):
                    os.remove(archive_dl_dest)
                raise ValueError("Error during resources download.\n")

    """
    The results folder is emptied at each start, the downloaded resources are kept.
    """
    dest_dir = os.path.join(test_dir, "results")
    try:
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)
    except Exception as e:
        logging.error(f"Error during results folder setup with: \n {e}\n{traceback.format_exc()}")
        raise ValueError("Error during results folder setup.\n")
    return dest_dir
