# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import docker
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def pull_one_image(image, ghcr_token=None, logger=None):
    try:
        client = docker.from_env()
        client.ping()

        low_api = docker.APIClient(base_url="unix://var/run/docker.sock")
        logger.info(f"[{image}] start pull")
        start_time = datetime.now()

        pull_log = low_api.pull(image, stream=True, decode=True)
        last_status = None
        for line in pull_log:
            status = line.get("status", "").strip()
            if status and status != last_status:
                logger.info(f"{image}: {status}")
                last_status = status

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{image}] pull finish ({elapsed:.2f}s)")
        return True, image
    except Exception as e:
        logger.error(f"[{image}] pull fail: {e}")
        return False, image

def batch_pull_github_images(
    images_file="images.txt",
    log_file="pull.log",
    ghcr_token=None,
    max_workers=4   
):
    """
    Multi-threaded pull images
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger("ghcr_puller")

    if not os.path.exists(images_file):
        logger.error(f"Image list file {images_file} does not exist!")
        return 0, 0

    try:
        with open(images_file, "r") as f:
            images = [line.strip() for line in f if line.strip()]
        if not images:
            logger.warning("Image list is empty!")
            return 0, 0
        logger.info(f"Read {len(images)} images from {images_file}")
    except Exception as e:
        logger.error(f"Read image list file {images_file} fail: {e}")
        return 0, 0

    # Test Docker connection
    try:
        client = docker.from_env()
        client.ping()
        logger.info("Docker connection success")
    except Exception as e:
        logger.error(f"Docker connection fail: {e}")
        return 0, 0

    success, fail = 0, 0

    # Multi-threaded pull images
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(pull_one_image, image, ghcr_token, logger)
            for image in images
        ]
        for future in as_completed(futures):
            ok, image = future.result()
            if ok:
                success += 1
            else:
                fail += 1

    logger.info(f"Pull images finish, success: {success}, fail: {fail}")
    return success, fail

if __name__ == "__main__":
    batch_pull_github_images(
        images_file="images.txt",
        log_file="pull_images.log",
        ghcr_token="",  
        max_workers=16,  
    )