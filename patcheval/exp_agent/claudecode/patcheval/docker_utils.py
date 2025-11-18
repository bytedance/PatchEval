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
"""Docker container management for CVE benchmark.

Adapted from augment-swebench-agent docker utilities to work with CVE docker images
and integrate with cline-cli container routing.
"""
import docker
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import threading


MAX_DOCKER_CONCURRENCY = 4
RETRY_DELAY_BASE = 2  # Base delay for exponential backoff
MAX_RETRIES = 3


def run_work_container_no_mount(image_name: str,
                              problem_id: str,
                              semaphore: Any,
                              modelname: str) -> str:
    """Start work container without any volume mounting.
    
    Args:
        image_name: Docker image name
        problem_id: Problem identifier for naming
        semaphore: Concurrency control semaphore
        
    Returns:
        Container ID of work container
        
    Raises:
        RuntimeError: If container creation fails
    """
    container_name = f"bench.{problem_id}.{modelname}.work"
    
    # Stop any existing work containers
    stop_container(container_name)
    
    client = docker.from_env()
    
    # Simple command to keep container alive
    command = "sleep 7200"  # Keep alive for 2 hours
    
    # Collect API keys from environment
    api_keys = {}
    cline_supported_keys = [
        "API_KEY", "ANTHROPIC_API_KEY", "DEEP_SEEK_API_KEY", "DEEPSEEK_API_KEY",
        "OPEN_AI_API_KEY", "GEMINI_API_KEY", "OPEN_AI_NATIVE_API_KEY",
        "OPEN_ROUTER_API_KEY", "CLINE_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY",
        "TOGETHER_API_KEY", "QWEN_API_KEY", "DOUBAO_API_KEY", "REQUESTY_API_KEY",
        "LITE_LLM_API_KEY", "ASKSAGE_API_KEY", "SAMBANOVA_API_KEY"
    ]
    
    for key in cline_supported_keys:
        value = os.getenv(key)
        if value:
            api_keys[key] = value
    with semaphore:
        logging.info(f"Starting container-native work container: {container_name}")
        try:
            container = client.containers.run(
                name=container_name,
                image=image_name,
                command=command,
                detach=True,
                remove=False,
                environment=api_keys,
                mem_limit="4g",
                cpu_quota=400000,  # 4 CPU cores
                extra_hosts={"host.docker.internal":"172.17.0.1"},
                volumes={
                    "claude_tgz": {
                        "bind": "/workspace/claude_tgz",
                        "mode": "rw"
                    }
                }
            )
            
            # Wait for container to stabilize
            time.sleep(2)
            
            # Verify container is running
            if container.status != "running":
                container.reload()
                if container.status != "running":
                    raise RuntimeError(f"Container {container_name} failed to start")
            
            logging.info(f"Container-native work container started: {container.id[:12]}")
            return container.id
            
        except Exception as e:
            logging.error(f"Failed to start container-native work container: {e}")
            raise RuntimeError(f"Container-native work container creation failed: {e}")


def pull_image_with_retry(image_name: str,
                         semaphore: Any,
                         max_retries: int = MAX_RETRIES) -> None:
    """Pull docker image with exponential backoff retry.
    
    Args:
        image_name: Docker image to pull
        semaphore: Concurrency control semaphore  
        max_retries: Maximum number of retry attempts
        
    Raises:
        RuntimeError: If all pull attempts fail
    """
    client = docker.from_env()
    
    for attempt in range(max_retries + 1):
        try:
            with semaphore:
                logging.info(f"Pulling image {image_name} (attempt {attempt + 1}/{max_retries + 1})")
                client.images.pull(image_name)
                logging.info(f"Successfully pulled {image_name}")
                return
                
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Failed to pull image {image_name} after {max_retries + 1} attempts: {e}")
            
            delay = RETRY_DELAY_BASE ** (attempt + 1)
            logging.warning(f"Pull attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


def stop_container(container_name: str, force: bool = False) -> None:
    """Stop and remove container by name.
    
    Args:
        container_name: Name or ID of container to stop
        force: If True, forcefully kill the container with shorter timeout
    """
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        
        timeout = 3 if force else 10  
        logging.info(f"Stopping container: {container_name} (timeout: {timeout}s)")
        

            
        container.remove(force=True)
        logging.info(f"Removed container: {container_name}")
        
    except docker.errors.NotFound:
        logging.debug(f"Container {container_name} not found")
    except Exception as e:
        logging.warning(f"Failed to stop container {container_name}: {e}")


def set_volume_permissions(mount_dir: Path, 
                          max_retries: int = 2) -> None:
    """Set appropriate permissions on volume mount directory.
    
    On macOS, this is usually not needed. On Linux, may require sudo.
    
    Args:
        mount_dir: Directory to fix permissions for
        max_retries: Maximum number of retry attempts
    """
    if not mount_dir.exists():
        logging.warning(f"Mount directory does not exist: {mount_dir}")
        return
        
    # Skip on macOS - usually not needed
    if os.uname().sysname == "Darwin":
        logging.debug("Skipping permission fix on macOS")
        return
    
    my_uid = os.getuid()
    my_gid = os.getgid()
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Fixing permissions for {mount_dir} to {my_uid}:{my_gid}")
            
            # Make parent directories readable
            subprocess.check_call([
                "sudo", "chmod", "a+rx"
            ] + [str(p) for p in mount_dir.parents])
            
            # Change ownership to current user
            subprocess.check_call([
                "sudo", "chown", "-R", f"{my_uid}:{my_gid}", str(mount_dir)
            ])
            
            logging.info("Permission fix completed")
            return
            
        except subprocess.CalledProcessError as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to fix permissions after {max_retries} attempts: {e}")
                raise RuntimeError(f"Permission fix failed: {e}")
            else:
                logging.warning(f"Permission fix attempt {attempt + 1} failed: {e}")
                time.sleep(5)


def execute_in_container(container_id: str, 
                        command: str,
                        work_dir: str = "/workspace",
                        timeout: int = 300) -> Tuple[int, str]:
    """Execute command in running container.
    
    Args:
        container_id: Container ID or name
        command: Command to execute
        work_dir: Working directory for command execution
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (exit_code, output)
        
    Raises:
        RuntimeError: If execution fails
    """
    try:
        client = docker.from_env()
        container = client.containers.get(container_id)
        
        logging.info(f"Executing in container {container_id[:12]}: {command}")
        
        # Execute command
        exec_result = container.exec_run(
            cmd=f"bash -c 'cd {work_dir} && {command}'",
            stdout=True,
            stderr=True,
            demux=False
        )
        
        exit_code = exec_result.exit_code
        output = exec_result.output.decode('utf-8', errors='replace') if exec_result.output else ""
        
        if exit_code == 0:
            logging.info(f"Command succeeded: {command}")
        else:
            logging.warning(f"Command failed (exit {exit_code}): {command}")
            
        return exit_code, output
        
    except Exception as e:
        logging.error(f"Failed to execute command in container: {e}")
        raise RuntimeError(f"Container command execution failed: {e}")


def cleanup_containers_by_prefix(prefix: str = "bench.") -> None:
    """Clean up containers with given name prefix.
    
    Args:
        prefix: Container name prefix to match
    """
    try:
        client = docker.from_env()
        containers = client.containers.list(all=True)
        
        for container in containers:
            if container.name and container.name.startswith(prefix):
                logging.info(f"Cleaning up container: {container.name}")
                try:
                    container.stop(timeout=10)
                    container.remove()
                except Exception as e:
                    logging.warning(f"Failed to cleanup {container.name}: {e}")
                    
    except Exception as e:
        logging.error(f"Failed to list containers for cleanup: {e}")