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
"""Patch file generation and handling for CVE benchmark."""
import logging
import subprocess
from pathlib import Path
from typing import Optional


def write_patch_file(patch_content: str,
                    patch_file_path: Path) -> None:
    """Write patch content to file.
    
    Args:
        patch_content: Patch content to write
        patch_file_path: Path where patch should be written
    """
    # Ensure parent directory exists
    patch_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write patch content
    with open(patch_file_path, 'w', encoding='utf-8') as f:
        f.write(patch_content)
    
    logging.info(f"Wrote patch file: {patch_file_path} ({len(patch_content)} chars)")


def validate_patch(patch_content: str, relaxed: bool = True) -> bool:
    """Basic validation of patch content.
    
    Args:
        patch_content: Patch content to validate
        relaxed: If True, allow non-git-diff patches as valid
        
    Returns:
        True if patch appears valid, False otherwise
    """
    if not patch_content or not patch_content.strip():
        logging.info("Empty patch content")
        return True  # Empty patches are valid (no changes)
    
    patch_content = patch_content.strip()
    logging.info(f"Validating patch with {len(patch_content)} characters")
    logging.debug(f"Patch preview: {patch_content[:200]}...")
    
    # Check for basic patch format indicators
    patch_indicators = [
        "diff --git",
        "index ", 
        "@@",
        "+++",
        "---"
    ]
    
    has_indicators = any(indicator in patch_content for indicator in patch_indicators)
    
    if not has_indicators:
        if relaxed:
            # In relaxed mode, accept any non-empty content as potential patch
            logging.warning("Patch content does not appear to be in git diff format, but accepting in relaxed mode")
            return True
        else:
            logging.warning("Patch content does not appear to be in git diff format")
            return False
    
    logging.info("Patch validation passed")
    return True


def apply_patch_check(patch_content: str,
                     repo_path: Path) -> bool:
    """Test if patch can be applied cleanly (dry run).
    
    Args:
        patch_content: Patch content to test
        repo_path: Path to repository
        
    Returns:
        True if patch applies cleanly, False otherwise
    """
    if not patch_content or not patch_content.strip():
        return True  # Empty patch always applies
    
    try:
        # Test patch application with --check flag (dry run)
        result = subprocess.run([
            "git", "apply", "--check", "--verbose", "-"
        ], cwd=repo_path, input=patch_content, text=True, 
          capture_output=True, timeout=30)
        
        if result.returncode == 0:
            logging.info("Patch applies cleanly")
            return True
        else:
            logging.warning(f"Patch does not apply cleanly: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.warning("Patch apply check timed out")
        return False
    except Exception as e:
        logging.warning(f"Patch apply check failed: {e}")
        return False


def get_patch_stats(patch_content: str) -> dict:
    """Get statistics about patch content.
    
    Args:
        patch_content: Patch content to analyze
        
    Returns:
        Dictionary with patch statistics
    """
    if not patch_content or not patch_content.strip():
        return {
            "files_changed": 0,
            "lines_added": 0,
            "lines_removed": 0,
            "total_changes": 0
        }
    
    lines = patch_content.split('\n')
    
    files_changed = len([line for line in lines if line.startswith('diff --git')])
    lines_added = len([line for line in lines if line.startswith('+')])
    lines_removed = len([line for line in lines if line.startswith('-')])
    
    # Subtract header lines that start with +++ and ---
    lines_added -= len([line for line in lines if line.startswith('+++')])
    lines_removed -= len([line for line in lines if line.startswith('---')])
    
    return {
        "files_changed": files_changed,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "total_changes": lines_added + lines_removed
    }