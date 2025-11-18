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
"""Dataset loading and filtering for CVE benchmark."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass
class CVERecord:
    """Single CVE repair record from dataset."""
    cve_id: str
    image_name: str
    work_dir: str  # Absolute path in container, e.g., /workspace/markdown-it
    problem_statement: str
    
    @property
    def problem_id(self) -> str:
        """Unique identifier for this problem."""
        return self.cve_id


def load_dataset(jsonl_path: Path, 
                 include_ids: Optional[Set[str]] = None,
                 exclude_ids: Optional[Set[str]] = None,
                 limit: Optional[int] = None) -> List[CVERecord]:
    """Load and filter CVE dataset from JSONL file.
    
    Args:
        jsonl_path: Path to dataset JSONL file
        include_ids: If provided, only include CVEs with these IDs
        exclude_ids: If provided, exclude CVEs with these IDs  
        limit: Maximum number of records to return
        
    Returns:
        List of CVERecord instances
        
    Raises:
        ValueError: If required fields are missing
        FileNotFoundError: If dataset file doesn't exist
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
    
    records = []
    seen_ids = set()
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue
                
            # Validate required fields
            required_fields = ['cve_id', 'image_name', 'work_dir', 'problem_statement']
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                logging.warning(f"Skipping record at line {line_num}, missing fields: {missing_fields}")
                continue
                
            cve_id = data['cve_id']
            
            # Apply filters
            if include_ids and cve_id not in include_ids:
                continue
            if exclude_ids and cve_id in exclude_ids:
                continue
                
            # Check for duplicates
            if cve_id in seen_ids:
                logging.warning(f"Duplicate CVE ID {cve_id} at line {line_num}, skipping")
                continue
            seen_ids.add(cve_id)
            
            # Validate work_dir is absolute path
            work_dir = data['work_dir']
            if not work_dir.startswith('/'):
                logging.warning(f"work_dir must be absolute path, got: {work_dir}")
                continue
            
            record = CVERecord(
                cve_id=cve_id,
                image_name=data['image_name'],
                work_dir=work_dir,
                problem_statement=data['problem_statement']
            )
            records.append(record)
            
            # Apply limit
            if limit and len(records) >= limit:
                break
    
    logging.info(f"Loaded {len(records)} CVE records from {jsonl_path}")
    return records


def filter_existing_results(records: List[CVERecord], 
                           workspace_root: Path) -> List[CVERecord]:
    """Filter out records that already have complete results.
    
    Args:
        records: List of CVE records
        workspace_root: Base workspace directory
        
    Returns:
        Filtered list excluding records with existing results
    """
    filtered = []
    
    for record in records:
        result_dir = workspace_root / record.problem_id / "rollout_1"
        patch_file = result_dir / "outputs" / "patches" / f"{record.problem_id}.patch"
        
        if patch_file.exists():
            logging.info(f"Skipping {record.problem_id}, already has results")
        else:
            filtered.append(record)
    
    logging.info(f"Filtered to {len(filtered)} records without existing results")
    return filtered