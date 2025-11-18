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
"""Command-line interface for Claude Code CVE benchmark."""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Set
from .batch_runner import run_batch_cves
from .single_runner import run_single_cve
from .docker_utils import cleanup_containers_by_prefix
from .dataset import load_dataset


def get_available_strategies() -> list[str]:
    
    templates_dir = Path(__file__).parent.parent / "templates"
    if not templates_dir.exists():
        return ["iterative", "smart"]  
    
    strategies = []
    for template_file in templates_dir.glob("*.md"):
        strategy_name = template_file.stem
        strategies.append(strategy_name)
    
    return strategies if strategies else ["iterative", "smart"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("--dataset", type=Path, default="./dataset.jsonl")
    batch_parser.add_argument("--outputs-root", type=Path, default="./outputs")
    batch_parser.add_argument("--max-workers", type=int, default=1)
    batch_parser.add_argument("--timeout", type=str, default="45m")
    batch_parser.add_argument("--claude-timeout", type=str, default="30m")
    batch_parser.add_argument("--limit", type=int)
    batch_parser.add_argument("--resume", action="store_true")
    batch_parser.add_argument("--keep-containers", action="store_true")
    batch_parser.add_argument("--strategy", choices=get_available_strategies(), default="iterative")
    batch_parser.add_argument("--api-provider", choices=["anthropic", "bedrock", "vertex"], default="anthropic")
    

    batch_parser.add_argument("--tool-limits", type=str, help="(tool1:limit1,tool2:limit2 or total:500)")
    batch_parser.add_argument("--max-cost-usd", type=float, default=10.0)
    batch_parser.add_argument("--enable-detailed-logging", action="store_true", default=True,)
    batch_parser.add_argument("--save-process-logs", action="store_true")
    batch_parser.add_argument("--allow-git-diff-fallback", action="store_true")
    batch_parser.add_argument("--settings", type=str)
    batch_parser.add_argument("--port", type=str)
    
    # Single command  
    single_parser = subparsers.add_parser("single")
    single_parser.add_argument("--dataset", type=Path, default="./dataset.jsonl")
    single_parser.add_argument("--outputs-root", type=Path, default="./outputs")
    single_parser.add_argument("--cve-id", type=str, required=True)
    single_parser.add_argument("--timeout", type=str, default="45m")
    single_parser.add_argument("--claude-timeout", type=str, default="30m")
    single_parser.add_argument("--keep-container", action="store_true")
    single_parser.add_argument("--strategy", choices=get_available_strategies(), default="iterative")
    single_parser.add_argument("--api-provider", choices=["anthropic", "bedrock", "vertex"], default="anthropic")
    single_parser.add_argument("--interactive", action="store_true")
    

    single_parser.add_argument("--tool-limits", type=str)
    single_parser.add_argument("--max-cost-usd", type=float, default=10.0)
    single_parser.add_argument("--enable-detailed-logging", action="store_true", default=True)
    single_parser.add_argument("--save-process-logs", action="store_true")
    single_parser.add_argument("--allow-git-diff-fallback", action="store_true")
    single_parser.add_argument("--settings", type=str)
    single_parser.add_argument("--port", type=str)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--all", action="store_true")
    
    return parser.parse_args()


def parse_tool_limits(tool_limits_str: str) -> tuple:
    if not tool_limits_str:
        return {}, None
    
    if tool_limits_str.strip().lower().startswith('total:'):
        try:
            total_limit = int(tool_limits_str.split(':', 1)[1].strip())
            return {}, total_limit
        except ValueError as e:
            raise ValueError(f"Total tool limit format error: {e}")
    
    limits = {}
    try:
        for pair in tool_limits_str.split(","):
            tool, limit = pair.strip().split(":")
            limits[tool.strip()] = int(limit.strip())
    except ValueError as e:
        raise ValueError(f"Tool limits format error: {e}")
    
    return limits, None
def parse_timeout(timeout_str: str) -> int:
    """Parse timeout string like '45m', '2h' to seconds."""
    if timeout_str.endswith('s'):
        return int(timeout_str[:-1])
    elif timeout_str.endswith('m'):
        return int(timeout_str[:-1]) * 60
    elif timeout_str.endswith('h'):
        return int(timeout_str[:-1]) * 3600
    else:
        return int(timeout_str)  # Assume seconds


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_api_key_and_validate(api_provider: str) -> str:
    if api_provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY environment variable")
    elif api_provider == "bedrock":
        aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bedrock_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        
        if not aws_region:
            raise RuntimeError("Missing AWS_REGION or AWS_DEFAULT_REGION environment variable")
        if not (aws_access_key and aws_secret_key) and not bedrock_token:
            raise RuntimeError("Missing AWS credentials (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY) or AWS_BEARER_TOKEN_BEDROCK")
        api_key = bedrock_token or f"{aws_access_key}:{aws_secret_key}:{aws_region}"
    elif api_provider == "vertex":
        vertex_token = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("VERTEX_AUTH_TOKEN")
        vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT_ID")
        vertex_region = os.getenv("CLOUD_ML_REGION") or os.getenv("VERTEX_REGION")
        
        if not vertex_token:
            raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS or VERTEX_AUTH_TOKEN")
        if not vertex_project:
            raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT or VERTEX_PROJECT_ID")
        api_key = f"{vertex_token}:{vertex_project}:{vertex_region or 'us-central1'}"
    else:
        raise RuntimeError(f"Unsupported API provider: {api_provider}")
    
    return api_key


def handle_batch_command(args):
    """Handle batch processing command."""
    timeout_seconds = parse_timeout(args.timeout)
    claude_timeout_seconds = parse_timeout(args.claude_timeout)
    
    # Check API credentials  
    try:
        api_key = get_api_key_and_validate(args.api_provider)
    except RuntimeError as e:
        logging.error(str(e))
        return 1
    
    try:
        tool_limits_dict, max_total_calls = parse_tool_limits(getattr(args, 'tool_limits', None))
        
        summary = run_batch_cves(
            dataset_path=args.dataset,
            outputs_root=args.outputs_root,
            max_workers=args.max_workers,
            timeout_seconds=timeout_seconds,
            claude_timeout_seconds=claude_timeout_seconds,
            strategy=args.strategy,
            api_provider=args.api_provider,
            resume=args.resume,
            limit=args.limit,
            keep_containers=args.keep_containers,
            tool_limits=tool_limits_dict,
            max_total_tool_calls=max_total_calls,
            max_cost_usd=getattr(args, 'max_cost_usd', 10.0),
            enable_detailed_logging=getattr(args, 'enable_detailed_logging', True),
            save_process_logs=getattr(args, 'save_process_logs', False),
            allow_git_diff_fallback=getattr(args, 'allow_git_diff_fallback', False),
            settings_file=getattr(args, 'settings', None),
            port=args.port
        )
        
        print(f"\\n🎉 Batch processing completed!")
        print(f"   Total: {summary['total_processed']}")
        print(f"   Success: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success Rate: {summary['success_rate']:.2%}")
        print(f"   Duration: {summary['total_duration']:.1f}s")
        
        return 0 if summary['successful'] > 0 else 1
        
    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        return 1


def handle_single_command(args):
    """Handle single CVE processing command."""
    timeout_seconds = parse_timeout(args.timeout)
    claude_timeout_seconds = parse_timeout(args.claude_timeout)
    
    # Check API credentials
    try:
        api_key = get_api_key_and_validate(args.api_provider)
    except RuntimeError as e:
        logging.error(str(e))
        return 1
    
    # Load specific CVE record
    records = load_dataset(args.dataset)
    record = None
    for r in records:
        if r.cve_id == args.cve_id or r.problem_id == args.cve_id:
            record = r
            break
    
    if not record:
        logging.error(f"CVE not found: {args.cve_id}")
        return 1
    
    try:
        tool_limits_dict, max_total_calls = parse_tool_limits(getattr(args, 'tool_limits', None))
        
        result = run_single_cve(
            record=record,
            outputs_root=args.outputs_root,
            timeout_seconds=timeout_seconds,
            claude_timeout_seconds=claude_timeout_seconds,
            strategy=args.strategy,
            api_provider=args.api_provider,
            keep_container=args.keep_container,
            tool_limits=tool_limits_dict,
            max_total_tool_calls=max_total_calls,
            max_cost_usd=getattr(args, 'max_cost_usd', 10.0),
            enable_detailed_logging=getattr(args, 'enable_detailed_logging', True),
            save_process_logs=getattr(args, 'save_process_logs', False),
            allow_git_diff_fallback=getattr(args, 'allow_git_diff_fallback', False),
            settings_file=getattr(args, 'settings', None),
            port = args.port
        )
        
        if result["is_success"]:
            print(f"\\n✅ CVE repair successful: {args.cve_id}")
            print(f"   Duration: {result.get('agent_duration', 0):.1f}s")
            print(f"   Patch: {args.outputs_root}/patches/{record.problem_id}.patch")
        else:
            print(f"\\n❌ CVE repair failed: {args.cve_id}")
            print(f"   Stage: {result.get('stage', 'unknown')}")
            print(f"   Error: {result.get('error_message', 'No error message')}")
        
        return 0 if result["is_success"] else 1
        
    except Exception as e:
        logging.error(f"Single CVE processing failed: {e}")
        return 1


def handle_cleanup_command(args):
    """Handle cleanup command."""
    try:
        if args.all:
            cleanup_containers_by_prefix("bench.")
            print("🧹 All benchmark containers cleaned up")
        else:
            print("Use --all flag to clean up all benchmark containers")
        return 0
    except Exception as e:
        logging.error(f"Cleanup failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    if not args.command:
        print("Error: Please specify a command. Use --help for usage.")
        return 1
    
    setup_logging()
    
    try:
        if args.command == "batch":
            return handle_batch_command(args)
        elif args.command == "single":
            return handle_single_command(args)
        elif args.command == "cleanup":
            return handle_cleanup_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\\n⚠️ Operation interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())