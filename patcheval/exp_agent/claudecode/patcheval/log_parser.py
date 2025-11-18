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
"""Claude Code stream-json log parser - generate human-readable logs"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class StreamJsonLogParser:
    
    def __init__(self):
        self.events = []
        self.tool_usage = {}
        self.current_session = None
    
    def parse_log_file(self, log_file_path: str) -> Dict[str, Any]:
        events = []
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                agent_log = json.loads(content)
                if 'claude_output' in agent_log:
                    return self._parse_agent_log_format(agent_log, log_file_path)
            except json.JSONDecodeError:
                pass
                
            lines = content.strip().split('\n')
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    event = json.loads(line)
                    event['_line_number'] = line_num
                    events.append(event)
                except json.JSONDecodeError:
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            fixed_line = self._fix_json_line(line)
                            if fixed_line:
                                event = json.loads(fixed_line)
                                event['_line_number'] = line_num
                                events.append(event)
                        except:
                            pass
                    
        except Exception as e:
            print(f"Error reading log file: {e}")
            return {"error": str(e), "events": []}
        
        return {
            "total_events": len(events),
            "events": events,
            "summary": self._generate_summary(events)
        }
    
    def _parse_agent_log_format(self, agent_log: Dict, log_file_path: str) -> Dict[str, Any]:
        events = []
        
        claude_output = agent_log.get('claude_output', '')
        
        json_events = []
        for line in claude_output.split('\n'):
            line = line.strip()
            if line.startswith('{"type":'):
                try:
                    event = json.loads(line)
                    json_events.append(event)
                except json.JSONDecodeError:
                    continue
        
        base_info = {
            'cve_id': agent_log.get('cve_id', 'unknown'),
            'strategy': agent_log.get('strategy', 'unknown'),
            'duration': agent_log.get('duration', 0),
            'patch_stats': agent_log.get('patch_stats', {}),
            'api_provider': agent_log.get('api_provider', 'unknown')
        }
        
        return {
            "total_events": len(json_events),
            "events": json_events,
            "base_info": base_info,
            "summary": self._generate_agent_summary(json_events, base_info),
            "format": "agent_logs"
        }
    
    def _generate_agent_summary(self, events: List[Dict], base_info: Dict) -> Dict[str, Any]:
        summary = {
            "cve_id": base_info.get('cve_id'),
            "strategy": base_info.get('strategy'),
            "duration": base_info.get('duration'),
            "patch_stats": base_info.get('patch_stats'),
            "tool_usage": {},
            "message_count": 0,
            "error_count": 0,
            "main_activities": []
        }
        
        for event in events:
            event_type = event.get("type", "unknown")
            
            if event_type == "assistant":
                message = event.get("message", {})
                content = message.get("content", [])
                
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_use":
                            tool_name = item.get("name", "unknown")
                            summary["tool_usage"][tool_name] = summary["tool_usage"].get(tool_name, 0) + 1
                
                summary["message_count"] += 1
                
                content_str = str(content)
                if any(err in content_str.lower() for err in ["error", "failed", "exception"]):
                    summary["error_count"] += 1
        
        summary["main_activities"] = self._extract_activities_from_agent_log(events)
        
        return summary
    
    def _extract_activities_from_agent_log(self, events: List[Dict]) -> List[str]:
        activities = set()
        
        for event in events:
            if event.get("type") == "assistant":
                message = event.get("message", {})
                content = message.get("content", [])
                
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text = item.get("text", "").lower()
                                if "analyz" in text:
                                    activities.add("Code Analysis")
                                if "fix" in text or "fix" in text:
                                    activities.add("Vulnerability Fix")  
                                if "test" in text or "test" in text:
                                    activities.add("Testing/Verification")
                                if "patch" in text or "patch" in text:
                                    activities.add("Patch Generation")
                            elif item.get("type") == "tool_use":
                                tool_name = item.get("name", "")
                                if tool_name in ["Read", "Grep", "LS"]:
                                    activities.add("Code Analysis")
                                elif tool_name in ["Edit", "Write", "MultiEdit"]:
                                    activities.add("Code Modification")
                                elif tool_name == "Bash":
                                    activities.add("Command Execution")
        
        return list(activities)
    
    def _fix_json_line(self, line: str) -> Optional[str]:
        line = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', line)
        
        if line.count('{') > line.count('}'):
            line += '}' * (line.count('{') - line.count('}'))
        
        return line if line.strip() else None
    
    def _generate_summary(self, events: List[Dict]) -> Dict[str, Any]:
        summary = {
            "session_count": 0,
            "tool_usage": {},
            "message_count": 0,
            "error_count": 0,
            "session_durations": [],
            "main_activities": []
        }
        
        current_session_start = None
        
        for event in events:
            event_type = event.get("type", "unknown")
            
            if event_type == "session_start":
                summary["session_count"] += 1
                current_session_start = event.get("timestamp")
                
            elif event_type == "session_end" and current_session_start:
                duration = event.get("timestamp", 0) - current_session_start
                summary["session_durations"].append(duration)
                
            elif event_type == "tool_call":
                tool_name = event.get("tool", {}).get("name", "unknown")
                summary["tool_usage"][tool_name] = summary["tool_usage"].get(tool_name, 0) + 1
                
            elif event_type == "message":
                summary["message_count"] += 1
                content = event.get("content", "")
                if isinstance(content, str) and any(err in content.lower() for err in ["error", "failed", "exception"]):
                    summary["error_count"] += 1

        summary["main_activities"] = self._extract_main_activities(events)
        
        return summary
    
    def _extract_main_activities(self, events: List[Dict]) -> List[str]:
        activities = []
        
        for event in events:
            if event.get("type") == "message":
                content = event.get("content", "")
                if isinstance(content, str):
                    if "analyzing" in content.lower():
                        activities.append("Code Analysis")
                    elif "fixing" in content.lower() or "fix" in content:
                        activities.append("Vulnerability Fix")
                    elif "testing" in content.lower() or "test" in content:
                        activities.append("Testing/Verification")
                    elif "generating patch" in content.lower() or "ss" in content:
                        activities.append("Patch Generation")
        
        return list(set(activities))  # deduplicate
    
    def generate_human_readable_log(self, log_file_path: str, output_path: Optional[str] = None) -> str:
        data = self.parse_log_file(log_file_path)
        
        if "error" in data:
            return f"❌ Log parsing failed: {data['error']}"
        
        events = data["events"]
        summary = data["summary"]
        is_agent_format = data.get("format") == "agent_logs"
        
        report_lines = []
        report_lines.append("# Claude CVE Fix Task Report")
        report_lines.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if is_agent_format:
            base_info = data.get("base_info", {})
            report_lines.append(f"🎯 CVE ID: {base_info.get('cve_id', 'unknown')}")
            report_lines.append(f"🔧 Fix Strategy: {base_info.get('strategy', 'unknown')}")
            report_lines.append(f"⏱️ Duration: {base_info.get('duration', 0):.1f} seconds")
            
            patch_stats = base_info.get('patch_stats', {})
            if patch_stats:
                report_lines.append(f"📊 Patch Stats: {patch_stats.get('files_changed', 0)} files, +{patch_stats.get('lines_added', 0)}/-{patch_stats.get('lines_removed', 0)} lines")
        
        report_lines.append("")
        report_lines.append("## 📊 Summary")
        report_lines.append(f"- Total events: {data['total_events']}")
        
        if is_agent_format:
            report_lines.append(f"- API Provider: {summary.get('api_provider', 'unknown')}")
        else:
            report_lines.append(f"- Sessions: {summary.get('session_count', 0)}")
        
        report_lines.append(f"- Messages: {summary['message_count']}")
        report_lines.append(f"- Errors: {summary['error_count']}")
        
        if not is_agent_format and summary.get('session_durations'):
            avg_duration = sum(summary['session_durations']) / len(summary['session_durations'])
            report_lines.append(f"- Avg. session duration: {avg_duration:.1f} seconds")
        
        report_lines.append("")
        
        if summary.get('tool_usage'):
            report_lines.append("## 🛠️ Tool Usage")
            for tool, count in sorted(summary['tool_usage'].items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"- {tool}: {count} times")
            report_lines.append("")
        
        if summary.get('main_activities'):
            report_lines.append("## 🎯 Main Activities")
            for activity in summary['main_activities']:
                report_lines.append(f"- {activity}")
            report_lines.append("")
        
        report_lines.append("## 📝 Timeline")
        
        if is_agent_format:
            self._generate_agent_timeline(report_lines, events)
        else:
            self._generate_stream_timeline(report_lines, events)
        
        error_events = [e for e in events if "error" in str(e).lower()]
        if error_events:
            report_lines.append("\n## ⚠️ Errors and Warnings")
            for event in error_events[:10]:  
                content = str(event.get("content", event))
                if len(content) > 150:
                    content = content[:150] + "..."
                report_lines.append(f"- {content}")
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ Human-readable log saved to: {output_path}")
        
        return report_content
    
    def _generate_agent_timeline(self, report_lines: List[str], events: List[Dict]) -> None:
        conversation_step = 0
        
        for event in events:
            event_type = event.get("type", "unknown")
            
            if event_type == "system":
                subtype = event.get("subtype", "")
                if subtype == "init":
                    report_lines.append(f"\n### 🚀 System Initialization")
                    tools = event.get("tools", [])
                    if tools:
                        report_lines.append(f"🔧 Tools: {', '.join(tools[:5])}{'...' if len(tools) > 5 else ''}")
                
            elif event_type == "assistant":
                conversation_step += 1
                message = event.get("message", {})
                content = message.get("content", [])
                
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text = item.get("text", "")
                                report_lines.append(f"\n**Step {conversation_step}**: {text}")
                            elif item.get("type") == "tool_use":
                                tool_name = item.get("name", "unknown")
                                tool_input = item.get("input", {})
                                report_lines.append(f"🔧 **Tool Call**: {tool_name}")
                                if tool_name == "Read" and "file_path" in tool_input:
                                    report_lines.append(f"   📂 Read file: {tool_input['file_path']}")
                                elif tool_name == "Edit" and "file_path" in tool_input:
                                    report_lines.append(f"   ✏️ Edit file: {tool_input['file_path']}")
                                elif tool_name == "Bash" and "command" in tool_input:
                                    cmd = tool_input['command']
                                    if len(cmd) > 2000:
                                        cmd = cmd[:2000] + "..."
                                    report_lines.append(f"   💻 Execute command: {cmd}")
            
            elif event_type == "user":
                message = event.get("message", {})
                content = message.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            result = item.get("content", "")
                            report_lines.append(f"📤 **Result**: {result}")
    
    def _generate_stream_timeline(self, report_lines: List[str], events: List[Dict]) -> None:
        current_session = 0
        for event in events:
            timestamp = event.get("timestamp", 0)
            event_type = event.get("type", "unknown")
            
            if event_type == "session_start":
                current_session += 1
                report_lines.append(f"\n### 🚀 Session {current_session} started")
                
            elif event_type == "message":
                content = event.get("content", "")
                role = event.get("role", "unknown")
                
                if isinstance(content, str):
                    if role == "user":
                        report_lines.append(f"👤 **User**: {content}")
                    elif role == "assistant":
                        report_lines.append(f"🤖 **Assistant**: {content}")
                        
            elif event_type == "tool_call":
                tool_info = event.get("tool", {})
                tool_name = tool_info.get("name", "unknown")
                report_lines.append(f"🔧 **Tool Call**: {tool_name}")
                
            elif event_type == "tool_result":
                result = event.get("result", "")
                report_lines.append(f"📤 **Tool Result**: {result}")
                
            elif event_type == "session_end":
                report_lines.append(f"✅ **Session Ended**")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse Claude Code stream-json logs")
    parser.add_argument("log_file", help="Path to input stream-json log file")
    parser.add_argument("-o", "--output", help="Path to save human-readable log file")
    parser.add_argument("-p", "--print", action="store_true", help="Print to console")
    
    args = parser.parse_args()
    
    parser = StreamJsonLogParser()
    report = parser.generate_human_readable_log(args.log_file, args.output)
    
    if args.print:
        print(report)


if __name__ == "__main__":
    main()