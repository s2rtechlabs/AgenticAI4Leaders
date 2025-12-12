"""
A2A Exercise - Run All Agents

This script provides a cross-platform way to start all agents.
It works on Windows, macOS, and Linux.

Usage:
    python run_all_agents.py
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path


# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_colored(text: str, color: str = Colors.RESET):
    """Print colored text."""
    print(f"{color}{text}{Colors.RESET}")


def main():
    """Start all agents."""
    script_dir = Path(__file__).parent.absolute()
    
    print_colored("\n" + "="*60, Colors.BLUE)
    print_colored("  A2A Exercise - Starting All Agents", Colors.BLUE)
    print_colored("="*60 + "\n", Colors.BLUE)
    
    # Agent configurations
    agents = [
        {
            "name": "Calculator Agent",
            "path": script_dir / "agents" / "calculator",
            "port": 10001
        },
        {
            "name": "Greeter Agent",
            "path": script_dir / "agents" / "greeter",
            "port": 10002
        },
        {
            "name": "Weather Agent",
            "path": script_dir / "agents" / "weather",
            "port": 10003
        }
    ]
    
    processes = []
    
    def cleanup(signum=None, frame=None):
        """Clean up all processes."""
        print_colored("\n\nStopping all agents...", Colors.YELLOW)
        for proc, agent in zip(processes, agents):
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print_colored(f"  ✓ Stopped {agent['name']}", Colors.GREEN)
            except Exception as e:
                print_colored(f"  ✗ Error stopping {agent['name']}: {e}", Colors.RED)
        print_colored("\nAll agents stopped.", Colors.GREEN)
        sys.exit(0)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Start each agent
    for i, agent in enumerate(agents, 1):
        print_colored(f"[{i}/{len(agents)}] Starting {agent['name']} on port {agent['port']}...", Colors.GREEN)
        
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "__main__"],
                cwd=agent["path"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )
            processes.append(proc)
            time.sleep(1)  # Give each agent time to start
            
        except Exception as e:
            print_colored(f"  ✗ Failed to start {agent['name']}: {e}", Colors.RED)
    
    print_colored("\n" + "="*60, Colors.GREEN)
    print_colored("  All agents are running!", Colors.GREEN)
    print_colored("="*60, Colors.GREEN)
    
    print("""
Agent URLs:
  • Calculator: http://localhost:10001
  • Greeter:    http://localhost:10002
  • Weather:    http://localhost:10003

To test agent discovery:
  curl http://localhost:10001/.well-known/agent.json

To run the simple client:
  cd client && python simple_client.py

To run the orchestrator:
  cd client && python orchestrator.py

Press Ctrl+C to stop all agents...
""")
    
    # Wait for processes
    try:
        while True:
            time.sleep(1)
            # Check if any process has died
            for proc, agent in zip(processes, agents):
                if proc.poll() is not None:
                    print_colored(f"  ⚠ {agent['name']} has stopped unexpectedly", Colors.YELLOW)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()

