"""
Run both frontend and backend servers simultaneously
"""
import subprocess
import sys
import signal
import time
from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).parent.absolute()
BACKEND_DIR = ROOT_DIR / "app" / "backend"
FRONTEND_DIR = ROOT_DIR / "app" / "frontend"

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.OKGREEN):
    """Print colored message"""
    print(f"{color}{message}{Colors.ENDC}")

def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════╗
    ║     SemanticReads Development Server      ║
    ╚═══════════════════════════════════════════╝
    """
    print_colored(banner, Colors.HEADER)

def check_port(port):
    """Check if a port is in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run_servers():
    """Start both backend and frontend servers"""
    print_banner()

    # Check if ports are already in use
    if check_port(8000):
        print_colored("  Warning: Port 8000 is already in use!", Colors.WARNING)
        print_colored("   Backend server might already be running or the port is occupied.", Colors.WARNING)

    if check_port(8080):
        print_colored("  Warning: Port 8080 is already in use!", Colors.WARNING)
        print_colored("   Frontend server might already be running or the port is occupied.", Colors.WARNING)

    print()
    print_colored(" Starting servers...", Colors.OKBLUE)
    print()

    processes = []

    try:
        # Start Backend (FastAPI with uvicorn)
        print_colored(" Starting Backend Server (FastAPI)...", Colors.OKCYAN)
        print_colored(f"   Location: {BACKEND_DIR}", Colors.ENDC)
        print_colored("   URL: http://127.0.0.1:8000", Colors.OKGREEN)
        print()

        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(BACKEND_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        )
        processes.append(("Backend", backend_process))

        # Give backend a moment to start
        time.sleep(2)

        # Start Frontend (Python HTTP Server)
        print_colored(" Starting Frontend Server (HTTP)...", Colors.OKCYAN)
        print_colored(f"   Location: {FRONTEND_DIR}", Colors.ENDC)
        print_colored("   URL: http://127.0.0.1:8080", Colors.OKGREEN)
        print()

        frontend_process = subprocess.Popen(
            [sys.executable, "-m", "http.server", "8080"],
            cwd=str(FRONTEND_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        )
        processes.append(("Frontend", frontend_process))

        print()
        print_colored(" Both servers are running!", Colors.OKGREEN)
        print()
        print_colored(" Access the application:", Colors.BOLD)
        print_colored("   Frontend: http://127.0.0.1:8080", Colors.OKGREEN)
        print_colored("   Backend API: http://127.0.0.1:8000", Colors.OKGREEN)
        print_colored("   API Docs: http://127.0.0.1:8000/docs", Colors.OKGREEN)
        print()
        print_colored("Press Ctrl+C to stop both servers", Colors.WARNING)
        print()

        # Wait for processes
        while True:
            # Check if any process has died
            for name, proc in processes:
                if proc.poll() is not None:
                    print_colored(f"\n {name} server stopped unexpectedly!", Colors.FAIL)
                    raise KeyboardInterrupt
            time.sleep(1)

    except KeyboardInterrupt:
        print()
        print_colored("\n Stopping servers...", Colors.WARNING)

        # Terminate all processes
        for name, proc in processes:
            try:
                if sys.platform == 'win32':
                    # Windows: Send CTRL_BREAK_EVENT
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()

                print_colored(f"   Stopped {name} server", Colors.OKCYAN)
            except Exception as e:
                print_colored(f"   Error stopping {name}: {e}", Colors.FAIL)

        # Wait for processes to finish
        for name, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        print()
        print_colored("All servers stopped successfully!", Colors.OKGREEN)
        sys.exit(0)

if __name__ == "__main__":
    # Check if required dependencies are installed
    try:
        import uvicorn
    except ImportError:
        print_colored("Error: uvicorn is not installed!", Colors.FAIL)
        print_colored("   Please run: pip install uvicorn", Colors.WARNING)
        sys.exit(1)

    run_servers()

