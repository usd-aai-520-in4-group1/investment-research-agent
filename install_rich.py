#!/usr/bin/env python3
"""
Installation script for Rich library to enable enhanced visualization.
Run this script to install Rich and enable the enhanced workflow visualization.
"""

import subprocess
import sys
import os

def install_rich():
    """Install Rich library using pip."""
    print("Installing Rich library for enhanced workflow visualization...")
    print("=" * 60)
    
    try:
        # Install Rich
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich>=13.0.0"])
        print("✅ Rich library installed successfully!")
        
        # Test import
        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            
            console.print(Panel(
                "Rich library is now installed and ready to use!\n"
                "Your investment research workflow will now display:\n"
                "• Real-time progress bars with spinners\n"
                "• Colorful status indicators\n"
                "• Formatted tables and panels\n"
                "• Professional terminal output\n"
                "• Live updating displays during execution",
                title="🎉 Rich Installation Complete",
                border_style="green"
            ))
            
            print("\nYou can now run the enhanced workflow with Rich visualization!")
            print("Try running: python workflow_integration.py")
            
        except ImportError as e:
            print(f"❌ Error testing Rich import: {e}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Rich: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Investment Research Agent - Rich Visualization Setup")
    print("=" * 50)
    
    success = install_rich()
    
    if success:
        print("\n🎉 Setup complete! Rich visualization is now available.")
        print("\nNext steps:")
        print("1. Run: python workflow_integration.py")
        print("2. Or run the Jupyter notebook: Investment_Research_Agent_Complete.ipynb")
        print("3. Enjoy the enhanced visual experience!")
    else:
        print("\n❌ Setup failed. Please install Rich manually:")
        print("pip install rich>=13.0.0")
