#!/usr/bin/env python3
"""
GUI Launcher for Hotel Review Sentiment Analysis

This script launches the graphical user interface for sentiment analysis.
Run this to start the interactive application.
"""

import sys
import os

def check_requirements():
    """Check if all required packages are available."""
    required_packages = ['pandas', 'nltk', 'sklearn', 'numpy', 'tkinter']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("📥 Please install them with: pip install -r requirements.txt")
        return False
    
    return True

def launch_gui():
    """Launch the sentiment analysis GUI."""
    if not check_requirements():
        return
    
    try:
        print("🚀 Launching Hotel Review Sentiment Analyzer GUI...")
        print("📋 Features available:")
        print("   • Interactive sentiment analysis")
        print("   • Detailed explanations")
        print("   • Model training interface")
        print("   • Example reviews to try")
        print()
        
        # Import and run the GUI
        from sentiment_gui import main
        main()
        
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        print("💡 Try running in a graphical environment (not headless)")

if __name__ == "__main__":
    launch_gui()
