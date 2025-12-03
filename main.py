#!/usr/bin/env python3
"""
Neural Ocean - Marine Life and Pollution Detection
Main entry point for the application
"""

import streamlit as st
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point for the Neural Ocean application"""
    try:
        from src.apps.main_app import main as run_app
        run_app()
    except ImportError as e:
        st.error(f"Import error: {e}")
        st.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error("Please check the logs for more details")

if __name__ == "__main__":
    main()
