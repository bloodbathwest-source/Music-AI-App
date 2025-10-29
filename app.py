"""Music AI App - Streamlit Interface.

This is a simple Streamlit interface for the Music AI application.
For the full application, use the backend (FastAPI) and frontend (Next.js).
"""
import streamlit as st


def main():
    """Main application entry point."""
    st.title("Welcome to the Music AI App")
    st.write("This app uses AI to generate music based on your preferences.")


if __name__ == "__main__":
    main()
