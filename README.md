# Clustering K-Anonymity (CKA) Protocol Implementation

**An interactive privacy protection engine for Edge Computing and IoT networks.**

## Overview
This project is an implementation of the **Clustering K-Anonymity (CKA)** protocol, designed to secure location data in Vehicular Ad-hoc Networks (VANETs). It mitigates "narrow region attacks" by dynamically generating virtual (dummy) nodes to blend with real user coordinates.

This tool allows researchers and security engineers to:
- **Simulate** eavesdropping attacks on ad-hoc networks.
- **Visualize** the dynamic clustering algorithm in real-time.
- **Audit** privacy metrics ($D_K$, $A_K$) and data efficiency.

## Tech Stack
- **Python 3.10+**
- **Streamlit**: Interactive UI and real-time parameter tuning.
- **NumPy & SciPy**: Vectorized spatial calculations and distance matrices.
- **Plotly**: Rendering spatial graphs of nodes and clusters.

