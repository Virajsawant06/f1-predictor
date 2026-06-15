import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

st.title("🏁 F1 DNF & Position Predictor")

tab1, tab2 = st.tabs(["Single Driver", "Full Race Simulator"])

with tab1:
    st.header("Single Driver Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Race Info")
        grid = st.number_input("Grid Position", 1, 22, 1)
        qualifying_position = st.number_input("Qualifying Position", 1, 22, 1)
        grid_penalty = st.number_input("Grid Penalty", 0, 22, 0)
        year = st.number_input("Year", 2014, 2030, 2024)
        round_no = st.number_input("Round", 1, 25, 1)
        is_street_circuit = st.selectbox("Circuit Type", [0, 1], format_func=lambda x: "Street" if x == 1 else "Traditional")
        circuit_dnf_rate = st.slider("Circuit DNF Rate", 0.0, 1.0, 0.17)
    
    with col2:
        st.subheader("Driver Stats")
        driver_championship_position = st.number_input("Driver Championship Position", 1, 25, 1)
        driver_championship_points = st.number_input("Driver Championship Points", 0, 600, 100)
        driver_wins = st.number_input("Driver Wins", 0, 25, 0)
        driver_experience = st.number_input("Driver Experience", 0, 400, 100)
        driver_avg_position = st.slider("Driver Avg Position", 1.0, 20.0, 5.0)
        driver_dnf_rate = st.slider("Driver DNF Rate", 0.0, 1.0, 0.17)
        driver_dnf_rate_last10 = st.slider("Driver DNF Rate (Last 10)", 0.0, 1.0, 0.17)
        driver_dnf_rate_last5 = st.slider("Driver DNF Rate (Last 5)", 0.0, 1.0, 0.17)
    
    with col3:
        st.subheader("Constructor Stats")
        constructor_championship_position = st.number_input("Constructor Championship Position", 1, 10, 1)
        constructor_championship_points = st.number_input("Constructor Championship Points", 0, 1000, 200)
        constructor_wins = st.number_input("Constructor Wins", 0, 25, 0)
        constructor_avg_position = st.slider("Constructor Avg Position", 1.0, 20.0, 5.0)
        constructor_dnf_rate = st.slider("Constructor DNF Rate", 0.0, 1.0, 0.17)
        constructor_dnf_rate_last10 = st.slider("Constructor DNF Rate (Last 10)", 0.0, 1.0, 0.17)
        constructor_dnf_rate_last5 = st.slider("Constructor DNF Rate (Last 5)", 0.0, 1.0, 0.17)

    if st.button("PREDICT", type="primary", use_container_width=True):
        
        dnf_payload = {
            "grid": grid,
            "qualifying_position": qualifying_position,
            "driver_championship_points": driver_championship_points,
            "driver_championship_position": driver_championship_position,
            "driver_wins": driver_wins,
            "constructor_championship_points": constructor_championship_points,
            "constructor_championship_position": constructor_championship_position,
            "constructor_wins": constructor_wins,
            "year": year,
            "round": round_no,
            "driver_experience": driver_experience,
            "constructor_dnf_rate": constructor_dnf_rate,
            "constructor_dnf_rate_last10": constructor_dnf_rate_last10,
            "constructor_dnf_rate_last5": constructor_dnf_rate_last5,
            "driver_dnf_rate": driver_dnf_rate,
            "driver_dnf_rate_last10": driver_dnf_rate_last10,
            "driver_dnf_rate_last5": driver_dnf_rate_last5,
            "circuit_dnf_rate": circuit_dnf_rate,
            "is_street_circuit": is_street_circuit
        }
        
        dnf_response = requests.post(f"{API_URL}/predict/dnf", json=dnf_payload)
        dnf_result = dnf_response.json()
        
        position_payload = dnf_payload.copy()
        position_payload["grid_penalty"] = grid_penalty
        position_payload["driver_avg_position"] = driver_avg_position
        position_payload["constructor_avg_position"] = constructor_avg_position
        position_payload["dnf_probability"] = dnf_result["dnf_probability"]
        
        position_response = requests.post(f"{API_URL}/predict/position", json=position_payload)
        position_result = position_response.json()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("DNF Prediction", "DNF" if dnf_result["prediction"] == 1 else "FINISH")
        with col2:
            st.metric("DNF Probability", f"{dnf_result['dnf_probability']*100:.1f}%")
        with col3:
            st.metric("Predicted Position", f"P{position_result['predicted_position']}")

with tab2:
    st.header("Full Race Simulator")
    
    if "drivers" not in st.session_state:
        st.session_state.drivers = []
    if "form_version" not in st.session_state:
        st.session_state.form_version = 0
    
    v = st.session_state.form_version
    
    # --- JSON Upload ---
    st.markdown("**Upload drivers as JSON** (optional)")

    uploaded_file = st.file_uploader(
        "Choose a JSON file",
        type="json",
        key="json_upload"
    )

    if uploaded_file is not None:
        import json

        try:
            json_data = json.load(uploaded_file)

            st.write("JSON loaded successfully")
            

            if isinstance(json_data, list):
                st.write(f"Drivers found: {len(json_data)}")

                st.session_state.drivers = json_data

                st.success(
                    f"Loaded {len(st.session_state.drivers)} drivers!"
                )

            else:
                st.error(
                    "JSON must contain a list of driver objects."
                )

        except Exception as e:
            st.error(f"Error reading JSON: {e}")

    st.write(
        f"Current drivers in session: {len(st.session_state.drivers)}"
    )

    
    # --- Add/Clear Buttons ---
    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("➕ Add Driver"):
            st.session_state.drivers.append({})
    with col_b:
        if st.button("🗑️ Clear All"):
            st.session_state.drivers = []
            st.session_state.form_version += 1
            st.rerun()
    
    # --- Driver Forms ---
    for i, driver in enumerate(st.session_state.drivers):
        with st.expander(f"Driver {i+1}", expanded=True):
            driver['driver_name'] = st.text_input("Name", value=driver.get('driver_name', f"Driver {i+1}"), key=f"name_{v}_{i}")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                driver['grid'] = st.number_input("Grid", 1, 22, value=int(driver.get('grid', 1)), key=f"grid_{v}_{i}")
                driver['qualifying_position'] = st.number_input("Quali Pos", 1, 22, value=int(driver.get('qualifying_position', 1)), key=f"quali_{v}_{i}")
                driver['grid_penalty'] = st.number_input("Grid Penalty", 0, 22, value=int(driver.get('grid_penalty', 0)), key=f"penalty_{v}_{i}")
                driver['year'] = st.number_input("Year", 2014, 2030, value=int(driver.get('year', 2024)), key=f"year_{v}_{i}")
                driver['round'] = st.number_input("Round", 1, 25, value=int(driver.get('round', 1)), key=f"round_{v}_{i}")
                driver['is_street_circuit'] = st.selectbox("Circuit Type", [0, 1], index=int(driver.get('is_street_circuit', 0)), format_func=lambda x: "Street" if x else "Traditional", key=f"street_{v}_{i}")
                driver['circuit_dnf_rate'] = st.slider("Circuit DNF Rate", 0.0, 1.0, value=float(driver.get('circuit_dnf_rate', 0.17)), key=f"cdnf_{v}_{i}")
            
            with c2:
                driver['driver_championship_position'] = st.number_input("Driver Champ Pos", 1, 25, value=int(driver.get('driver_championship_position', 10)), key=f"dcp_{v}_{i}")
                driver['driver_championship_points'] = st.number_input("Driver Champ Pts", 0, 600, value=int(driver.get('driver_championship_points', 50)), key=f"dchp_{v}_{i}")
                driver['driver_wins'] = st.number_input("Driver Wins", 0, 25, value=int(driver.get('driver_wins', 0)), key=f"dwins_{v}_{i}")
                driver['driver_experience'] = st.number_input("Driver Experience", 0, 600, value=int(driver.get('driver_experience', 50)), key=f"dexp_{v}_{i}")
                driver['driver_avg_position'] = st.slider("Driver Avg Pos", 1.0, 20.0, value=float(driver.get('driver_avg_position', 10.0)), key=f"davg_{v}_{i}")
                driver['driver_dnf_rate'] = st.slider("Driver DNF Rate", 0.0, 1.0, value=float(driver.get('driver_dnf_rate', 0.17)), key=f"ddnf_{v}_{i}")
            
            with c3:
                driver['driver_dnf_rate_last10'] = st.slider("Driver DNF (Last 10)", 0.0, 1.0, value=float(driver.get('driver_dnf_rate_last10', 0.17)), key=f"ddnf10_{v}_{i}")
                driver['driver_dnf_rate_last5'] = st.slider("Driver DNF (Last 5)", 0.0, 1.0, value=float(driver.get('driver_dnf_rate_last5', 0.17)), key=f"ddnf5_{v}_{i}")
                driver['constructor_championship_position'] = st.number_input("Constructor Champ Pos", 1, 12, value=int(driver.get('constructor_championship_position', 5)), key=f"ccp_{v}_{i}")
                driver['constructor_championship_points'] = st.number_input("Constructor Champ Pts", 0, 1000, value=int(driver.get('constructor_championship_points', 100)), key=f"cchp_{v}_{i}")
                driver['constructor_wins'] = st.number_input("Constructor Wins", 0, 25, value=int(driver.get('constructor_wins', 0)), key=f"cwins_{v}_{i}")
            
            with c4:
                driver['constructor_avg_position'] = st.slider("Constructor Avg Pos", 1.0, 20.0, value=float(driver.get('constructor_avg_position', 10.0)), key=f"cavg_{v}_{i}")
                driver['constructor_dnf_rate'] = st.slider("Constructor DNF Rate", 0.0, 1.0, value=float(driver.get('constructor_dnf_rate', 0.17)), key=f"cdnf2_{v}_{i}")
                driver['constructor_dnf_rate_last10'] = st.slider("Constructor DNF (Last 10)", 0.0, 1.0, value=float(driver.get('constructor_dnf_rate_last10', 0.17)), key=f"cdnf10_{v}_{i}")
                driver['constructor_dnf_rate_last5'] = st.slider("Constructor DNF (Last 5)", 0.0, 1.0, value=float(driver.get('constructor_dnf_rate_last5', 0.17)), key=f"cdnf5_{v}_{i}")
    
    st.divider()
    
    if len(st.session_state.drivers) < 2:
        st.warning("Add at least 2 drivers to simulate a race.")
    else:
        if st.button("🏁 SIMULATE RACE", type="primary", use_container_width=True):
            payload = [driver for driver in st.session_state.drivers]
            response = requests.post(f"{API_URL}/predict/race", json=payload)
            
            if response.status_code == 200:
                results = response.json()
                results_df = pd.DataFrame(results)
                results_df = results_df.rename(columns={
                    'driver': 'Driver',
                    'predicted_position': 'Predicted Position',
                    'dnf_probability': 'DNF Risk (%)'
                })
                st.markdown("### 🏆 Predicted Race Result")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
            else:
                st.error(f"API Error: {response.text}")