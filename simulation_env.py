# simulation_environment.py (Updated to print the specific error)
import math
import time

TARGET_FREQ = 1.842e9
TARGET_BW = 80e6
TARGET_GAIN = 70.0
TARGET_AMPLITUDE = 0.7
TARGET_AMPLITUDE_WIDTH = 0.05
TARGET_SAMPLING_FREQ = 40e6
TARGET_NUM_SAMPLES = 20000

def mock_run_simulation_and_get_reward(config: dict) -> float:
    print(f"--- Running Simulation ---")
    print(f"Config: {config}")

    if not config:
        print("Invalid config: The provided configuration was None or empty. Reward is -1.0.")
        return -1.0
        
    try:
        center_freq = float(config.get("center_frequency", 0))
        bandwidth = float(config.get("bandwidth", 0))
        tx_gain = float(config.get("tx_gain", 0))
        amplitude = float(config.get("amplitude", 0))
        amplitude_width = float(config.get("amplitude_width", 0))
        sampling_freq = float(config.get("sampling_freq", 0))
        num_samples = int(config.get("num_samples", 0))
        
        if config.get("device_args") != "type=b200":
             return -0.2 
             
    # --- CHANGED: Capture the exception as 'e' and print it ---
    except (TypeError, ValueError, AttributeError) as e:
        print(f"!!! VALUE PARSING FAILED !!! Error was: {e}")
        print("Invalid config format or missing/invalid value types. Reward is -0.1.")
        return -0.1

    # --- Score calculation (unchanged) ---
    freq_score = max(0.0, 1.0 - (abs(center_freq - TARGET_FREQ) / 200e6))
    bw_score = max(0.0, 1.0 - (abs(bandwidth - TARGET_BW) / (TARGET_BW * 2)))
    gain_score = max(0.0, 1.0 - (abs(tx_gain - TARGET_GAIN) / 50.0))
    amplitude_score = max(0.0, 1.0 - (abs(amplitude - TARGET_AMPLITUDE) / 1.0))
    amp_width_score = max(0.0, 1.0 - (abs(amplitude_width - TARGET_AMPLITUDE_WIDTH) / 0.2))
    sampling_freq_score = max(0.0, 1.0 - (abs(sampling_freq - TARGET_SAMPLING_FREQ) / (TARGET_SAMPLING_FREQ * 2)))
    num_samples_score = max(0.0, 1.0 - (abs(num_samples - TARGET_NUM_SAMPLES) / (TARGET_NUM_SAMPLES * 2)))

    final_reward = (freq_score * 0.30) + (bw_score * 0.15) + (gain_score * 0.10) + \
                   (amplitude_score * 0.10) + (amp_width_score * 0.05) + \
                   (sampling_freq_score * 0.15) + (num_samples_score * 0.15)
    
    if final_reward > 0.98: final_reward = 1.0

    print(f"Scores -> Freq: {freq_score:.2f}, BW: {bw_score:.2f}, Gain: {gain_score:.2f}, Amp: {amplitude_score:.2f}, AmpW: {amp_width_score:.2f}, SampF: {sampling_freq_score:.2f}, NSamp: {num_samples_score:.2f}")
    print(f"Result -> Reward: {final_reward:.3f}")
    print(f"--------------------------\n")
    
    return final_reward