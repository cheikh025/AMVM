project_specs = {
    "order":12,
    "p_bits": 4,
    "pass_edge": 2/5,        # Used for lowpass or 1st passband
    "stop_edge": 4/7,       # Used for lowpass or 1st stopband
    "pass_edge2": None,      # Optional: for bandpass/bandstop
    "stop_edge2": None,      # Optional: for bandpass/bandstop
    "fs": 2.0,
    "K": 1.0,
    "grid_density": 16,
    "filter_type": "lowpass" # 'lowpass' or 'highpass', 'bandpass', 'notch' or 'bandstop'
}