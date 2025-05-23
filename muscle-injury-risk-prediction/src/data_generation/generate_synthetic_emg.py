import numpy as np
import pandas as pd

def simulate_emg_features(signal, fs):
    # Time-domain features
    rms = np.sqrt(np.mean(signal**2))
    mav = np.mean(np.abs(signal))
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    wl = np.sum(np.abs(np.diff(signal)))
    # Frequency-domain features
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))
    total_power = np.sum(fft_vals)
    mdf = freqs[np.where(np.cumsum(fft_vals) >= total_power/2)[0][0]]
    mnf = np.sum(freqs * fft_vals) / np.sum(fft_vals)
    return rms, mav, zc, ssc, wl, mdf, mnf

def generate_demographics(n):
    ages = np.random.randint(18, 36, n)
    heights = np.random.normal(190, 8, n)  # cm
    weights = np.random.normal(85, 10, n)  # kg
    bmi = weights / ((heights/100)**2)
    training_freq = np.random.randint(3, 8, n)
    prev_injury = np.random.choice(['none', 'calves', 'hamstrings', 'quadriceps'], n, p=[0.7, 0.1, 0.1, 0.1])
    return ages, heights, weights, bmi, training_freq, prev_injury

def generate_synthetic_emg_row(muscle, contraction, fs, duration, noise_level, fatigue_level):
    t = np.linspace(0, duration, int(fs*duration))
    # Frequency and amplitude base per muscle, add random variation
    freq_map = {'calves': 10, 'hamstrings': 8, 'quadriceps': 12}
    amp_map = {'calves': 1.0, 'hamstrings': 1.1, 'quadriceps': 1.2}
    freq = freq_map[muscle] + np.random.uniform(-1, 1)  # add freq jitter
    amp = amp_map[muscle] * np.random.uniform(0.9, 1.1)  # add amp jitter
    # Fatigue: RMS increases, MDF/MNF decrease over time
    fatigue_trend = 1 + 0.5 * fatigue_level + np.random.uniform(-0.1, 0.1)  # add fatigue jitter
    freq_trend = 1 - 0.3 * fatigue_level + np.random.uniform(-0.05, 0.05)   # add freq trend jitter
    # Isometric: less fluctuation; Isotonic: more fluctuation
    fluct = 0.1 if contraction == 'isometric' else 0.3
    base_signal = amp * np.sin(2 * np.pi * freq * t) * fatigue_trend
    base_signal += fluct * np.random.randn(len(t))
    # Add baseline drift
    drift = np.sin(2 * np.pi * np.random.uniform(0.1, 0.5) * t) * np.random.uniform(0.05, 0.15)
    base_signal += drift
    # Add burst noise (simulate motion artifact)
    if np.random.rand() < 0.2:
        burst_start = np.random.randint(0, len(t) - 50)
        base_signal[burst_start:burst_start+50] += np.random.normal(0, 0.5, 50)
    # Add sensor/motion noise
    noise = np.random.normal(0, noise_level * np.random.uniform(0.8, 1.5), len(t))
    emg = base_signal * freq_trend + noise
    # Feature extraction
    rms, mav, zc, ssc, wl, mdf, mnf = simulate_emg_features(emg, fs)
    return rms, mav, zc, ssc, wl, mdf, mnf

def estimate_fatigue_from_emg(rms, mdf, mnf, rms_range=(0.5, 2.0), mdf_range=(80, 150), mnf_range=(80, 150)):
    rms_norm = (rms - rms_range[0]) / (rms_range[1] - rms_range[0])
    mdf_norm = 1 - (mdf - mdf_range[0]) / (mdf_range[1] - mdf_range[0])
    mnf_norm = 1 - (mnf - mnf_range[0]) / (mnf_range[1] - mnf_range[0])
    rms_norm = np.clip(rms_norm, 0, 1)
    mdf_norm = np.clip(mdf_norm, 0, 1)
    mnf_norm = np.clip(mnf_norm, 0, 1)
    fatigue_level = 0.5 * rms_norm + 0.25 * mdf_norm + 0.25 * mnf_norm
    return float(np.clip(fatigue_level, 0, 1))

def generate_synthetic_emg_dataset(n_samples=2000, fs=1000, duration=1.0, noise_level=0.12):
    muscles = ['calves', 'hamstrings', 'quadriceps']
    contractions = ['isometric', 'isotonic']
    data = []
    ages, heights, weights, bmis, train_freqs, prev_injs = generate_demographics(n_samples)
    for i in range(n_samples):
        row = {
            'age': ages[i],
            'height': heights[i],
            'weight': weights[i],
            'bmi': bmis[i],
            'training_frequency': train_freqs[i],
            'previous_injury': prev_injs[i],
            # 'contraction_type' will be set after fatigue calculation
        }
        fatigue_levels = []
        for muscle in muscles:
            fatigue_for_signal = np.random.uniform(0, 1)
            rms, mav, zc, ssc, wl, mdf, mnf = generate_synthetic_emg_row(
                muscle, 'isometric', fs, duration, noise_level, fatigue_for_signal  # temp contraction type
            )
            row[f'rms_{muscle}'] = rms
            row[f'mav_{muscle}'] = mav
            row[f'zc_{muscle}'] = zc
            row[f'ssc_{muscle}'] = ssc
            row[f'wl_{muscle}'] = wl
            row[f'mdf_{muscle}'] = mdf
            row[f'mnf_{muscle}'] = mnf
            muscle_fatigue = estimate_fatigue_from_emg(rms, mdf, mnf)
            row[f'fatigue_level_{muscle}'] = muscle_fatigue
            fatigue_levels.append(muscle_fatigue)
        # Set global fatigue level
        avg_fatigue = np.mean(fatigue_levels)
        row['fatigue_level'] = avg_fatigue
        # Set contraction type based on fatigue
        if avg_fatigue > 0.7:
            row['contraction_type'] = 'isotonic'
        elif avg_fatigue < 0.3:
            row['contraction_type'] = 'isometric'
        else:
            row['contraction_type'] = np.random.choice(['isometric', 'isotonic'])
        # Derived: correlation trends (simulate as random for now)
        row['rms_time_corr'] = np.random.uniform(-0.5, 1)
        row['mnf_time_corr'] = np.random.uniform(-1, 0.5)
        # Risk label (simulate, not rule-based)
        risk_score = (
            0.4 * row['fatigue_level'] +
            0.2 * (row['rms_calves'] + row['rms_hamstrings'] + row['rms_quadriceps'])/3 +
            0.2 * (1 - (row['mnf_calves'] + row['mnf_hamstrings'] + row['mnf_quadriceps'])/3/100) +
            0.1 * (row['training_frequency']/7) +
            0.3 * (1 if row['previous_injury'] != 'none' else 0)
        )
        if risk_score > 0.5:
            row['injury_risk'] = 'high'
        elif risk_score > 0.3:
            row['injury_risk'] = 'medium'
        else:
            row['injury_risk'] = 'low'
        data.append(row)
    return pd.DataFrame(data)

def split_dataset_by_muscle(df):
    muscles = ['calves', 'hamstrings', 'quadriceps']
    base_cols = ['age', 'height', 'weight', 'bmi', 'training_frequency', 'previous_injury',
                 'contraction_type', 'fatigue_level', 'rms_time_corr', 'mnf_time_corr', 'injury_risk']
    muscle_datasets = {}
    for muscle in muscles:
        muscle_cols = [f'{feat}_{muscle}' for feat in ['rms', 'mav', 'zc', 'ssc', 'wl', 'mdf', 'mnf']]
        cols = base_cols + muscle_cols
        muscle_datasets[muscle] = df[cols].copy()
    return muscle_datasets

if __name__ == "__main__":
    df = generate_synthetic_emg_dataset(n_samples=500)
    print(df.head())
    df.to_csv("synthetic_emg_full_features.csv", index=False)
    # Split and save per muscle
    muscle_datasets = split_dataset_by_muscle(df)
    for muscle, muscle_df in muscle_datasets.items():
        muscle_df.to_csv(f"synthetic_emg_{muscle}.csv", index=False)