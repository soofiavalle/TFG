



import os
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from scipy.signal import find_peaks

# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS USED IN STATISTICS ANALYSIS
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------


def extract_peak_to_peak_values(signal, fs, window_size_s):
    """
    Extracts all average peak-to-peak amplitudes from the signal, one per window.


    Parameters:
        signal (np.array): The PIC signal.
        fs (float): Sampling frequency in Hz.
        window_size_s (int): Duration of each window in seconds.


    Returns:
        List[float]: List of average peak-to-peak amplitudes per window.
    """
    from scipy.signal import find_peaks
    import numpy as np


    samples_per_window = int(fs * window_size_s)
    num_windows = len(signal) // samples_per_window
    all_amplitudes = []


    for i in range(num_windows):
        start_idx = i * samples_per_window
        end_idx = start_idx + samples_per_window
        segment = signal[start_idx:end_idx]


        peaks, _ = find_peaks(segment, distance=fs * 0.5, prominence=0.1)
        valleys, _ = find_peaks(-segment, distance=fs * 0.5, prominence=0.1)


        window_amplitudes = []
        for peak in peaks:
            future_valleys = valleys[valleys > peak]
            if len(future_valleys) == 0:
                continue
            valley = future_valleys[0]
            amplitude = segment[peak] - segment[valley]
            window_amplitudes.append(amplitude)


        if window_amplitudes:
            all_amplitudes.append(np.mean(window_amplitudes))


    return all_amplitudes


# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS USED IN SIGNAL FRAGMENTS ANALYSIS
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------


# --- Load signal and metadata from LabChart .txt file ---
# Returns: signal (np.array), sampling rate (fs), start time, and unit
def load_signal_and_metadata(file_path, verbose=True):
    fs = None
    start_time = None
    unit = "unknown"
    signal = []




    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()




    for line in lines:
        if line.startswith("Interval="):
            interval_str = line.split("=")[1].strip().replace(",", ".").split()[0]
            fs = 1.0 / float(interval_str)
        elif line.startswith("ExcelDateTime="):
            parts = line.split("\t")
            if len(parts) >= 2:
                datetime_str = parts[-1].split(",")[0].strip().replace(" UT", "")
                try:
                    start_time = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S.%f")
                except ValueError:
                    start_time = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
        elif line.startswith("UnitName="):
            unit_parts = line.strip().split("\t")
            if len(unit_parts) >= 3:
                unit = unit_parts[2].strip()
        elif line.strip().startswith("BottomValue="):
            break




    if fs is None or start_time is None:
        raise ValueError("Missing metadata: 'Interval=' or 'ExcelDateTime=' not found.")




    for line in lines:
        if line.strip() and line[0].isdigit():
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    value = float(parts[2].replace(",", "."))
                    signal.append(value)
                except ValueError:
                    continue




    signal = np.array(signal)




    if verbose:
        print(f"\n--- File loaded: {file_path}")
        print(f"Extracted {len(signal)} samples | fs = {fs:.2f} Hz | Start: {start_time}")
        duration_min = len(signal) / fs / 60
        print(f"Duration: {duration_min:.2f} min")
        print(f"Unit: {unit}")
        print("First PIC values:", signal[:3])




    return signal, fs, start_time, unit


# ----------------------------------------------------------------------------------------------------------------------------------
# --- Save analysis result to summary .txt file ---
# Appends one line per result with metadata and values

def save_result_to_txt(result_dict, fragment_name, start_time, unit, window_size_s):
    import os, numpy as np

    result_path = "final_results/all_results.txt"
    os.makedirs("final_results", exist_ok=True)

    # Base fields (siempre primero)
    base_fields = ["fragment_name", "start_time", "unit", "window_size_s"]

    # Orden deseado para tus métricas
    preferred_metric_order = [
        "time_peak_to_peak_amp",
        "first_harmonic_peak",
        "first_harmonic_p2p",
        "multi_harmonics_peak",
    ]

    # Campos extra presentes en este resultado (excluye internos)
    extra_fields_raw = [
        k for k in result_dict.keys()
        if k not in base_fields and k not in ("window_means", "num_windows")
    ]
    # Ordena métricas según preferencia; conserva cualquier otra que aparezca
    extras_sorted = [k for k in preferred_metric_order if k in extra_fields_raw] + \
                    [k for k in extra_fields_raw if k not in preferred_metric_order]

    # Cabecera objetivo
    all_fields = base_fields + ["num_windows"] + extras_sorted

    # Valores de esta línea
    formatted_values = [
        fragment_name,
        start_time.strftime('%Y-%m-%d %H:%M:%S'),
        unit,
        f"{window_size_s}"
    ]
    num_windows = result_dict.get("num_windows", "")
    formatted_values.append(
        f"{num_windows:.4f}"
        if isinstance(num_windows, (float, int, np.integer, np.floating))
        else str(num_windows)
    )
    for k in extras_sorted:
        v = result_dict.get(k, "")
        if isinstance(v, (float, int, np.integer, np.floating)):
            formatted_values.append(f"{v:.4f}")
        else:
            formatted_values.append(str(v))
    new_line = ", ".join(formatted_values)

    if not os.path.exists(result_path):
        # Crear archivo con cabecera nueva
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("# Results file generated by Analysis.ipynb\n")
            f.write("# Columns:\n")
            f.write("# " + ", ".join(all_fields) + "\n")
            f.write(new_line + "\n")
        return

    # Si existe, comprobar si la cabecera incluye todas las columnas
    with open(result_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Buscar línea de columnas (la que empieza por "# " y tiene comas)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("# ") and "," in line and "Columns" not in line:
            header_idx = i
            break

    if header_idx is None:
        # Cabecera ausente: reescribir con cabecera nueva
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("# Results file generated by Analysis.ipynb\n")
            f.write("# Columns:\n")
            f.write("# " + ", ".join(all_fields) + "\n")
            f.writelines(lines)
            f.write(new_line + "\n")
        return

    existing_fields = [c.strip() for c in lines[header_idx][2:].split(",")]
    # Si faltan columnas nuevas, actualiza la cabecera (añade al final en orden preferido)
    if any(col not in existing_fields for col in all_fields):
        updated = existing_fields[:]
        for col in all_fields:
            if col not in updated:
                updated.append(col)
        lines[header_idx] = "# " + ", ".join(updated) + "\n"
        with open(result_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
            f.write(new_line + "\n")
    else:
        # Cabecera ya ok → solo append
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(new_line + "\n")


# ----------------------------------------------------------------------------------------------------------------------------------

from datetime import timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
def time_peak_to_peak_amp(signal, fs, window_size_s, fragment_name, start_time, html_fig=None, subplot_row_start=1):
    from datetime import timedelta
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.signal import find_peaks




    samples_per_window = int(fs * window_size_s)
    num_windows = len(signal) // samples_per_window
    amplitudes_per_window = []




    # Create or reuse figure
    if html_fig is None:
        fig = make_subplots(
            rows=9, cols=1,
            shared_xaxes=False,
            subplot_titles=[
                "8s – First", "8s – Last", "8s – Summary",
                "15s – First", "15s – Last", "15s – Summary",
                "30s – First", "30s – Last", "30s – Summary"
            ],
            vertical_spacing=0.07
        )
    else:
        fig = html_fig




    for i in range(num_windows):
        start_idx = i * samples_per_window
        end_idx = start_idx + samples_per_window
        segment = signal[start_idx:end_idx]
        time_segment = [start_time + timedelta(seconds=(start_idx + j) / fs) for j in range(samples_per_window)]




        peaks, _ = find_peaks(segment, distance=fs * 0.5, prominence=0.1)
        valleys, _ = find_peaks(-segment, distance=fs * 0.5, prominence=0.1)




        window_amplitudes = []
        for peak in peaks:
            future_valleys = valleys[valleys > peak]
            if len(future_valleys) == 0:
                continue
            valley = future_valleys[0]
            amp = segment[peak] - segment[valley]
            window_amplitudes.append(amp)




        amplitudes_per_window.append(np.mean(window_amplitudes) if window_amplitudes else np.nan)




        if i == 0 or i == num_windows - 1:
            row = subplot_row_start if i == 0 else subplot_row_start + 1




            fig.add_trace(go.Scatter(
                x=time_segment, y=segment, mode='lines', line=dict(color='gray')
            ), row=row, col=1)




            fig.add_trace(go.Scatter(
                x=[time_segment[p] for p in peaks],
                y=[segment[p] for p in peaks],
                mode='markers', marker=dict(color='blue', size=6), showlegend=False
            ), row=row, col=1)




            fig.add_trace(go.Scatter(
                x=[time_segment[v] for v in valleys],
                y=[segment[v] for v in valleys],
                mode='markers', marker=dict(color='green', size=6), showlegend=False
            ), row=row, col=1)




            for peak in peaks:
                future_valleys = valleys[valleys > peak]
                if len(future_valleys) == 0:
                    continue
                valley = future_valleys[0]
                fig.add_trace(go.Scatter(
                    x=[time_segment[peak], time_segment[peak]],
                    y=[segment[valley], segment[peak]],
                    mode='lines', line=dict(color='red', width=2), showlegend=False
                ), row=row, col=1)




    mean_amplitudes = [a for a in amplitudes_per_window if not np.isnan(a)]
    result = {
        "num_windows": num_windows,
        "window_means": np.round(mean_amplitudes, 3),
        "time_peak_to_peak_amp": round(np.mean(mean_amplitudes), 3) if mean_amplitudes else np.nan
    }




    if len(mean_amplitudes) > 0:
        amplitudes_str = ""
        for i in range(0, len(mean_amplitudes), 10):
            group = mean_amplitudes[i:i + 10]
            amplitudes_str += ", ".join(f"{a:.3f}" for a in group) + "<br>"




        summary_text = (
            f"<b>Window size:</b> {window_size_s}s &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Total valid windows:</b> {len(mean_amplitudes)} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Mean amplitude:</b> {np.mean(mean_amplitudes):.3f} mmHg<br>"
            f"<b>Amplitudes:</b><br>{amplitudes_str}"
        )




        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.01,
            y=1 - (subplot_row_start / 9) - 0.05,
            showarrow=False,
            align="left",
            font=dict(size=13, color="black"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=6
        )




    fig.update_layout(showlegend=False)
    return result, fig


#--------------------------------------------------------------------------------------------------------------------------

def first_harmonic_amp(signal, fs, window_size_s, fragment_name, start_time, html_fig=None, subplot_row_start=1):
    """
    Compute the 1st harmonic amplitude in the cardiac range using FFT.
    Returns TWO metrics:
      - first_harmonic_peak : peak amplitude from FFT (same scale as the spectrum)
      - first_harmonic_p2p  : 2xpeak, conventionally comparable to time-domain p2p if all energy were in f1
    Plotting is done in peak scale (to match FFT y-axis).
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    samples_per_window = int(fs * window_size_s)
    num_windows = len(signal) // samples_per_window
    peaks_list = []  # store A1 (peak)

    interval = 1 / fs
    fft_algorithm = "numpy.fft.fft"

    # Create/reuse figure
    if html_fig is None:
        fig = make_subplots(
            rows=9, cols=1, shared_xaxes=False,
            subplot_titles=[
                "8s – First", "8s – Last", "8s – Summary",
                "15s – First", "15s – Last", "15s – Summary",
                "30s – First", "30s – Last", "30s – Summary"
            ],
            vertical_spacing=0.07
        )
    else:
        fig = html_fig

    for i in range(num_windows):
        start_idx = i * samples_per_window
        end_idx = start_idx + samples_per_window
        segment = signal[start_idx:end_idx]
        N = len(segment)

        # FFT and normalization to 'peak' amplitude
        yf = np.fft.fft(segment)
        xf = np.fft.fftfreq(N, d=1/fs)
        pos_mask = xf >= 0
        xf_pos = xf[pos_mask]
        yf_abs = 2.0 / N * np.abs(yf[pos_mask])

        # Cardiac band search
        freq_min, freq_max = 0.5, 2.0
        range_mask = (xf_pos >= freq_min) & (xf_pos <= freq_max)

        if np.any(range_mask):
            yf_range = yf_abs[range_mask]
            xf_range = xf_pos[range_mask]
            idx_max = np.argmax(yf_range)

            first_harmonic_freq = float(xf_range[idx_max])
            A1_peak = float(yf_range[idx_max])     # <-- PEAK amplitude (same scale as FFT)
        else:
            first_harmonic_freq = np.nan
            A1_peak = np.nan

        peaks_list.append(A1_peak)

        # Plot only first & last windows
        if i == 0 or i == num_windows - 1:
            row = subplot_row_start if i == 0 else subplot_row_start + 1

            fig.add_trace(go.Scatter(
                x=xf_pos, y=yf_abs, mode='lines', name='FFT spectrum'
            ), row=row, col=1)

            if not np.isnan(A1_peak):
                # marker at the 1st harmonic (in peak scale)
                fig.add_trace(go.Scatter(
                    x=[first_harmonic_freq], y=[A1_peak],
                    mode='markers+text', text=[f"{A1_peak:.2f}"],
                    textposition="top center", showlegend=False
                ), row=row, col=1)

                # vertical helper line
                fig.add_trace(go.Scatter(
                    x=[first_harmonic_freq, first_harmonic_freq],
                    y=[0, A1_peak],
                    mode='lines', line=dict(dash='dash'), showlegend=False
                ), row=row, col=1)

    # Aggregate results
    valid_peaks = [a for a in peaks_list if not np.isnan(a)]
    mean_A1_peak = float(np.mean(valid_peaks)) if valid_peaks else np.nan
    mean_A1_p2p  = float(2.0 * mean_A1_peak) if valid_peaks else np.nan

    result = {
        "num_windows": num_windows,
        "window_means": np.round(valid_peaks, 3),
        "first_harmonic_peak": round(mean_A1_peak, 3) if valid_peaks else np.nan,
        "first_harmonic_p2p":  round(mean_A1_p2p,  3) if valid_peaks else np.nan
    }

    # Summary box
    if valid_peaks:
        amps_str = ""
        for i in range(0, len(valid_peaks), 10):
            group = valid_peaks[i:i + 10]
            amps_str += ", ".join(f"{a:.3f}" for a in group) + "<br>"

        summary_text = (
            f"<b>Window size:</b> {window_size_s}s &nbsp;|&nbsp; "
            f"<b>Total windows:</b> {len(valid_peaks)} &nbsp;|&nbsp; "
            f"<b>Mean 1st harmonic (peak):</b> {mean_A1_peak:.3f} mmHg &nbsp;|&nbsp; "
            f"<b>as p2p (2×):</b> {mean_A1_p2p:.3f} mmHg<br>"
            f"<b>Interval:</b> {interval:.6f} s &nbsp;|&nbsp; "
            f"<b>Samples/window:</b> {samples_per_window} &nbsp;|&nbsp; "
            f"<b>FFT:</b> {fft_algorithm}<br>"
            f"<b>Per-window peaks:</b><br>{amps_str}"
        )

        fig.add_annotation(
            text=summary_text, xref="paper", yref="paper",
            x=0.01, y=1 - (subplot_row_start / 9) - 0.05,
            showarrow=False, align="left",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="gray", borderwidth=1, borderpad=6
        )

    fig.update_layout(showlegend=False)
    return result, fig

#--------------------------------------------------------------------------------------------------------------------------
def rms_of_harmonics(signal, fs, window_size_s, fragment_name, start_time, html_fig=None, subplot_row_start=1):
    """
    Combine multiple harmonics (including the fundamental) as the square root of
    the sum of squared PEAK amplitudes detected near multiples of f1.
    Returns:
      - multi_harmonics_peak : sqrt(A1^2 + A2^2 + ...), PEAK scale (same as FFT y-axis)
    Notes:
      * Not a classic full-signal RMS. It's a partial RMS-like measure over selected harmonics.
      * We explicitly include A1 so multi_harmonics_peak >= first_harmonic_peak (numerically coherent).
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    samples_per_window = int(fs * window_size_s)
    num_windows = len(signal) // samples_per_window
    combined_list = []

    interval = 1 / fs
    fft_algorithm = "numpy.fft.fft"

    if html_fig is None:
        fig = make_subplots(
            rows=9, cols=1, shared_xaxes=False,
            subplot_titles=[
                "8s – First", "8s – Last", "8s – Summary",
                "15s – First", "15s – Last", "15s – Summary",
                "30s – First", "30s – Last", "30s – Summary"
            ],
            vertical_spacing=0.07
        )
    else:
        fig = html_fig

    freq_max_limit = 15.0         # hard cap
    search_margin_hz = 0.2        # ± margin around each k*f1
    max_harmonics = 5             # up to 5 harmonics (including k=1)

    for i in range(num_windows):
        start_idx = i * samples_per_window
        end_idx = start_idx + samples_per_window
        segment = signal[start_idx:end_idx]
        N = len(segment)

        # FFT and normalization (peak amplitude)
        yf = np.fft.fft(segment)
        xf = np.fft.fftfreq(N, d=1/fs)
        pos_mask = xf >= 0
        xf_pos = xf[pos_mask]
        yf_abs = 2.0 / N * np.abs(yf[pos_mask])

        # 1) Fundamental = argmax in 0.5–2.0 Hz
        band_mask = (xf_pos >= 0.5) & (xf_pos <= 2.0)
        harmonic_values = []
        harmonic_freqs_found = []

        if np.any(band_mask):
            band_amps = yf_abs[band_mask]
            band_freqs = xf_pos[band_mask]
            idx_max = np.argmax(band_amps)
            f1 = float(band_freqs[idx_max])
            A1 = float(band_amps[idx_max])

            # include fundamental
            harmonic_values.append(A1)
            harmonic_freqs_found.append(f1)

            # 2) Higher harmonics near k*f1
            max_harmonic_freq = min(freq_max_limit, 5 * f1)
            target_freqs = [f1 * k for k in range(2, max_harmonics + 1) if f1 * k <= max_harmonic_freq]

            for target in target_freqs:
                mask = (xf_pos >= target - search_margin_hz) & (xf_pos <= target + search_margin_hz)
                if np.any(mask):
                    local_freqs = xf_pos[mask]
                    local_amps = yf_abs[mask]
                    kmax = int(np.argmax(local_amps))
                    harmonic_values.append(float(local_amps[kmax]))
                    harmonic_freqs_found.append(float(local_freqs[kmax]))

            # combine as sqrt(sum of squares)
            combined_amp = float(np.sqrt(np.sum(np.square(harmonic_values)))) if len(harmonic_values) > 0 else np.nan
        else:
            combined_amp = np.nan

        combined_list.append(combined_amp)

        # Plot first & last windows
        if i == 0 or i == num_windows - 1:
            row = subplot_row_start if i == 0 else subplot_row_start + 1

            fig.add_trace(go.Scatter(
                x=xf_pos, y=yf_abs, mode='lines', name='FFT spectrum'
            ), row=row, col=1)

            if harmonic_freqs_found:
                fig.add_trace(go.Scatter(
                    x=harmonic_freqs_found,
                    y=[yf_abs[np.argmin(np.abs(xf_pos - f))] for f in harmonic_freqs_found],
                    mode='markers', marker=dict(size=7),
                    name='Harmonics', showlegend=False
                ), row=row, col=1)

                for f in harmonic_freqs_found:
                    fig.add_trace(go.Scatter(
                        x=[f, f], y=[0, max(yf_abs)],
                        mode='lines', line=dict(dash='dot'),
                        showlegend=False
                    ), row=row, col=1)

    # Aggregate
    valid_combined = [a for a in combined_list if not np.isnan(a)]
    mean_multi_peak = float(np.mean(valid_combined)) if valid_combined else np.nan

    result = {
        "num_windows": num_windows,
        "window_means": np.round(valid_combined, 3),
        "multi_harmonics_peak": round(mean_multi_peak, 3) if valid_combined else np.nan
    }

    # Summary
    if valid_combined:
        amps_str = ""
        for i in range(0, len(valid_combined), 10):
            group = valid_combined[i:i + 10]
            amps_str += ", ".join(f"{a:.3f}" for a in group) + "<br>"

        summary_text = (
            f"<b>Window size:</b> {window_size_s}s &nbsp;|&nbsp; "
            f"<b>Total windows:</b> {len(valid_combined)} &nbsp;|&nbsp; "
            f"<b>Mean multi-harmonics (peak):</b> {mean_multi_peak:.3f} mmHg<br>"
            f"<b>Method:</b> f1=argmax in 0.5–2.0 Hz; search ±{search_margin_hz:.1f} Hz up to 5×f1 (≤{freq_max_limit} Hz)<br>"
            f"<b>FFT:</b> {fft_algorithm}<br>"
            f"<b>Per-window combined peaks:</b><br>{amps_str}"
        )

        fig.add_annotation(
            text=summary_text, xref="paper", yref="paper",
            x=0.01, y=1 - (subplot_row_start / 9) - 0.05,
            showarrow=False, align="left",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="gray", borderwidth=1, borderpad=6
        )

    fig.update_layout(showlegend=False)
    return result, fig


#--------------------------------------------------------------------------------------------------------------------------
