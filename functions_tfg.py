import os
import numpy as np
import re
import pandas as pd
import inspect
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from plotly.offline import plot
from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
import re
from datetime import datetime

# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS FOR PREPROCESSING, EXTRACTING FRAGMENTS, CHOOSING FILTER AND FOR VISUALIZATION OF SIGNALS
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------

def load_pic_and_comments(file_path, verbose=True):
    """
    Loads PIC Direct values and accompanying comments from a LabChart-exported .txt file.
    Automatically detects the sampling frequency (Fs), the start time, and the correct column to use.

    Parameters:
        file_path (str): Path to the text file that contains the data.
        verbose (bool): If True, prints detailed information about the file content.

    Returns:
        df_pic (pd.DataFrame): DataFrame with line numbers and PIC Direct values.
        df_comments (pd.DataFrame): DataFrame with line numbers and detected comments.
        pic_values (list): List of extracted PIC values.
        Fs (float): Sampling frequency (Hz).
        start_time (datetime): Real-world time corresponding to the start of the recording.
    """

    Fs = None
    start_time = None
    pic_values = []
    pic_line_indices = []
    comments = []
    ignored_values = 0
    value_index = None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # --- Extract Fs and start time ---
    for line in lines:
        if line.startswith("Interval="):
            interval_str = line.split("=")[1].split()[0].replace(",", ".")
            Fs = 1 / float(interval_str)
        elif line.startswith("ExcelDateTime="):
            datetime_str = line.split("\t")[-1].split(",")[0].strip().replace(" UT", "")
            try:
                start_time = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S.%f")
            except ValueError:
                start_time = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
        elif line.strip().startswith("BottomValue="):
            break

    if Fs is None or start_time is None:
        raise ValueError("Header missing: 'Interval=' or 'ExcelDateTime=' not found.")

    # --- Detect column with numeric PIC values ---
    sample_data_line = next((line for line in lines if re.match(r"^\d", line.strip())), None)
    if sample_data_line:
        parts = sample_data_line.strip().split()
        for idx in reversed(range(len(parts))):  # Empieza por la última columna
            try:
                _ = float(parts[idx].replace(",", "."))
                value_index = idx
                break
            except ValueError:
                continue
        if value_index is None:
            raise ValueError("No numeric column found in sample data line.")
    else:
        raise ValueError("Could not find a valid data line to determine column structure.")

    if verbose:
        print(f"Detected PIC value column: {value_index} (0-based index)")

    # --- Parse data lines ---
    for i, line in enumerate(lines):
        if any(h in line for h in ["Interval=", "ExcelDateTime=", "TimeFormat=", "DateFormat=",
                                   "ChannelTitle=", "Range=", "UnitName=", "TopValue=", "BottomValue="]):
            continue

        parts = line.strip().split()
        if len(parts) > value_index:
            try:
                value = float(parts[value_index].replace(",", "."))
                pic_values.append(value)
                pic_line_indices.append(i)

                if "#*" in line:
                    comment = line.split("#*", 1)[1].strip()
                    comments.append((i, comment))
            except ValueError:
                ignored_values += 1
        else:
            ignored_values += 1

    # --- Build DataFrames ---
    df_pic = pd.DataFrame({
        "Line": pic_line_indices,
        "PIC_Direct": pic_values
    })

    df_comments = pd.DataFrame(comments, columns=["Line", "Comment"])

    # --- Verbose output ---
    if verbose:
        print("First 10 PIC values:")
        print(df_pic.head(10))
        print("\nFirst comments:")
        print(df_comments.head(10))
        print(f"\nTotal comments: {len(df_comments)}")
        print(f"Ignored values: {ignored_values}")
        if pic_values:
            print(f"Mean PIC: {np.mean(pic_values):.2f} mmHg")
            total_sec = len(pic_values) / Fs
            print(f"Total duration: {int(total_sec // 3600)}h {int((total_sec % 3600) // 60)}min")
        else:
            print("No valid PIC values found.")

    return df_pic, df_comments, pic_values, Fs, start_time



#--------------------------------------------------------------------------------------------------------------------------------

def plot_raw_pic_signal(pic_values, df_comments, file_path, Fs, start_time):
    """
    Plot raw PIC signal with interactive controls and comment markers using real clock time.
    Handles both cases: with and without clinical comments.
    """

    import os
    import numpy as np
    import plotly.graph_objects as go
    from datetime import timedelta

    # --- Time axis and downsampling ---
    dt = 1 / Fs
    initial_window_sec = 600
    num_samples = len(pic_values)
    downsample_factor = 1

    time_vector = [start_time + timedelta(seconds=i * dt) for i in range(num_samples)]
    t_down = time_vector[:num_samples:downsample_factor]
    pic_down = pic_values[:num_samples:downsample_factor]

    # --- Extract folder name for title ---
    folder_name = os.path.basename(os.path.dirname(file_path))
    title_text = f"Raw PIC Signal - {folder_name}"

    # --- Create the Plotly figure ---
    fig = go.Figure()

    # Raw signal
    fig.add_trace(go.Scattergl(
        x=t_down,
        y=pic_down,
        mode='lines',
        line=dict(color='red', width=1.2),
        name="PIC Signal"
    ))

    # Comments: vertical lines + text annotation (if any)
    if df_comments is not None and len(df_comments) > 0:
        for _, row in df_comments.iterrows():
            comment_time = start_time + timedelta(seconds=row["Line"] * dt)

            fig.add_trace(go.Scattergl(
                x=[comment_time, comment_time],
                y=[-20, 80],
                mode="lines",
                line=dict(color="blue", width=1, dash="dash"),
                showlegend=False
            ))

            fig.add_annotation(
                x=comment_time,
                y=-18,
                text=row["Comment"],
                xref="x",
                yref="y",
                showarrow=False,
                yshift=-10,
                font=dict(size=10, color="blue")
            )
    else:
        print("No comments to display for this signal.")

    # --- Safe x-axis range ---
    if len(t_down) > 0:
        x_range_start = t_down[0]
        x_range_end = x_range_start + timedelta(seconds=initial_window_sec)
    else:
        x_range_start = start_time
        x_range_end = start_time + timedelta(seconds=initial_window_sec)

    # --- Layout ---
    fig.update_layout(
        title=title_text,
        xaxis_title="Time of Day",
        yaxis_title="Pressure (mmHg)",
        width=1100,
        height=630,
        xaxis=dict(
            range=[x_range_start, x_range_end],
            rangeslider=dict(visible=True, thickness=0.05),
            fixedrange=False,
            type="date"
        ),
        yaxis=dict(
            range=[-20, 80],
            fixedrange=False
        ),
        hovermode="x unified",
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                text="Double-click on the graph to reset the scale.",
                xref="paper", yref="paper",
                x=1, y=1.2,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right",
                yanchor="top"
            )
        ]
    )

    # --- Show plot ---
    fig.show(config={
        "displayModeBar": False,
        "scrollZoom": True
    })


#--------------------------------------------------------------------------------------------------------------------------------


import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

def plot_comparison_raw_noartifact_signal(pic_values, filtered_pic_values, df_comments, file_path, Fs=200, start_datetime=None,
                                          primary_threshold=None, secondary_threshold=None, comment_threshold=None):
    """
    Plot comparison between raw PIC signal and filtered signal (with artifacts removed).
    Explicitly receives threshold values, including optional comment threshold.
    
    Parameters:
        pic_values (array-like): Raw PIC signal.
        filtered_pic_values (array-like): Filtered signal with NaNs where artifacts were removed.
        df_comments (DataFrame): Comments with "Line" and "Comment".
        file_path (str): Path used to extract patient name.
        Fs (int): Sampling frequency in Hz.
        start_datetime (datetime): Start time of recording.
        primary_threshold (float): Threshold for primary artifact detection (optional).
        secondary_threshold (float): Threshold for secondary artifact detection (optional).
        comment_threshold (float): Threshold related to comment proximity (optional).
    """
    dt = 1 / Fs
    initial_window_sec = 600
    num_samples = len(pic_values)
    downsample_factor = 10

    # Generate time axis
    if start_datetime:
        t = [start_datetime + timedelta(seconds=i * dt) for i in range(num_samples)]
    else:
        t = np.arange(0, num_samples * dt, dt)

    t_down = t[::downsample_factor]
    pic_down = pic_values[::downsample_factor]
    filtered_down = filtered_pic_values[::downsample_factor]

    folder_name = os.path.basename(os.path.dirname(file_path))
    title_text = f"PIC signal: raw vs no artifacts - {folder_name}"

    fig = go.Figure()

    # Raw signal
    fig.add_trace(go.Scattergl(
        x=t_down,
        y=pic_down,
        mode='lines',
        line=dict(color='red', width=1.2),
        name="Raw PIC"
    ))

    # Filtered signal
    fig.add_trace(go.Scattergl(
        x=t_down,
        y=filtered_down,
        mode='lines',
        line=dict(color='green', width=1.5),
        name="Filtered PIC"
    ))

    # Primary threshold line
    if primary_threshold is not None:
        fig.add_trace(go.Scattergl(
            x=[t_down[0], t_down[-1]],
            y=[primary_threshold, primary_threshold],
            mode='lines',
            line=dict(color='orange', width=1.2, dash='dash'),
            name=f"Primary Threshold ({primary_threshold:.1f} mmHg)"
        ))
        print(f"Primary Threshold: {primary_threshold:.1f} mmHg")

    # Secondary threshold line
    if secondary_threshold is not None:
        fig.add_trace(go.Scattergl(
            x=[t_down[0], t_down[-1]],
            y=[secondary_threshold, secondary_threshold],
            mode='lines',
            line=dict(color='purple', width=1.2, dash='dash'),
            name=f"Secondary Threshold ({secondary_threshold:.1f} mmHg)"
        ))
        print(f"Secondary Threshold: {secondary_threshold:.1f} mmHg")

    # Comment threshold line
    if comment_threshold is not None:
        fig.add_trace(go.Scattergl(
            x=[t_down[0], t_down[-1]],
            y=[comment_threshold, comment_threshold],
            mode='lines',
            line=dict(color='blue', width=1.2, dash='dot'),
            name=f"Comment Threshold ({comment_threshold:.1f} mmHg)"
        ))
        print(f"Comment Threshold: {comment_threshold:.1f} mmHg")

    # Comments (vertical lines and annotations)
    for _, row in df_comments.iterrows():
        comment_time = row["Line"] * dt
        if comment_time <= num_samples * dt:
            comment_x = start_datetime + timedelta(seconds=comment_time) if start_datetime else comment_time / 3600
            fig.add_trace(go.Scattergl(
                x=[comment_x, comment_x],
                y=[-20, 80],
                mode="lines",
                line=dict(color="blue", width=1, dash="dash"),
                showlegend=False
            ))
            fig.add_annotation(
                x=comment_x,
                y=-18,
                text=row["Comment"],
                showarrow=False,
                yshift=-10,
                font=dict(size=10, color="blue")
            )

    # Layout and axis configuration
    xaxis_range = [t[0], t[0] + timedelta(seconds=initial_window_sec)] if start_datetime else [0, initial_window_sec / 3600]

    fig.update_layout(
        title=title_text,
        xaxis_title="Time" if start_datetime else "Time (hours)",
        yaxis_title="Pressure (mmHg)",
        width=1100,
        height=630,
        xaxis=dict(
            range=xaxis_range,
            rangeslider=dict(visible=True, thickness=0.05),
            fixedrange=False
        ),
        yaxis=dict(
            range=[-20, 80],
            fixedrange=False
        ),
        hovermode="x unified",
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="center",
            x=0.5
        ),
        annotations=[
            dict(
                text="Double-click on the graph to reset the scale.",
                xref="paper", yref="paper",
                x=1, y=1.2,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right",
                yanchor="top"
            )
        ]
    )

    # Show figure
    fig.show(config={
        "displayModeBar": False,
        "scrollZoom": True
    })



#---------------------------------------------------------------------------------------------------------------------------

def plot_artifact_removed_pic_signal(filtered_pic_values, mask, df_comments, file_path, Fs=200, start_time=None):
    """
    Plots only the filtered PIC signal (after artifact removal), with comment markers.

    Parameters:
        filtered_pic_values (array-like): PIC signal after artifact removal (contains NaNs).
        mask (array-like): Binary mask (1 = valid, 0 = artifact). Not shown but needed for compatibility.
        df_comments (DataFrame): DataFrame with columns ["Line", "Comment"].
        file_path (str): Path to the file (used to extract folder name for the title).
        Fs (int): Sampling frequency in Hz. Default is 200 Hz.
        start_time (datetime): Datetime object marking the start of the recording.
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    from datetime import timedelta
    import inspect

    dt = 1 / Fs
    initial_window_sec = 600
    num_samples = len(filtered_pic_values)
    downsample_factor = 10

    # Time axis
    if start_time:
        t = [start_time + timedelta(seconds=i * dt) for i in range(num_samples)]
    else:
        t = np.arange(0, num_samples * dt, dt)

    t_down = t[::downsample_factor]
    filtered_down = filtered_pic_values[::downsample_factor]

    # Plot setup
    folder_name = os.path.basename(os.path.dirname(file_path))
    title_text = f"Filtered PIC Signal - {folder_name}"

    fig = go.Figure()

    # Filtered signal (linea continua, sin 'dot')
    fig.add_trace(go.Scattergl(
        x=t_down,
        y=filtered_down,
        mode='lines',
        name='Filtered PIC',
        line=dict(color='green', width=1.5)  # Línea continua
    ))

    # Comments
    for _, row in df_comments.iterrows():
        comment_time = start_time + timedelta(seconds=row["Line"] * dt)
        fig.add_trace(go.Scattergl(
            x=[comment_time, comment_time],
            y=[-20, 80],
            mode="lines",
            line=dict(color="blue", width=1, dash="dash"),
            showlegend=False
        ))
        fig.add_annotation(
            x=comment_time,
            y=-18,
            text=row["Comment"],
            xref="x",
            yref="y",
            showarrow=False,
            yshift=-10,
            font=dict(size=10, color="blue")
        )

    # Layout
    fig.update_layout(
        title=title_text,
        xaxis_title="Time of Day",
        yaxis_title="Pressure (mmHg)",
        width=1100,
        height=630,
        xaxis=dict(
            range=[t_down[0], t_down[0] + timedelta(seconds=initial_window_sec)],
            rangeslider=dict(visible=True, thickness=0.05),
            fixedrange=False,
            type="date"
        ),
        yaxis=dict(
            range=[-20, 80],
            fixedrange=False
        ),
        hovermode="x unified",
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="center",
            x=0.5
        ),
        annotations=[
            dict(
                text="Double-click on the graph to reset the scale.",
                xref="paper", yref="paper",
                x=1, y=1.2,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right",
                yanchor="top"
            )
        ]
    )

    fig.show(config={
        "displayModeBar": False,
        "scrollZoom": True
    })


#--------------------------------------------------------------------------------------------------------------------------------


def plot_full_artifact_analysis(pic_values, filtered_values, mask, df_comments, file_path, Fs=200, start_datetime=None):
    """
    Plot synchronized multi-panel view of the PIC signal analysis:
    - Raw vs filtered signal
    - Binary artifact mask
    - Clean signal (valid values only)
    """

    import os
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import timedelta

    # --- Time axis configuration ---
    dt = 1 / Fs
    num_samples = len(pic_values)
    downsample_factor = 10  # Reduce resolution to speed up plotting

    # Create full time axis
    if start_datetime is not None:
        t = [start_datetime + timedelta(seconds=i * dt) for i in range(num_samples)]
    else:
        t = np.arange(0, num_samples * dt, dt)

    # Downsample
    t_down = t[::downsample_factor]
    raw_down = pic_values[::downsample_factor]
    filtered_down = filtered_values[::downsample_factor]
    mask_down = mask[::downsample_factor]

    clean_signal = np.array(pic_values, dtype=float)
    clean_signal[mask == 0] = np.nan
    clean_down = clean_signal[::downsample_factor]

    folder_name = os.path.basename(os.path.dirname(file_path))

    # --- Subplots ---
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.4, 0.2, 0.4],
        subplot_titles=[
            "Original Signal (red) vs Filtered (green)",
            "Binary Mask (blue)",
            "Clean Signal (valid data only)"
        ]
    )

    # Row 1: raw and filtered
    fig.add_trace(go.Scattergl(x=t_down, y=raw_down, mode="lines",
                               line=dict(color="red", width=1), name="Original"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=t_down, y=filtered_down, mode="lines",
                               line=dict(color="green", width=1.5), name="Filtered"), row=1, col=1)

    # Row 2: mask
    fig.add_trace(go.Scattergl(x=t_down, y=mask_down, mode="lines",
                               line=dict(color="blue", width=1), name="Mask"), row=2, col=1)

    # Row 3: clean signal
    fig.add_trace(go.Scattergl(x=t_down, y=clean_down, mode="lines",
                               line=dict(color="darkgreen", width=1.5), name="Clean Signal"), row=3, col=1)

    # --- Comments ---
    for comment_time, comment_text in df_comments.itertuples(index=False):
        if comment_time < num_samples:
            comment_x = (start_datetime + timedelta(seconds=comment_time * dt)) if start_datetime else comment_time * dt
            for row in [1, 2, 3]:
                fig.add_vline(x=comment_x, line=dict(color="purple", width=1, dash="dot"), row=row, col=1)
            fig.add_annotation(
                x=comment_x, y=75,
                text=comment_text,
                showarrow=False,
                font=dict(size=9, color="purple"),
                xref="x", yref="y1"
            )

    # --- Initial zoom (first 10 minutes) ---
    initial_range = [t_down[0], t_down[min(120 * Fs // downsample_factor, len(t_down)-1)]]

    # --- Layout ---
    fig.update_layout(
        title=f"PIC Signal Analysis with Artifact Detection - {folder_name}",
        width=1200,
        height=900,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100, b=80, l=80, r=80),
        xaxis3=dict(
            range=initial_range,
            rangeslider=dict(visible=True, thickness=0.05),
            fixedrange=False,
            type="date"
        ),
        yaxis=dict(range=[-20, 80], fixedrange=False),
        yaxis2=dict(range=[0, 1], fixedrange=False),
        yaxis3=dict(range=[-20, 80], fixedrange=False),
        annotations=[
            dict(
                text="Double-click on the graph to reset the scale.",
                xref="paper", yref="paper",
                x=1, y=1.2,
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="right",
                yanchor="top"
            )
        ]
    )

    # --- Axes ---
    fig.update_yaxes(title_text="Pressure (mmHg)", row=1, col=1)
    fig.update_yaxes(title_text="Mask", tickvals=[0, 1], ticktext=["Artifact", "Valid"], row=2, col=1)
    fig.update_yaxes(title_text="Pressure (mmHg)", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    # --- Show ---
    fig.show(config={
        "scrollZoom": True,
        "displayModeBar": False
    })



#---------------------------------------------------------------------------------------------------------------------------------


def print_artifact_removal_summary(pic_values, mask, file_path, Fs=200):
    """
    Prints a summary of the signal duration before and after artifact removal.

    Parameters:
        pic_values (array-like): Original PIC signal.
        mask (array-like): Binary mask where 1 = valid, 0 = artifact.
        file_path (str): Path to the file, used to extract patient name.
        Fs (int): Sampling frequency in Hz (default: 200).
    """

    # --- Compute durations in seconds ---
    total_duration_sec = len(pic_values) / Fs               # Total duration of the raw signal
    valid_duration_sec = sum(mask) / Fs                    # Duration after removing artifacts
    artifact_duration_sec = total_duration_sec - valid_duration_sec  # Time lost to artifacts

    # --- Helper function to convert seconds to H:M:S format ---
    def format_duration(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m}min {s}s"

    # --- Extract patient name from folder path ---
    patient_name = os.path.basename(os.path.dirname(file_path))

    # --- Print formatted summary ---
    print(f"--- Artifact Removal Summary for {patient_name} ---")
    print(f"Total raw signal duration:     {format_duration(total_duration_sec)}")
    print(f"Valid signal after filtering:  {format_duration(valid_duration_sec)}")
    print(f"Total artifact duration:       {format_duration(artifact_duration_sec)}")


#---------------------------------------------------------------------------------------------------------------------------------


def export_filtered_signal_to_txt_and_csv(file_path, filtered_signal, output_txt_name="filtered_signal.txt", output_csv_name="filtered_signal.csv"):
    """
    Exports the filtered PIC signal with NaNs to a TXT and CSV file.
    Preserves the original header and replaces the PIC column with the filtered one.
    The output filenames include the folder name (patient ID).

    Parameters:
        file_path (str): Path to the original .txt file used as input.
        filtered_signal (array-like): Signal with NaNs where artifacts were removed.
        output_txt_name (str): Base name for the TXT file (prefix added with folder name).
        output_csv_name (str): Base name for the CSV file (prefix added with folder name).
    """

    # --- Read the original LabChart-exported text file ---
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # --- Extract header section up to "BottomValue=" ---
    header_lines = []
    data_lines_start = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith("BottomValue="):
            header_lines = lines[:idx + 1]  # Include the "BottomValue=" line
            data_lines_start = idx + 1      # Start of actual data
            break

    data_lines = lines[data_lines_start:]

    # --- Prepare columns for new output file ---
    time_col = []          # Time strings (e.g., "00:00:01.000")
    date_col = []          # Date strings (e.g., "24/03/2025")
    output_values = []     # Filtered PIC values or "NaN"
    filtered_index = 0     # Index for accessing filtered_signal

    for line in data_lines:
        parts = line.strip().split()

        # Ensure line has at least 3 columns and signal index is in bounds
        if len(parts) >= 3 and filtered_index < len(filtered_signal):
            try:
                # Try parsing the third column to confirm it is a valid value
                float(parts[2].replace(",", "."))

                # Append corresponding time and date
                time_col.append(parts[0])
                date_col.append(parts[1])

                # Add filtered value or "NaN" based on mask
                value = filtered_signal[filtered_index]
                if np.isnan(value):
                    output_values.append("NaN")
                else:
                    output_values.append(f"{value:.4f}".replace(".", ","))  # Use comma as decimal

                filtered_index += 1
            except ValueError:
                continue  # Skip lines that are not valid data rows

    # --- Build output filenames using patient folder ---
    output_folder = os.path.dirname(file_path)
    folder_name = os.path.basename(output_folder)

    txt_filename = f"{folder_name}_{output_txt_name}"
    csv_filename = f"{folder_name}_{output_csv_name}"

    txt_path = os.path.join(output_folder, txt_filename)
    csv_path = os.path.join(output_folder, csv_filename)

    # --- Save new TXT file with original header and filtered data ---
    with open(txt_path, "w", encoding="utf-8") as f_out:
        for line in header_lines:
            # Make sure each line ends with a newline character
            f_out.write(line if line.endswith("\n") else line + "\n")
        for t, d, v in zip(time_col, date_col, output_values):
            f_out.write(f"{t} {d} {v}\n")

    # --- Save clean CSV file (easier to load in pandas) ---
    df_csv = pd.DataFrame({
        "Time": time_col,
        "Date": date_col,
        "Filtered_PIC": [np.nan if v == "NaN" else float(v.replace(",", ".")) for v in output_values]
    })

    df_csv.to_csv(csv_path, index=False, sep=";", encoding="utf-8")

    print(f"Exported files:\n- TXT with header: {txt_path}\n- CSV clean: {csv_path}")


#-----------------------------------------------------------------------------------------------------------------------------

def save_dual_pic_plot_to_html(pic_values, filtered_pic_values, df_comments, file_path, Fs=200, start_time=None,
                               primary_threshold=None, secondary_threshold=None):
    """
    Generates an interactive HTML file with two synchronized plots:
    - Filtered PIC signal
    - Raw vs Filtered PIC signal (comparison)
    Includes comment markers, zoom & pan, and uses explicitly passed thresholds.
    
    Parameters:
        pic_values (array-like): Raw PIC signal.
        filtered_pic_values (array-like): Filtered PIC signal with NaNs where artifacts were removed.
        df_comments (DataFrame): DataFrame with columns ["Line", "Comment"].
        file_path (str): Original file path to determine the output name and folder.
        Fs (int): Sampling frequency (default 200 Hz).
        start_time (datetime): Start time of the recording (used for x-axis).
        primary_threshold (float): Threshold for primary artifact detection (optional).
        secondary_threshold (float): Threshold for secondary artifact detection (optional).
    """
      # --- Time setup ---
    dt = 1 / Fs
    num_samples = len(pic_values)
    downsample_factor = 20

    if start_time:
        time_vector = [start_time + timedelta(seconds=i * dt) for i in range(num_samples)]
    else:
        time_vector = np.arange(0, num_samples * dt, dt)

    t_down = time_vector[::downsample_factor]
    raw_down = pic_values[::downsample_factor]
    filtered_down = filtered_pic_values[::downsample_factor]

    folder_name = os.path.basename(os.path.dirname(file_path))
    output_path = os.path.join(os.path.dirname(file_path), f"{folder_name}_dual_pic_plot.html")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            "Filtered PIC Signal (Artifacts Removed)",
            "Raw PIC vs Filtered PIC Comparison"
        ]
    )

    # --- Top plot: filtered signal ---
    fig.add_trace(go.Scatter(
        x=t_down,
        y=filtered_down,
        mode="lines",
        name="Filtered PIC",
        line=dict(color="green", width=1.5)
    ), row=1, col=1)

    # --- Bottom plot: raw vs filtered ---
    fig.add_trace(go.Scatter(
        x=t_down,
        y=raw_down,
        mode="lines",
        name="Raw PIC",
        line=dict(color="red", width=1.2)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t_down,
        y=filtered_down,
        mode="lines",
        name="Filtered PIC",
        line=dict(color="green", width=1.5)
    ), row=2, col=1)

    # --- Thresholds (bottom plot only) ---
    if primary_threshold is not None:
        fig.add_trace(go.Scatter(
            x=[t_down[0], t_down[-1]],
            y=[primary_threshold] * 2,
            mode="lines",
            name=f"Primary Threshold ({primary_threshold:.1f} mmHg)",
            line=dict(color="orange", dash="dash")
        ), row=2, col=1)

    if secondary_threshold is not None:
        fig.add_trace(go.Scatter(
            x=[t_down[0], t_down[-1]],
            y=[secondary_threshold] * 2,
            mode="lines",
            name=f"Secondary Threshold ({secondary_threshold:.1f} mmHg)",
            line=dict(color="purple", dash="dot")
        ), row=2, col=1)

    # --- Comments on both plots ---
    for _, row_data in df_comments.iterrows():
        comment_sec = row_data["Line"] * dt
        if comment_sec < num_samples:
            comment_time = start_time + timedelta(seconds=comment_sec) if start_time else comment_sec
            for row_idx in [1, 2]:
                fig.add_trace(go.Scatter(
                    x=[comment_time, comment_time],
                    y=[-20, 80],
                    mode="lines",
                    line=dict(color="blue", width=1, dash="dash"),
                    showlegend=False
                ), row=row_idx, col=1)
            fig.add_annotation(
                x=comment_time,
                y=75,
                text=row_data["Comment"],
                showarrow=False,
                font=dict(size=10, color="blue"),
                xref="x1", yref="y1"
            )

    # --- Layout ---
    initial_range = [t_down[0], t_down[0] + timedelta(seconds=600)] if start_time else [0, 10 / 60]
    fig.update_layout(
        title=f"Interactive PIC Signal Visualization – {folder_name}",
        width=1600,
        height=1000,
        hovermode="x unified",
        dragmode="pan",
        margin=dict(t=100, b=80, l=80, r=80),
        xaxis2=dict(
            range=initial_range,
            rangeslider=dict(visible=True, thickness=0.02, bgcolor="lightgray", bordercolor="gray"),
            fixedrange=False,
            type="date" if start_time else "linear"
        ),
        yaxis=dict(range=[-20, 80], fixedrange=False),
        yaxis2=dict(range=[-20, 80], fixedrange=False),
        legend=dict(
            orientation="v",
            xanchor="left",
            x=1.02,
            y=1,
            font=dict(size=11)
        ),
        annotations=[dict(
            text="Double-click to reset zoom",
            xref="paper", yref="paper",
            x=1, y=1.18,
            showarrow=False,
            font=dict(size=12, color="gray"),
            xanchor="right", yanchor="top"
        )]
    )

    # --- Save as HTML ---
    fig.write_html(output_path, config={
        "scrollZoom": True,
        "displayModeBar": False
    })

    print(f"Interactive HTML plot saved to:\n{output_path}")

# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS USED IN SIGNAL FRAGMENTS ANALYSIS
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------











