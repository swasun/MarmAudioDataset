import os
import torch
import models
from scipy import signal
import soundfile as sf
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import argparse
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import glob
import re
import datetime
import seaborn as sns
import functools
from threading import Lock


def retreive_date_time_from_fn(fn):
    if 'Recording' in fn:
        # Regular expression pattern to find dates in the format yyyy-mm-dd with optional hour, minute, and second
        date_hour_minute_second_pattern = r'(\d{4})-(\d{2})-(\d{2})(?:_(\d{2})-(\d{2})-(\d{2}))?'

        # Find all matching dates and optional hours, minutes, and seconds in the string
        match = next(re.finditer(date_hour_minute_second_pattern, fn))

        year, month, day, hour, minute, second = match.groups()
        hour, minute, second = int(hour), int(minute), int(second)
        optional_id = None
    else:
        date_pattern = r"(\d{4})_(\d{2})_(\d{2})(?:_(\d+))?"
        # Find all matching dates and optional hours, minutes, and seconds in the string
        match = next(re.finditer(date_pattern, fn))

        year, month, day, optional_id = match.groups()
        hour, minute, second = None, None, None
    year, month, day = int(year), int(month), int(day)

    return year, month, day, hour, minute, second, optional_id

def read_csv_file(file_path):
    return pd.read_csv(file_path, sep='\t')

def process_dataframe(index_df_tuple, args):
    df_id, df = index_df_tuple

    all_file_id = list()
    all_prediction_type = list()
    all_prediction_confidence = list()
    all_year = list()
    all_month = list()
    all_day = list()
    all_hour = list()
    all_minute = list()
    all_second = list()
    all_milliseconds = list()
    all_duration = list()
    all_raw_recording_onset = list()
    all_raw_recording_offset = list()
    all_raw_recording_id = list()

    file_name_id = os.path.splitext(df['fn'].unique()[0])[0]
    output_csv_file_name = os.path.join(args.output_folder_name, file_name_id + '.tsv')
    if os.path.isfile(output_csv_file_name):
        print(f'{output_csv_file_name} already exists')
        return

    df = df[df.pred_label.isin(args.types_to_save)].reset_index(drop=True)
    if len(df) == 0:
        print(f"{file_name_id} doesn't have any prediction with the targeted type")
        return

    df = df[df.pred_conf_SM >= args.conf_threshold_to_save].reset_index(drop=True)
    if len(df) == 0:
        print(f"{file_name_id} doesn't have any prediction above the global threshold")
        return

    """
    infant cry: 0.3-0.4 okish; 0.5 ok
    phee: 0.6 not sure; 0.7 ok
    seep: 0.8 bof; 0.9 tolook
    seep-ek: 0.8 okish; 0.9 tolook
    trill: 0.8 okish, there are screams. tolook higher
    trill-phee: 0.8 too much problem; to remove for now
    tsik: 0.6 60% good? 0.7 good
    tsik-ek: 0.7 too many without ek; 0.8: same;
    twitter: 0.6 mostly ok; 0.7 ok
    """
    if args.manual_conf_threshold:
        if args.species_name == 'marmoset':
            df.loc[((df['pred_label'] == 'Infant_cry') & (df['pred_conf_SM'] < 0.5)), 'pred_label'] = 'Vocalization'
            df.loc[((df['pred_label'] == 'Phee') & (df['pred_conf_SM'] < 0.7)), 'pred_label'] = 'Vocalization'
            df.loc[((df['pred_label'] == 'Seep') & (df['pred_conf_SM'] < 0.86)), 'pred_label'] = 'Vocalization'
            df.loc[((df['pred_label'] == 'Trill') & (df['pred_conf_SM'] < 0.86)), 'pred_label'] = 'Vocalization'
            df.loc[((df['pred_label'] == 'Tsik') & (df['pred_conf_SM'] < 0.7)), 'pred_label'] = 'Vocalization'
            df.loc[((df['pred_label'] == 'Twitter') & (df['pred_conf_SM'] < 0.7)), 'pred_label'] = 'Vocalization'

            df.loc[df['pred_label'] == 'Seep-Ek', 'pred_label'] = 'Vocalization'
            df.loc[df['pred_label'] == 'Trill-Phee', 'pred_label'] = 'Vocalization'
            df.loc[df['pred_label'] == 'Tsik-Ek', 'pred_label'] = 'Vocalization'

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        info = sf.info(os.path.join(args.audio_folder, row['fn']))
        fs = info.samplerate

        if args.dynamic_segmentation or (args.semi_dynamic_segmentation and args.species_name == 'marmoset' and row['pred_label'] in ['Phee', 'Trill-Phee', 'Twitter', 'Tsik']):
            if row['pred_first_offset'] < row['offset'] < row['pred_last_offset']:
                start_time = int(row['pred_first_offset']*fs)
                stop_time = int(row['pred_last_offset']*fs)
                if (stop_time - start_time) < int(args.lensample*fs):
                    stop_time = start_time + int(args.lensample*fs)
            elif row['pred_first_offset'] == row['offset'] == row['pred_last_offset']:
                start_time = int(row['offset']*fs)
                stop_time = int(row['offset']*fs) + int(args.lensample*fs)
                assert((stop_time - start_time) >= args.lensample*fs)
            elif row['pred_first_offset'] == row['offset']:
                start_time = int(row['offset']*fs)
                stop_time = int(row['offset']*fs) + int(args.lensample*fs) if (row['pred_last_offset'] - row['offset']) < args.lensample else int(row['pred_last_offset']*fs)
                assert((stop_time - start_time) >= args.lensample*fs)
            elif row['offset'] == row['pred_last_offset']:
                start_time = int(row['pred_first_offset']*fs)
                stop_time = int(row['pred_first_offset']*fs) + int(args.lensample*fs) if (row['pred_last_offset'] - row['pred_first_offset']) < args.lensample else int(row['pred_last_offset']*fs)
                assert((stop_time - start_time) >= args.lensample*fs)
        else:
            start_time = int(row['offset']*fs)
            stop_time = int(row['offset']*fs) + int(args.lensample*fs)

        if args.manual_offsets:
            if args.species_name == 'marmoset':
                if row['pred_label'] == 'Infant_cry':
                    args.additional_onset = 0.1
                    args.additional_offset = 0.3
                elif row['pred_label'] == 'Phee':
                    args.additional_onset = 0.15
                    args.additional_offset = 0.5
                elif row['pred_label'] == 'Seep':
                    args.additional_offset = 0.1
                elif row['pred_label'] == 'Seep-Ek':
                    args.additional_offset = 0.2
                elif row['pred_label'] == 'Trill':
                    args.additional_offset = 0.4
                elif row['pred_label'] == 'Trill-Phee':
                    args.additional_onset = 0.35
                    args.additional_offset = 0.45
                elif row['pred_label'] == 'Tsik':
                    args.additional_offset = 0.2
                elif row['pred_label'] == 'Tsik-Ek':
                    args.additional_offset = 0.2
                elif row['pred_label'] == 'Twitter':
                    args.additional_onset = 0.05
                    args.additional_offset = 0.5

        start_time -= int(args.additional_onset*fs) if (start_time - int(args.additional_onset*fs)) > 0 else 0
        stop_time += int(args.additional_offset*fs) if (stop_time + int(args.additional_offset*fs)) < int(info.duration*fs) else int(info.duration*fs)

        if start_time >= stop_time:
            stop_time = start_time + int(args.lensample * fs)

        duration = (stop_time - start_time) / fs

        if duration > args.max_duration:
            stop_time = int(start_time + args.max_duration * fs)

        # Ensure stop_time does not exceed the file's total number of samples
        stop_time = min(stop_time, int(info.duration * fs))

        # Check if start_time is equal to the file duration
        if start_time == int(info.duration * fs):
            continue

        duration = (stop_time - start_time) / fs

        year, month, day, hour, minute, second, optional_id = retreive_date_time_from_fn(row['fn'])
        milliseconds = None

        raw_recording_onset = start_time / fs
        raw_recording_offset = stop_time / fs

        sig, fs = sf.read(os.path.join(args.audio_folder, row['fn']),
            start=start_time,
            stop=stop_time,
            always_2d=True)

        sig = sig[:, args.channel]
        if fs != model_fs:
            sig = signal.resample(sig, int(args.lensample * model_fs))

        raw_recording_id = int(optional_id) if optional_id is not None else 1

        if hour is not None:
            dt_recording = datetime.datetime(year, month, day, hour, minute, second)
            delta_onset = datetime.timedelta(seconds=raw_recording_onset)
            dt_onset = dt_recording + delta_onset
            year = dt_onset.year
            month = dt_onset.month
            day = dt_onset.day
            hour = dt_onset.hour
            minute = dt_onset.minute
            second = dt_onset.second
            milliseconds = dt_onset.microsecond // 1000
            datetime_str = f'{str(year).zfill(2)}-{str(month).zfill(2)}-{str(day).zfill(2)}_{str(hour).zfill(2)}-{str(minute).zfill(2)}-{str(second).zfill(2)}-{milliseconds}'
        else:
            datetime_str = f'{str(year).zfill(2)}-{str(month).zfill(2)}-{str(day).zfill(2)}_{raw_recording_onset:.1f}-{raw_recording_offset:.1f}'

        output_file_id = f'{row["pred_label"]}_{raw_recording_id}_{datetime_str}'

        prediction_type = 'Infant cry' if row["pred_label"] == 'Infant_cry' else row["pred_label"]

        all_file_id.append(output_file_id)
        all_prediction_type.append(prediction_type)
        all_prediction_confidence.append(row['pred_conf_SM'])
        all_year.append(year)
        all_month.append(month)
        all_day.append(day)
        all_hour.append(hour)
        all_minute.append(minute)
        all_second.append(second)
        all_milliseconds.append(milliseconds)
        all_duration.append(duration)
        all_raw_recording_onset.append(raw_recording_onset)
        all_raw_recording_offset.append(raw_recording_offset)
        all_raw_recording_id.append(raw_recording_id)

        if not os.path.isfile(f'{args.output_folder_name}/images/{row["pred_label"]}/{output_file_id}.png'):
            spec = frontend(torch.tensor(norm(sig)).to(args.device).float().unsqueeze(0)).cpu().detach().numpy().squeeze()
            pathlib.Path(f'{args.output_folder_name}/images/{row["pred_label"]}').mkdir(parents=True, exist_ok=True)
            sample_length = (stop_time - start_time) / fs
            yl = np.linspace(0, fs / 2, 5).astype(int).tolist()
            xl = np.round(np.linspace(0.0, sample_length, 5), 2).tolist()
            with args.lock:
                fig, ax = plt.subplots()
                ax.imshow(spec, interpolation='nearest', aspect="auto", origin="lower", extent=(0.0, sample_length, 0, fs / 2))
                ax.set_xlabel('Time (s)')
                ax.set(xticks=xl, xticklabels=xl)
                ax.set_ylabel('Frequency (Hz)')
                ax.set(yticks=yl, yticklabels=yl)
                #fig.canvas.draw()
                plt.savefig(f'{args.output_folder_name}/images/{row["pred_label"]}/{output_file_id}.png', bbox_inches='tight', dpi=100)
                plt.close()

        if not os.path.isfile(f'{args.output_folder_name}/audios/{row["pred_label"]}/{output_file_id}.wav'):
            pathlib.Path(f'{args.output_folder_name}/audios/{row["pred_label"]}').mkdir(parents=True, exist_ok=True)
            sf.write(f'{args.output_folder_name}/audios/{row["pred_label"]}/{output_file_id}.wav', sig, fs)

    output_df = pd.DataFrame({
        'file_id': all_file_id,
        'prediction_type': all_prediction_type,
        'prediction_confidence': all_prediction_confidence,
        'year': all_year,
        'month': all_month,
        'day': all_day,
        'hour': all_hour,
        'minute': all_minute,
        'second': all_second,
        'millisecond': all_milliseconds,
        'duration': all_duration,
        'raw_recording_onset': all_raw_recording_onset,
        'raw_recording_offset': all_raw_recording_offset,
        'raw_recording_id': all_raw_recording_id
    })
    output_df.to_csv(output_csv_file_name, sep='\t')

"""
python segment_filtered_predictions.py ../../../../Marmoset/uncut_raw filtered_predictions_lensample0.5_stride0.1_thresh0.0 marmoset_segmented_filtered_predictions_dynamic_and_manual_offsets_2023-05-13 --dynamic_segmentation --manual_offsets --manual_conf_threshold
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_folder', type=str, help='Path of the folder with audio files to process')
    parser.add_argument('prediction_folder', type=str, help='Path of the folder with audio files to process')
    parser.add_argument('output_folder_name', type=str)
    parser.add_argument('-lensample', type=float, help='Length of the signal for each sample (in seconds)', default=.5)
    parser.add_argument('-channel', type=int, help='Channel of the audio file to use in the model inference (starting from 0)', default=0)
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dynamic_segmentation', action='store_true', default=False)
    parser.add_argument('--semi_dynamic_segmentation', action='store_true', default=False)
    parser.add_argument('--additional_onset', type=float, default=0.0)
    parser.add_argument('--additional_offset', type=float, default=0.0)
    parser.add_argument('--manual_offsets', action='store_true', default=False)
    parser.add_argument('--types_to_save', nargs='+', type=str, default=['Phee', 'Trill', 'Seep', 'Twitter', 'Tsik', 'Infant_cry', 'Seep-Ek', 'Trill-Phee', 'Tsik-Ek'])
    parser.add_argument('--conf_threshold_to_save', type=float, default=0.1)
    parser.add_argument('--plot_prediction_hist', action='store_true', default=False)
    parser.add_argument('--manual_conf_threshold', action='store_true', default=False)
    parser.add_argument('--max_duration', type=float, default=2.5)
    parser.add_argument('--species_name', type=str, default='marmoset')
    args = parser.parse_args()

    device = torch.device(args.device)
    frontend = models.get[f'frontend_logMel_{args.sampling_rate_khz}'].to(device)

    norm = lambda arr: (arr - np.mean(arr) ) / np.std(arr)
    if args.sampling_rate_khz == '96':
        model_fs = 96_000
    elif args.sampling_rate_khz == '44_1':
        model_fs = 44_100

    file_paths = glob.glob(os.path.join(args.prediction_folder, '*.csv'))
    dfs = thread_map(read_csv_file, tqdm(file_paths))

    if args.plot_prediction_hist:
        df = pd.concat(dfs).reset_index(drop=True)
        df = df[df.pred_label.isin(args.types_to_save)].reset_index(drop=True)
        df = df[df.pred_conf_SM >= args.conf_threshold_to_save].reset_index(drop=True)

        if args.species_name == 'marmoset':
            df.loc[((df['pred_label'] == 'Infant_cry') & (df['pred_conf_SM'] < 0.5)), 'pred_label'] = 'Vocalization'
            df.loc[((df['pred_label'] == 'Phee') & (df['pred_conf_SM'] < 0.7)), 'pred_label'] = 'Vocalization'
            df.loc[((df['pred_label'] == 'Seep') & (df['pred_conf_SM'] < 0.86)), 'pred_label'] = 'Vocalization'
            df.loc[((df['pred_label'] == 'Trill') & (df['pred_conf_SM'] < 0.86)), 'pred_label'] = 'Vocalization'
            df.loc[((df['pred_label'] == 'Tsik') & (df['pred_conf_SM'] < 0.7)), 'pred_label'] = 'Vocalization'
            df.loc[((df['pred_label'] == 'Twitter') & (df['pred_conf_SM'] < 0.7)), 'pred_label'] = 'Vocalization'

            df.loc[df['pred_label'] == 'Seep-Ek', 'pred_label'] = 'Vocalization'
            df.loc[df['pred_label'] == 'Trill-Phee', 'pred_label'] = 'Vocalization'
            df.loc[df['pred_label'] == 'Tsik-Ek', 'pred_label'] = 'Vocalization'

        print(df)
        print(df.pred_label.value_counts())
        print(len(df))
        print(len(df.fn.unique()))
        for target_type in args.types_to_save:
            df2 = df[df.pred_label == target_type]
            sns.histplot(data=df2, x="pred_conf_SM")
            plt.title(f'Hist pred conf of {target_type}')
            plt.xlabel('Pred confidence')
            plt.savefig(f'hist_pred_conf_SM_{target_type}.png', bbox_inches='tight', dpi=100)
            plt.close()
        exit(0)

    pathlib.Path(args.output_folder_name).mkdir(parents=True, exist_ok=True)
    args.lock = Lock()

    enumerated_dfs = [(index, df) for index, df in enumerate(tqdm(dfs))]
    thread_map(functools.partial(process_dataframe, args=args), enumerated_dfs)
