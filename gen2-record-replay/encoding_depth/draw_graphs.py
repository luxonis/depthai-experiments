import pandas as pd
import matplotlib.pyplot as plt
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("-csv", "--csv_path", required=True, help="Path to the csv file")
args = argparser.parse_args()

# Read data from csv into a DataFrame
df = pd.read_csv(args.csv_path)


# Filter data for each codec
df_jpeg = df[df['codec'] == 'jpeg']
df_jpeg_lossless = df[df['codec'] == 'jpeg_lossless']
df_h264 = df[df['codec'] == 'h264']
df_h265 = df[df['codec'] == 'h265']

# Plot each codec's bitrate vs psnr
y_labels =  ['PSNR (dB)', 'Relative Fillrate', 'Average Difference in Subpixels', 'MSE']
titles = ['PSNR vs compression ratio', 'Relative fillrate vs compression ratio', 'Average difference in subpixels vs compression ratio', 'MSE vs compression ratio']
for i, name in enumerate(['psnr', 'relative_fillrate', 'average_difference', 'mse']):
    plt.figure(figsize=(10, 6))

    plt.plot(df_jpeg['compression_ratio'], df_jpeg[name], label='JPEG', marker='o')
    plt.plot(df_h264['compression_ratio'], df_h264[name], label='H264', marker='s')
    plt.plot(df_h265['compression_ratio'], df_h265[name], label='H265', marker='x')
    plt.plot(df_jpeg_lossless['compression_ratio'], df_jpeg_lossless[name], label='JPEG Lossless', marker='^')


    plt.xlabel('Compression ratio')
    plt.ylabel(ylabel=y_labels[i])
    plt.title(titles[i])
    plt.legend()
    plt.grid(True)
    # # Save the graph to a file
    # plt.savefig(f'./report_dataset/{name}.png')

    # plt.show()





