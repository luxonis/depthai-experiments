DATASET=[path to dataset]

# First run mjpeg lossess once
python main.py -p $DATASET --codec jpeg --lossless -nf 30 --mask

# Run 10 iterations for lossy jpeg with ranging the quality
for i in {0..100..10}
do
    python main.py -p $DATASET --codec jpeg -nf 10 --mask --quality $i
done

# Run 10 iterations for h264
for i in {1000..10000..1000}
do
    python main.py -p $DATASET --codec h264 -nf 30 --mask --bitrate $i
done

# Run 10 iterations for h265
for i in {1000..10000..1000}
do
    python main.py -p $DATASET --codec h265 -nf 30 --mask --bitrate $i
done

