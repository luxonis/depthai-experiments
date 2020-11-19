ffmpeg -i im0.png -vf scale="1280:-1",hue=s=0 tmp0.png
ffmpeg -i im1.png -vf scale="1280:-1",hue=s=0 tmp1.png
ffmpeg -i tmp0.png -filter_complex 'extractplanes=r[r]' -map '[r]' tmp2.png
ffmpeg -i tmp1.png -filter_complex 'extractplanes=r[r]' -map '[r]' tmp3.png
ffmpeg -i tmp2.png -filter:v "crop=1280:720:0:70" in_left.png
ffmpeg -i tmp3.png -filter:v "crop=1280:720:0:70" in_right.png
rm tmp0.png tmp1.png tmp2.png tmp3.png
