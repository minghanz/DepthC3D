wget -i splits/kitti_archives_to_download.txt -P kitti_data/
cd kitti_data
unzip "*.zip"
cd ..
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'