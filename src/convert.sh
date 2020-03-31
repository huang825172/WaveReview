#!/usr/bin/zsh

folder='Ori/'
out='Ori_Cover/'
mkdir $out
# shellcheck disable=SC2231
for file in ${folder}*
do
  # shellcheck disable=SC2006
  in_filename=$folder`basename "$file"`
  # shellcheck disable=SC2006
  filename=`basename "$file"`
  realname=${filename%.*}
  out_type=".wav"
  out_filename=$out$realname$out_type
  ffmpeg -i "$in_filename" -acodec pcm_s16le -y "$out_filename"
done
