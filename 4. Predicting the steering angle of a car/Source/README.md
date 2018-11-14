Dataset is https://drive.google.com/file/d/1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B/view

Create movie from saved images:

`ffmpeg -framerate 10 -pattern_type glob -i '*.jpg' -c:v libx264 -r 10  -y -an out.mp4`
