FROM ubuntu:16.04
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget
RUN apt-get install -y bzip2

# Setup locale for textract install later
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

# Install dependencies
RUN apt-get install -y python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr \
flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev libpoppler-cpp-dev

# Install Anaconda
RUN wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
RUN bash /Anaconda3-5.1.0-Linux-x86_64.sh -b \
    && echo "export PATH='$HOME/anaconda3/bin:$PATH'" >> ~/.bashrc
ENV PATH="${PATH}:/root/anaconda3/bin"

# Install Textract
RUN pip install pdftotext
RUN pip install textract


