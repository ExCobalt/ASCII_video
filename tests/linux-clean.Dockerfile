FROM ubuntu:22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    xvfb xauth x11-utils xdotool imagemagick libx11-6 libxext6 libxrender1 libxft2 \
    libfontconfig1 libxss1 libasound2 libasound2-plugins pulseaudio ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY dist/linux/ascii-video /opt/ascii-video
COPY tests/check_linux_packaged.sh /check.sh
COPY tests/check_linux_window.sh /check-window.sh
RUN chmod +x /opt/ascii-video/ascii-video /check.sh
ENTRYPOINT ["bash", "/check.sh"]
