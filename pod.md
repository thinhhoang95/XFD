bash -c '[ ! -d XFD ] && git clone https://github.com/thinhhoang95/XFD.git && cd XFD && chmod +x setup.sh && ./setup.sh && /start.sh'
10GB disk space
Expose HTTP 8888 [or use the template]

jupyter lab --NotebookApp.allow_origin='https://d6tbecq8u20oei-8888.proxy.runpod.net' --no-browser --ip 0.0.0.0