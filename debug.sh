source ../activate_conda.sh

exec python -m debugpy --listen 127.0.0.1:5678 --wait-for-client "$@"