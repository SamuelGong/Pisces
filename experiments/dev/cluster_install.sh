#!/bin/bash
# should only be executed at the coorindator

# cannot even proceed if not having sudo privilege
if ! groups | grep "\<sudo\>" &> /dev/null; then
   echo "[FAILED] You need to have sudo privilege."
fi

ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)

cwd=`pwd`
PROJECT_DIR=${WORKING_DIR}/../..
cd ${PROJECT_DIR}

sudo apt update
sudo apt install nginx -y

sudo mv /etc/nginx/sites-available/default /etc/nginx/sites-available/default.bak
sudo mv experiments/dev/server_nginx_config.txt /etc/nginx/sites-available/default
sudo mv /etc/nginx/sites-enabled/default .
sudo ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/

sudo nginx -t
sudo systemctl restart nginx

cd ${ORIGINAL_DIR}