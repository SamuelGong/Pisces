#!/bin/bash

pip install -q boto3
pip install -q paramiko
ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)
DEV_DIR=${WORKING_DIR}'/../dev'

EC2_CONFIG=${WORKING_DIR}'/ec2_cluster_config.yml'
EC2_LAUNCH_RESULT=${WORKING_DIR}'/ec2_launch_result.json'
EC2_NODE_TEMPLATE=${WORKING_DIR}'/ec2_node_template.yml'
LAST_RESPONSE=${WORKING_DIR}'/last_response.json'
EC2_HANDLER=${DEV_DIR}'/ec2.py'

case "$1" in
    launch)
        python ${EC2_HANDLER} launch ${EC2_CONFIG} ${EC2_NODE_TEMPLATE} ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    start)
        python ${EC2_HANDLER} start ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    stop)
        python ${EC2_HANDLER} stop ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    terminate)
        python ${EC2_HANDLER} terminate ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    reboot)
        python ${EC2_HANDLER} reboot ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    show)
        python ${EC2_HANDLER} show ${EC2_LAUNCH_RESULT}
        ;;
    *)
        echo "Unknown command!"
        ;;
esac

cd ${ORIGINAL_DIR}