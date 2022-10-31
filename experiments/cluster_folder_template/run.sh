#!/bin/bash

ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)
DEV_DIR=${WORKING_DIR}'/../dev'

PISCES_HANDLER=${DEV_DIR}'/pisces_related.py'
EC2_LAUNCH_RESULT=${WORKING_DIR}'/ec2_launch_result.json'
LAST_RESPONSE_REL=${WORKING_DIR}'/last_response.json'

# please change this value that suits you
LOCAL_PRIVATE_KEY=${HOME}'/.ssh/MyKeyPair.pem'

case "$1" in
    start_a_task)
        CONFIG_PATH=${WORKING_DIR}/$2
        START_TIME=$(date "+%Y%m%d-%H%M%S")
        bash setup.sh start_cluster
        python ${PISCES_HANDLER} remote_start ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE_REL} \
          ${START_TIME} ${CONFIG_PATH}
        ;;
    kill_a_task)
        TASK_FOLDER=${WORKING_DIR}/$2
        python ${PISCES_HANDLER} remote_kill ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE_REL} ${TASK_FOLDER}
        ;;
    conclude_a_task)
        TASK_FOLDER=${WORKING_DIR}/$2
        python ${PISCES_HANDLER} collect_result ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE_REL} ${TASK_FOLDER}
        ;;
    *)
      echo "Unknown command!"
      ;;
esac

cd ${ORIGINAL_DIR}