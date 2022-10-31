#!/bin/bash

EC2_LAUNCH_RESULT='./ec2_launch_result.json'
LOCAL_PRIVATE_KEY=${HOME}'/.ssh/MyKeyPair.pem'
BATCH_PLAN='./batch_plan.txt'
BATCH_HANDLER='./batch_run.py'
LOG_FILE='./batch_log.txt'

python ${BATCH_HANDLER} start_tasks ${EC2_LAUNCH_RESULT} \
    ${LOCAL_PRIVATE_KEY} ${BATCH_PLAN} > ${LOG_FILE} 2>&1 &
